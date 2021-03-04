from sys import setrecursionlimit
import numpy as np 
from matplotlib import pyplot as plt

from ModelsRL import TD3
import csv

import LibFunctions as lib
from mapping import PreMap




class BaseMod:
    def __init__(self, conf, agent_name) -> None:
        self.conf = conf
        self.name = agent_name
        self.path_name = None

        mu = conf.mu
        self.m = conf.m
        g = conf.g
        safety_f = conf.force_f
        self.f_max = mu * self.m * g #* safety_f
        self.max_v = conf.max_v
        self.max_d = conf.max_steer
        self.lookahead = conf.lookahead
        self.vgain = conf.v_gain
        self.wheelbase =  conf.l_f + conf.l_r

        self.wpts = None
        self.vs = None
        self.ss = None

        self.mod_history = []
        self.d_ref_history = []
        self.reward_history = []
        self.critic_history = []

        self.loop_counter = 0
        self.plan_f = conf.plan_frequency
        self.action = None

        try:
            # raise FileNotFoundError
            self._load_csv_track()
        except FileNotFoundError:
            print(f"Problem Loading map - generate new one")

    def _load_csv_track(self):
        track = []
        filename = 'maps/' + self.conf.map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.N = len(track)
        self.ss = track[:, 0]
        self.wpts = track[:, 1:3]
        self.vs = track[:, 5]

        self.expand_wpts()

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

        # self.plot_track_pts()

    def plot_track_pts(self):
        # plt.imshow(self.)
        plt.figure(1)
        plt.plot(self.wpts[:, 0], self.wpts[:, 1], 'x')
        plt.plot(self.wpts[0, 0], self.wpts[0, 1], '+', markersize=20)
        plt.gca().set_aspect('equal', 'datalim')
        # plt.show()

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.wpts
        o_ss = self.ss
        o_vs = self.vs
        new_line = []
        new_ss = []
        new_vs = []
        for i in range(self.N-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                ds = o_ss[i+1] - o_ss[i]
                new_ss.append(o_ss[i] + dz*j*ds)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        self.wpts = np.array(new_line)
        self.ss = np.array(new_ss)
        self.vs = np.array(new_vs)
        self.N = len(new_line)

    def _get_current_waypoint(self, position):
        # nearest_pt, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, self.wpts)
        nearest_pt, nearest_dist, t, i = self.nearest_pt(position)

        if nearest_dist < self.lookahead:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, self.lookahead, self.wpts, i+t, wrap=True)
            if i2 == None:
                return None
            i = i2
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = self.wpts[i2]
            # speed
            current_waypoint[2] = self.vs[i]
            return current_waypoint
        elif nearest_dist < 20:
            return np.append(self.wpts[i], self.vs[i])

    def act_pp(self, obs):
        ego_idx = obs['ego_idx']
        pose_th = obs['poses_theta'][ego_idx] 
        p_x = obs['poses_x'][ego_idx]
        p_y = obs['poses_y'][ego_idx]
        v_current = obs['linear_vels_x'][ego_idx]

        if v_current < self.conf.v_min_plan:
            return 0, 7

        pos = np.array([p_x, p_y], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos)

        if lookahead_point is None:
            return 0.0, 4.0

        speed, steering_angle = self.get_actuation(pose_th, lookahead_point, pos)
        speed = self.vgain * speed 

        return steering_angle, speed

    def reset_lap(self):
        self.mod_history = []

    def get_actuation(self, pose_theta, lookahead_point, position):
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
        
        speed = lookahead_point[2]
        if np.abs(waypoint_y) < 1e-6:
            return speed, 0.
        radius = 1/(2.0*waypoint_y/self.lookahead**2)
        steering_angle = np.arctan(self.wheelbase/radius)

        return speed, steering_angle

    def nearest_pt(self, point):
        dots = np.empty((self.wpts.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - self.wpts[i, :]), self.diffs[i, :])
        t = dots / self.l2s

        t = np.clip(dots / self.l2s, 0.0, 1.0)
        projections = self.wpts[:-1,:] + (t*self.diffs.T).T
        dists = np.linalg.norm(point - projections, axis=1)

        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

    def show_vehicle_history(self):
        plt.figure(1)
        plt.clf()
        plt.title("Mod History")
        plt.ylim([-1.1, 1.1])
        plt.plot(self.mod_history)
        np.save('Vehicles/mod_hist', self.mod_history)
        # plt.plot(self.d_ref_history)
        plt.legend(['NN'])

        plt.pause(0.001)

        # plt.figure(3)
        # plt.clf()
        # plt.title('Rewards')
        # plt.ylim([-1.5, 4])
        # plt.plot(self.reward_history, 'x', markersize=12)
        # plt.plot(self.critic_history)

    def transform_obs(self, obs):
        ego_idx = obs['ego_idx']
        v_current = obs['linear_vels_x'][ego_idx]
        d_current = obs['linear_vels_x'][ego_idx]

        steer_ref, speed_ref = self.act_pp(obs)

        cur_v = [v_current/self.max_v]
        cur_d = [d_current/self.max_d]
        vr_scale = [(speed_ref)/self.max_v]
        dr_scale = [steer_ref/self.max_d]

        scan = np.array(obs['scans'][ego_idx])
        scan_scale = 10 # TODO: move to parameter file
        scan = np.clip(scan/scan_scale, 0, 1)

        nn_obs = np.concatenate([cur_v, cur_d, vr_scale, dr_scale, scan])

        return nn_obs, steer_ref, speed_ref

    def modify_references(self, nn_action, d_ref):
        d_max = self.max_d
        d_phi = d_max * nn_action[0] # rad
        d_new = d_ref + d_phi

        return d_new

    def act(self, obs):
        if self.action is None or self.loop_counter == self.plan_f:
            self.loop_counter = 0
            self.action = self.act_nn(obs)
        self.loop_counter += 1
        return self.action


# @njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.
    Assumes that the first segment passes within a single radius of the point
    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

    # print min_dist_segment, dists[min_dist_segment], projections[min_dist_segment]




class ModVehicleTrain(BaseMod):
    def __init__(self, conf, name, load=False):
        BaseMod.__init__(self, conf, name)

        state_space = 4 + conf.n_beams
        self.agent = TD3(state_space, 1, 1, name)
        h_size = conf.h
        self.agent.try_load(load, h_size)

        self.reward_fcn = None
        self.state = None
        self.nn_state = None
        self.nn_act = None

    def set_reward_fcn(self, r_fcn):
        self.reward_fcn = r_fcn

    def act_nn(self, obs):
        nn_obs, steer_ref, speed_ref = self.transform_obs(obs)
        self.add_memory_entry(obs, nn_obs)

        self.state = obs
        nn_action = self.agent.act(nn_obs)
        self.nn_act = nn_action

        self.d_ref_history.append(steer_ref)
        self.mod_history.append(self.nn_act[0])
        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.nn_state = nn_obs

        steering_angle = self.modify_references(self.nn_act, steer_ref)

        return np.array([[steering_angle, speed_ref]])

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.state is not None:
            reward = self.reward_fcn(self.state, self.action, s_prime, self.nn_act)

            mem_entry = (self.nn_state, self.nn_act, nn_s_prime, reward, False)

            self.agent.replay_buffer.add(mem_entry)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime, d, v = self.transform_obs(s_prime)
        reward = self.reward_fcn(self.state, self.action, s_prime, self.nn_act)

        mem_entry = (self.nn_state, self.nn_act, nn_s_prime, reward, True)

        self.agent.replay_buffer.add(mem_entry)

class ModVehicleTest(BaseMod):
    def __init__(self, config, name):
        path = 'Vehicles/' + name + ''
        state_space = 4 
        self.agent = TD3(state_space, 1, 1, name)
        self.agent.load(directory=path)

        print(f"NN: {self.agent.actor.type}")

        nn_size = self.agent.actor.l1.in_features
        n_beams = nn_size - 4
        BaseMod.__init__(self, config, name)
        print(f"Agent loaded: {name}")

        self.current_v_ref = None
        self.current_phi_ref = None

    def act_nn(self, obs):
        nn_obs, steer_ref, speed_ref = self.transform_obs(obs)

        self.state = obs
        nn_action = self.agent.act(nn_obs, noise=0)
        self.nn_act = nn_action

        self.d_ref_history.append(steer_ref)
        self.mod_history.append(self.nn_act[0])
        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.nn_state = nn_obs

        steering_angle = self.modify_references(self.nn_act, steer_ref)

        return np.array([[steering_angle, speed_ref]])
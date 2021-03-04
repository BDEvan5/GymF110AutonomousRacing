from types import DynamicClassAttribute
import numpy as np 
from mapping import PreMap
import csv
from numba import njit
from matplotlib import pyplot as plt
import LibFunctions as lib

class TunerCar:
    def __init__(self, conf) -> None:
        self.conf = conf
        self.name = "TunerCar Agent: Following PP references"

        mu = conf.mu
        self.m = conf.m
        g = conf.g
        safety_f = conf.force_f
        self.f_max = mu * self.m * g #* safety_f

        self.wpts = None
        self.vs = None
        self.N = None
        self.ss = None

        self.lookahead = conf.lookahead
        self.vgain = conf.v_gain
        self.wheelbase =  conf.l_f + conf.l_r

        self.loop_counter = 0
        self.plan_f = conf.plan_frequency
        self.action = None

        self.wpt_ys = []
        self.prv_pt = np.array([0, 0, 0])
        self.wpts_followed = []

        try:
            # raise FileNotFoundError
            self._load_csv_track()
        except FileNotFoundError:
            print(f"Problem Loading map - generating")
            raise FileNotFoundError
            # pre_map = PreMap(self.conf)
            # pre_map.run_conversion()
            # self._load_csv_track()

    def _load_csv_track(self):
        track_data = []
        filename = 'maps/' + self.conf.map_name + '_opti.csv'
        
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
    
            for lines in csvFile:  
                track_data.append(lines)

        track = np.array(track_data)
        print(f"Track Loaded: {filename}")

        self.N = len(track)
        self.ss = track[:, 0]
        self.wpts = track[:, 1:3]
        self.vs = track[:, 5]

        self.expand_wpts()

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

        # plt.figure(1)
        # plt.plot(self.wpts[:, 0], self.wpts[:, 1])
        # plt.plot(self.wpts[0, 0], self.wpts[0, 1], 'x', markersize=20)
        # plt.gca().set_aspect('equal', 'datalim')

        # plt.pause(0.0001)

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

        pos = np.array([p_x, p_y], dtype=np.float)

        v_min_plan = 1
        if v_current < v_min_plan:
            return np.array([[0, 7]])

        lookahead_point = self._get_current_waypoint(pos)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = self.get_actuation(pose_th, lookahead_point, pos)
        speed = self.vgain * speed #* 0.8

        d_max = 0.4
        steering_angle = np.clip(steering_angle, -d_max, d_max)
        # print(f"Pose: {pose_th:.3f}, Pt: {lookahead_point[0:3]}, Ang: {ang_vel} --> Str {steering_angle}")

        return np.array([[steering_angle, speed]])

    def act(self, obs):
        # if self.action is None or self.loop_counter == self.plan_f:
        #     self.loop_counter = 0
        #     self.action = self.act_pp(obs)
        # self.loop_counter += 1
        self.action = self.act_pp(obs)
        return self.action

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


@njit(fastmath=False, cache=True)
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



from matplotlib import pyplot as plt
import numpy as np
import csv 


class SimHistory:
    def __init__(self, conf):
        self.conf = conf
        self.positions = []
        self.steering = []
        self.vel_cmds = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []

        self.wpts = None
        self.vs = None


        self.ctr = 0

    def save_history(self):
        pos = np.array(self.positions)
        vel = np.array(self.velocities)
        steer = np.array(self.steering)
        obs = np.array(self.obs_locations)

        d = np.concatenate([pos, vel[:, None], steer[:, None]], axis=-1)

        d_name = 'Vehicles/TrainData/' + f'data{self.ctr}'
        o_name = 'Vehicles/TrainData/' + f"obs{self.ctr}"
        np.save(d_name, d)
        np.save(o_name, obs)

    def reset_history(self):
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []

        self.ctr += 1

    def show_history(self, wait=False):
        self.plot_progress()
        plt.figure()
        plt.clf()
        plt.title("Steer history")
        plt.plot(self.steering)
        plt.pause(0.001)

        plt.figure()
        plt.clf()
        plt.title("Velocity history")
        plt.plot(self.velocities)
        plt.plot(self.vel_cmds)
        if self.vs is not None:
            try:
                r = len(self.vs) / len(self.velocities)
                new_vs = []
                for i in range(len(self.velocities)):
                    new_vs.append(self.vs[int(round(r*i))])
                plt.plot(new_vs)
                plt.legend(['Actual', 'Cmds', 'Planned'])
            except:
                print(f"Problem showing precalc velocities")
                plt.legend(['Actual', 'Cmds'])
        plt.pause(0.001)


        if wait:
            plt.show()

    def plot_progress(self):
        plt.figure(1)
        plt.clf()
        self.plot_wpts()
        poses = np.array(self.positions)
        plt.title('Position History')
        # plt.xlim([-10, 12])
        # plt.ylim([-2, 20])
        plt.plot(poses[:, 0], poses[:, 1])
        # plt.plot(poses[:, 0], poses[:, 1], 'x')
        plt.pause(0.0001)

    def show_forces(self):
        mu = self.conf.mu
        m = self.conf.m
        g = self.conf.g
        l_f = self.conf.l_f
        l_r = self.conf.l_r
        f_max = mu * m * g
        f_long_max = l_f / (l_r + l_f) * f_max

        self.velocities = np.array(self.velocities)
        self.thetas = np.array(self.thetas)

        # divide by time taken for change to get per second
        t = self.config['sim']['timestep'] * self.config['sim']['update_f']
        v_dot = (self.velocities[1:] - self.velocities[:-1]) / t
        oms = (self.thetas[1:] - self.thetas[:-1]) / t

        f_lat = oms * self.velocities[:-1] * m
        f_long = v_dot * m
        f_total = (f_lat**2 + f_long**2)**0.5

        plt.figure(3)
        plt.clf()
        plt.title("Forces (lat, long)")
        plt.plot(f_lat)
        plt.plot(f_long)
        plt.plot(f_total, linewidth=2)
        plt.legend(['Lat', 'Long', 'total'])
        plt.plot(np.ones_like(f_lat) * f_max, '--')
        plt.plot(np.ones_like(f_lat) * f_long_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_long_max, '--')
        plt.pause(0.001)

    def add_step(self, obs, action):
        ego_idx = obs['ego_idx']
        pose_th = obs['poses_theta'][ego_idx] 
        p_x = obs['poses_x'][ego_idx]
        p_y = obs['poses_y'][ego_idx]
        v_current = obs['linear_vels_x'][ego_idx]

        pos = np.array([p_x, p_y], dtype=np.float)

        self.velocities.append(v_current)
        self.positions.append(pos)
        self.steering.append(action[0, 0])
        self.vel_cmds.append(action[0, 1])

    def plot_wpts(self):
        if self.wpts is None:
            self.load_wpts()

        plt.figure(1)
        # plt.plot(self.wpts[:, 0], self.wpts[:, 1])
        plt.plot(self.wpts[:, 0], self.wpts[:, 1], 'x', markersize=12)
        # plt.plot(self.wpts[0, 0], self.wpts[0, 1], 'x', markersize=20)
        plt.gca().set_aspect('equal', 'datalim')

        plt.pause(0.0001)


    def load_wpts(self):
        track_data = []
        filename = 'maps/' + self.conf.map_name + '_opti.csv'
        
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
    
            for lines in csvFile:  
                track_data.append(lines)

        track = np.array(track_data)
        print(f"Track Loaded: {filename}")

        self.wpts = track[:, 1:3]
        self.vs = track[:, 5]

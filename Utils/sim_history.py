from matplotlib import pyplot as plt
import numpy as np


class SimHistory:
    def __init__(self, conf):
        self.conf = conf
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []


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

    def show_history(self, vs=None):
        plt.figure(1)
        plt.clf()
        plt.title("Steer history")
        plt.plot(self.steering)
        plt.pause(0.001)

        plt.figure(2)
        plt.clf()
        plt.title("Velocity history")
        plt.plot(self.velocities)
        if vs is not None:
            r = len(vs) / len(self.velocities)
            new_vs = []
            for i in range(len(self.velocities)):
                new_vs.append(vs[int(round(r*i))])
            plt.plot(new_vs)
            plt.legend(['Actual', 'Planned'])
        plt.pause(0.001)

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

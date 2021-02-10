import LibFunctions as lib
import os, shutil
import csv
import numpy as np
from matplotlib import pyplot as plt


class TrainHistory():
    def __init__(self, agent_name) -> None:
        self.agent_name = agent_name
        self.path = 'Vehicles/' + self.agent_name 

        # training data
        self.lengths = []
        self.rewards = [] 
        self.t_counter = 0 # total steps
        
        # espisode data
        self.ep_counter = 0 # ep steps
        self.ep_reward = 0
        self.ep_rewards = []

        self.init_file_struct()

    def init_file_struct(self):
        if os.path.exists(self.path):
            try:
                os.rmdir(self.path)
            except:
                shutil.rmtree(self.path)
        os.mkdir(self.path)

    def add_step_data(self, new_r):
        self.ep_reward += new_r
        self.ep_rewards.append(new_r)
        self.ep_counter += 1
        self.t_counter += 1 

    def lap_done(self, show_reward=False):
        self.lengths.append(self.ep_counter)
        self.rewards.append(self.ep_reward)

        if show_reward:
            plt.figure(8)
            plt.clf()
            plt.plot(self.ep_rewards)
            plt.plot(self.ep_rewards, 'x', markersize=10)
            plt.title(f"Ep rewards: total: {self.ep_reward:.4f}")
            plt.ylim([-1.1, 1.5])
            plt.pause(0.0001)

        self.ep_counter = 0
        self.ep_reward = 0
        self.ep_rewards = []


    def print_update(self):
        mean = np.mean(self.rewards)
        score = self.rewards[-1]
        print(f"Run: {self.t_counter} --> Score: {score:.2f} --> Mean: {mean:.2f} --> ")
        
        lib.plot(self.rewards, figure_n=2)

    def save_csv_data(self):
        data = []
        for i in range(len(self.rewards)):
            data.append([i, self.rewards[i], self.lengths[i]])
        with open(self.path + '/training_rewards.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

        plt.figure(2)
        plt.savefig(self.path + "/training_rewards.png")


class RewardAnalyser:
    def __init__(self) -> None:
        self.rewards = []
        self.t = 0

    def add_reward(self, new_r):
        self.rewards.append(new_r)
        self.t += 1

    def show_rewards(self, show=False):
        plt.figure(6)
        plt.plot(self.rewards, '-*')
        plt.ylim([-1, 1])
        plt.title('Reward History')
        if show:
            plt.show()

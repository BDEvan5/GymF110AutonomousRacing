from follow_the_gap import FollowTheGap
import gym 
import time 
import numpy as np 

from TunerCar import TunerCar
import LibFunctions as lib
from ModelsRL import ReplayBufferTD3
from HistoryStructs import TrainHistory
from Rewards import TimeReward, CthReward, SteerReward
from AgentMod import ModVehicleTrain, ModVehicleTest
from Utils import sim_history

config_test = 'config_test'

"""Train"""
def TrainVehicle(conf, vehicle, reward, steps=20000):
    path = 'Vehicles/' + vehicle.name
    buffer = ReplayBufferTD3()
    history = sim_history.SimHistory(conf)

    env = gym.make('f110_gym:f110-v0', map=conf.map_name, map_ext=conf.map_ext, num_agents=1)
    map_reset_pt = np.array([[conf.sx, conf.sy, conf.stheta]])
    state, step_reward, done, info = env.reset(map_reset_pt)
    env.render()

    t_his = TrainHistory(vehicle.name)
    print_n = 500

    done = False
    done_counter = 0

    for n in range(steps):
        a = vehicle.act(state)
        for i in range(conf.plan_frequency):
            s_prime, r, done, info = env.step(a)
            if s_prime['collisions'][0] == 1:
                break

        new_r = reward(state, a, s_prime)
        vehicle.add_memory_entry(new_r, done, s_prime, buffer)
        t_his.add_step_data(new_r)

        history.steering.append(a[0, 0])
        history.velocities.append(a[0, 1])

        state = s_prime
        vehicle.agent.train(buffer, 2)
        
        env.render('human_fast')

        if n % print_n == 0 and n > 0:
            t_his.print_update()
            vehicle.agent.save(directory=path)
        
        done_counter += 1

        if done or done_counter>200 or s_prime['collisions'][0] == 1:
            print(f"Done Done Done: Restart")
            done_counter = 0
            # t_his.lap_done(True)
            # vehicle.show_vehicle_history()
            # history.show_history()
            history.reset_history()

            vehicle.reset_lap()
            state, step_reward, done, info = env.reset(map_reset_pt)
            env.render()

            print("Reset")


    vehicle.agent.save(directory=path)
    t_his.save_csv_data()

    print(f"Finished Training: {vehicle.name}")

    return t_his.rewards


def test_vehicle(conf, vehicle):
    
    env = gym.make('f110_gym:f110-v0', map=conf.map_name, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    laptime = 0.0
    start = time.time()

    while not done:
        action = vehicle.act(obs)
        # print(action)
        obs, step_reward, done, info = env.step(action)
        laptime += step_reward
        env.render(mode='human_fast')
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)


def run_multi_test(conf, vehicle, n_tests=10):
    env = gym.make('f110_gym:f110-v0', map=conf.map_name, map_ext=conf.map_ext, num_agents=1)

    for i in range(n_tests):

        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        env.render()

        laptime = 0.0
        start = time.time()
        while not done and laptime < conf.max_time:
            action = vehicle.act(obs)
            # print(action)
            obs, step_reward, done, info = env.step(action)
            laptime += step_reward
            env.render(mode='human_fast')
        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)




def run_tuner_car():

    config_file = "config_test"
    conf = lib.load_config_namespace(config_file)

    vehicle = TunerCar(conf)
    # vehicle = FollowTheGap(conf)

    # test_vehicle(conf, vehicle)
    run_multi_test(conf, vehicle)


def train_mod_time():
    agent_name = "ModTime_test"
    conf = lib.load_config_namespace(config_test)
    vehicle = ModVehicleTrain(conf, agent_name, False)
    reward = TimeReward(conf, 0.06)

    # vehicle = TunerCar(conf)

    TrainVehicle(conf, vehicle, reward, 4000)




if __name__ == "__main__":
    run_tuner_car()

    # train_mod_time()



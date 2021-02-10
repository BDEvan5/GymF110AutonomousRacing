import gym 
import time 
import numpy as np 

from TunerCar import TunerCar, PurePursuitPlanner
import LibFunctions as lib

def test_vehicle(conf, vehicle):
    
    env = gym.make('f110_gym:f110-v0', map=conf.map_name, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()


    laptime = 0.0
    start = time.time()

    while not done:
        action = vehicle.act(obs)
        print(action)
        obs, step_reward, done, info = env.step(action)
        laptime += step_reward
        env.render(mode='human_fast')
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)




def run_tuner_car():
    # work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}

    config_file = "config_test"
    conf = lib.load_config_namespace(config_file)

    vehicle = TunerCar(conf)
    vehicle = PurePursuitPlanner(conf)

    test_vehicle(conf, vehicle)



if __name__ == "__main__":
    run_tuner_car()


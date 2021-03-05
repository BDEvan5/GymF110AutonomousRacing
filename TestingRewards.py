import numpy as np
import csv, yaml
import Rewards as r

import LibFunctions as lib
from LibFunctions import load_config
from matplotlib import pyplot as plt

from AgentMod import ModVehicleTest, ModVehicleTrain
from TunerCar import TunerCar
from follow_the_gap import FollowTheGap



from Testing import TestVehicles, TrainVehicle

config_test = "config_test"




def train_empty():
    agent_name = "ModEmpty_test"
    conf = lib.load_config_namespace(config_test)
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.EmptyReward())

    TrainVehicle(conf, vehicle, 400000)

def train_distance_centerline():
    agent_name = "ModDistCenter_test"
    conf = lib.load_config_namespace(config_test)
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.CenterDistanceReward(conf, 5))

    TrainVehicle(conf, vehicle, 400000)

def train_distance_glbl():
    agent_name = "ModDistGlbl_test"
    conf = lib.load_config_namespace(config_test)
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.RefDistanceReward(conf, 0.5))

    TrainVehicle(conf, vehicle, 400000)

def train_cth_centerline():
    agent_name = "ModCthCenter_test"
    conf = lib.load_config_namespace(config_test)
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.CenterCTHReward(conf, 0.04, 0.004))

    TrainVehicle(conf, vehicle, 400000)

def train_cth_glbl():
    agent_name = "ModCthGlbl_test"
    conf = lib.load_config_namespace(config_test)
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.RefCTHReward(conf, 0.04, 0.004))

    TrainVehicle(conf, vehicle, 400000)

def train_steer():
    agent_name = "ModSteer_test"
    conf = lib.load_config_namespace(config_test)
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.TrackSteerReward(conf, 0.05))

    TrainVehicle(conf, vehicle, 40000)



"""Tests """
def FullTrainRT():
    conf = lib.load_config_namespace(config_test)
    env_name = "porto"
    train_name = "_try1"
    n_train = 1000000
    # n_train = 1000

    # 1) no racing reward
    agent_name = "ModEmpty_" + env_name + train_name
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.EmptyReward())

    TrainVehicle(conf, vehicle, n_train)

    # 2) distance centerline
    agent_name = "ModDistCenter_" + env_name + train_name
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.CenterDistanceReward(conf, 5))

    TrainVehicle(conf, vehicle, n_train)

    # 3) distance glbl
    agent_name = "ModDistGlbl_" + env_name + train_name
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.RefDistanceReward(conf, 0.5))

    TrainVehicle(conf, vehicle, n_train)

    # 4) cth center
    agent_name = "ModCthCenter_" + env_name + train_name
    conf = lib.load_config_namespace(config_test)
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.CenterCTHReward(conf, 0.04, 0.004))

    TrainVehicle(conf, vehicle, n_train)

    # 5) cth glbl
    agent_name = "ModCthGlbl_" + env_name + train_name
    conf = lib.load_config_namespace(config_test)
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.RefCTHReward(conf, 0.04, 0.004))

    TrainVehicle(conf, vehicle, n_train)

    # 6) min steer
    agent_name = "ModSteer_" + env_name + train_name
    conf = lib.load_config_namespace(config_test)
    vehicle = ModVehicleTrain(conf, agent_name, False)
    vehicle.set_reward_fcn(r.TrackSteerReward(conf, 0.05))

    TrainVehicle(conf, vehicle, n_train)

def FullTest():
    conf = lib.load_config_namespace(config_test)

    env_name = "porto"
    train_name = "_try1"
    test_name = "compare_" + env_name + train_name
    # test_name = "compare_NoObs_" + env_name + train_name
    test = TestVehicles(conf, test_name)

    # 1) no racing reward
    agent_name = "ModEmpty_" + env_name + train_name
    vehicle = ModVehicleTest(conf, agent_name)
    test.add_vehicle(vehicle)

    # 2) Distance Centerline
    agent_name = "ModDistCenter_" + env_name + train_name
    vehicle = ModVehicleTest(conf, agent_name)
    test.add_vehicle(vehicle)

    # 3) distance ref
    agent_name = "ModDistGlbl_" + env_name + train_name
    vehicle = ModVehicleTest(conf, agent_name)
    test.add_vehicle(vehicle)

    # 4) CTH center
    agent_name = "ModCthCenter_" + env_name + train_name
    vehicle = ModVehicleTest(conf, agent_name)
    test.add_vehicle(vehicle)

    # 5) CTH ref
    agent_name = "ModCthGlbl_" + env_name + train_name
    vehicle = ModVehicleTest(conf, agent_name)
    test.add_vehicle(vehicle)

    # 6) Steering and Velocity
    agent_name = "ModSteer_"  + env_name + train_name
    vehicle = ModVehicleTest(conf, agent_name)
    test.add_vehicle(vehicle)


    # PP
    vehicle = TunerCar(conf)
    test.add_vehicle(vehicle)

    # FTG
    vehicle = FollowTheGap(conf)
    test.add_vehicle(vehicle)

    # test.run_eval(1, True, add_obs=False, save=True)
    # test.run_eval(10, True, add_obs=True, save=True)
    test.run_eval(1000, True, add_obs=True)
    # test.run_eval(1, True, add_obs=True, save=False)

    # test.run_eval(10, True)



"""Smaller tests"""

def test_ftg():
    # config = load_config(config_med)
    config = load_config(config_rt)

    # vehicle = TunerCar(config)
    vehicle = FollowTheGap(config)

    test = TestVehicles(config, "FTG", 'track')
    test.add_vehicle(vehicle)
    # test.run_eval(10, True, add_obs=False)
    test.run_eval(100, True, add_obs=True)
    # testVehicle(config, vehicle, True, 10)

def test_mod():
    config = load_config(config_rt)
    test = TestVehicles(config, "Mod_test", 'track')
    # agent_name = "ModTime_raceTrack"

    # agent_name = "ModTime_test_rt"
    # agent_name = "ModSteer_test_rt"
    # agent_name = "ModCth_test"

    env_name = "porto"
    train_name = "_fin1"
    # agent_name = "ModTime_" + env_name + train_name
    # agent_name = "ModRefCth_" + env_name + train_name
    agent_name = "ModStrVel_"  + env_name + train_name

    # agent_name = "ModDev_test_rt"
    # agent_name = "ModOld_test_rt"
    # agent_name = "ModStd_test_rt"
    
    # agent_name = "ModEmp_test_rt"
    # agent_name = "ModSteer_test_rt2"

    # agent_name = "ModTime_medForest"
    # agent_name = "ModDev_raceTrack"
    # agent_name = "ModSteer_test_om"
    vehicle = ModVehicleTest(config, agent_name)
    # vehicle = TunerCar(config)


    test.add_vehicle(vehicle)

    # agent_name = "ModStd_test_rt2"
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    # test.run_eval(10, True, add_obs=False)
    # test.run_eval(100, True, add_obs=True, wait=True)
    # test.run_eval(1, show=True, add_obs=False, wait=True)
    # test.run_eval(10, show=True, add_obs=True, wait=True)
    test.run_eval(100, True, add_obs=True, wait=False)
    # test.run_eval(1, True, add_obs=False)
    # plt.show()

def train_test():
    config = load_config(config_rt)

    agent_name = "ModSteer_test_rt2"
    reward = TrackSteerReward(config, 0.005, 0.005)
    


    vehicle = ModVehicleTrain(config, agent_name, load=False)
    TrainVehicle(config, agent_name, vehicle, reward, 100000, 'track', show=True)

    test = TestVehicles(config, "Mod_test", 'track')
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)


    test.run_eval(1, True, add_obs=False)
    test.run_eval(100, True, add_obs=True, wait=False)



def train():
    pass

    # train_empty()
    # train_distance_centerline()
    # train_distance_glbl()
    # train_cth_centerline()
    train_cth_glbl()

    # train_steer()


if __name__ == "__main__":
    # train()

    # test_compare()
    # test_compare_mod()
    # test_time_sweep()
    # test_steer_sweep()

    FullTrainRT()
    FullTest()

    # PartialTrain()
    # PartialTest()

    # train_test()
    # test_mod()
    # test_ftg()

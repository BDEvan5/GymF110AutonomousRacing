from numba.core.decorators import njit
import numpy as np 


class FollowTheGap:
    def __init__(self, conf):
        self.name = "Follow the Gap method"
        self.conf = conf
        self.map = None
        self.cur_scan = None
        self.cur_odom = None
    
        self.max_speed = conf.max_v
        self.max_steer = conf.max_steer
        
    def act(self, obs):
        ranges = np.array(obs['scans'][0], dtype=np.float)
        angle_increment = np.pi / len(ranges)

        max_range = 5

        ranges = preprocess_lidar(ranges, max_range)

        bubble_r = 0.1
        ranges = create_zero_bubble(ranges, bubble_r)
        
        start_i, end_i = find_max_gap(ranges)

        aim = find_best_point(start_i, end_i, ranges[start_i:end_i])

        half_pt = len(ranges) /2
        steering_angle =  angle_increment * (aim - half_pt)

        speed = self.max_speed * ranges[aim] / max_range * 0.7

        return np.array([[steering_angle, speed]])

@njit
def preprocess_lidar(ranges, max_range):
    ranges = np.array([min(ran, max_range) for ran in ranges])
    
    # moving_avg
    # n = 3
    # cumsum = np.cumsum(np.insert(ranges, 0, 0))
    # proc_ranges = (cumsum[n:] - cumsum[:-n])/float(n)

    proc_ranges = ranges

    return proc_ranges

@njit
def create_zero_bubble(input_vector, bubble_r):
    centre = np.argmin(input_vector)
    min_dist = input_vector[centre]
    input_vector[centre] = 0
    size = len(input_vector)

    current_idx = centre
    while(current_idx < size -1 and input_vector[current_idx] < (min_dist + bubble_r)):
        input_vector[current_idx] = 0
        current_idx += 1
    
    current_idx = centre
    while(current_idx > 0  and input_vector[current_idx] < (min_dist + bubble_r)):
        input_vector[current_idx] = 0
        current_idx -= 1

    return input_vector
    
@njit
def find_max_gap(input_vector):
    max_start = 0
    max_size = 0

    current_idx = 0
    size = len(input_vector)

    while current_idx < size:
        current_start = current_idx
        current_size = 0
        while current_idx< size and input_vector[current_idx] > 1:
            current_size += 1
            current_idx += 1
        if current_size > max_size:
            max_start = current_start
            max_size = current_size
            current_size = 0
        current_idx += 1
    if current_size > max_size:
        max_start = current_start
        max_size = current_size

    return max_start, max_start + max_size - 1

@njit  
def find_best_point(start_i, end_i, ranges):
    # return best index to goto
    mid_i = (start_i + end_i) /2
    best_i = np.argmax(ranges)  
    best_i = (mid_i + (best_i + start_i)) /2

    return int(best_i)






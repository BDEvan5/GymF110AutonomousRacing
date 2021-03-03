import numpy as np 

import LibFunctions as lib


def find_closest_pt(pt, wpts):
    """
    Returns the two closes points in order along wpts
    """
    dists = [lib.get_distance(pt, wpt) for wpt in wpts]
    min_i = np.argmin(dists)
    d_i = dists[min_i] 
    if min_i == len(dists) - 1:
        min_i -= 1
    if dists[max(min_i -1, 0) ] > dists[min_i+1]:
        p_i = wpts[min_i]
        p_ii = wpts[min_i+1]
        d_i = dists[min_i] 
        d_ii = dists[min_i+1] 
    else:
        p_i = wpts[min_i-1]
        p_ii = wpts[min_i]
        d_i = dists[min_i-1] 
        d_ii = dists[min_i] 

    return p_i, p_ii, d_i, d_ii

def get_tiangle_h(a, b, c):
    s = (a + b+ c) / 2
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))
    h = 2 * A / c

    return h

def distance_potential(s, s_p, end, beta=0.2, scale=0.5):
    prev_dist = lib.get_distance(s[0:2], end)
    cur_dist = lib.get_distance(s_p[0:2], end)
    d_dis = (prev_dist - cur_dist) / scale

    return d_dis * beta



# Mod
class SteerReward:
    def __init__(self, config, mv, ms) -> None:
        self.max_steer = config['lims']['max_steer']
        self.max_v = config['lims']['max_v']
        self.end = [config['map']['end']['x'], config['map']['end']['y']]
        self.mv = mv 
        self.ms = ms 

    def init_reward(self, pts, vs):
        pass
        
    def __call__(self, s, a, s_p, r, time=0) -> float:
        if r == -1:
            return r
        else:
            shaped_r = distance_potential(s, s_p, self.end)

            vel = a[0] / self.max_v 
            steer = abs(a[1]) / self.max_steer

            new_r = self.mv * vel - self.ms * steer 

            return new_r + r + shaped_r 

class CthReward:
    def __init__(self, config, mh, md) -> None:
        self.mh = mh 
        self.md = md
        self.dis_scale = config['lims']["dis_scale"]
        self.max_v = config['lims']["max_v"]
        self.end = [config['map']['end']['x'], config['map']['end']['y']]

        self.pts = None
        self.vs = None

    def init_reward(self, pts, vs):
        self.pts = pts
        self.vs = vs
            
    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r
        else:
            pt_i, pt_ii, d_i, d_ii = find_closest_pt(s_p[0:2], self.pts)
            d = lib.get_distance(pt_i, pt_ii)
            d_c = get_tiangle_h(d_i, d_ii, d) / self.dis_scale

            th_ref = lib.get_bearing(pt_i, pt_ii)
            th = s_p[2]
            d_th = abs(lib.sub_angles_complex(th_ref, th))
            v_scale = s_p[3] / self.max_v

            shaped_r = distance_potential(s, s_p, self.end)

            new_r =  self.mh * np.cos(d_th) * v_scale - self.md * d_c

            return new_r + r + shaped_r

class TimeReward:
    def __init__(self, conf, mt) -> None:
        self.mt = mt 
        self.dis_scale = conf.r_dis_scale
        # self.end = [config['map']['end']['x'], config['map']['end']['y']]
        self.max_steer = conf.max_steer

        self.ss = None
        self.steer = None
        self.diffs = None
        self.l2s = None

        self.load_map()

    def init_reward(self, pts, vs):
        pass

    def load_map(self):
        track = np.loadtxt('example_waypoints.csv', delimiter=';', skiprows=3)

        self.N = len(track)
        self.ss = track[:, 0]
        self.wpts = track[:, 1:3]

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 


    def find_s(self, point):
        dots = np.empty((self.wpts.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - self.wpts[i, :]), self.diffs[i, :])
        t = dots / self.l2s

        t = np.clip(dots / self.l2s, 0.0, 1.0)
        projections = self.wpts[:-1,:] + (t*self.diffs.T).T
        dists = np.linalg.norm(point - projections, axis=1)

        min_dist_segment = np.argmin(dists)

        s = self.ss[min_dist_segment] + dists[min_dist_segment]

        return s 

    def __call__(self, s, a, s_p) -> float:
        collision = s_p['collisions'][0]
        if collision == 1:
            return -1
        else:
            pos_t = np.array([s['poses_x'][0], s['poses_y'][0]])
            pos_tt = np.array([s_p['poses_x'][0], s_p['poses_y'][0]])


            s = self.find_s(pos_t)
            ss = self.find_s(pos_tt)
            shaped_r = 0.5 * (ss - s) 

            return shaped_r

            # new_r = - self.mt

            # return new_r + shaped_r




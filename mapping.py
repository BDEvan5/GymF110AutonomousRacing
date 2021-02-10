from numpy.core.fromnumeric import clip
from LibFunctions import load_config_namespace
import yaml 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import csv

import LibFunctions as lib
from scipy import ndimage 

import casadi as ca 




class PreMap:
    def __init__(self, conf) -> None:
        self.conf = conf 
        self.map_name = conf.map_name
        self.map_ext = conf.map_ext

        self.map_img = None
        self.origin = None
        self.resolution = None

        self.cline = None
        self.nvecs = None
        self.widths = None

        self.wpts = None
        self.vs = None

    def run_conversion(self):
        self.read_yaml_file()
        self.load_map()

        self.dt = ndimage.distance_transform_edt(self.map_img) 
        self.dt = np.array(self.dt *self.resolution)

        self.find_centerline(False)
        self.find_nvecs()
        self.set_true_widths()

        self.save_map()
        self.run_optimisation_no_obs()
        self.save_map_opti()

        self.render_map()
        
    def read_yaml_file(self):
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)

        self.yaml_file = dict(documents.items())

        self.resolution = self.yaml_file['resolution']
        self.origin = self.yaml_file['origin']

    def load_map(self):

        map_file_name = self.yaml_file['image']
        pgm_name = 'maps/' + map_file_name

        if self.map_ext == '.pgm':
            with open(pgm_name, 'rb') as f:
                codec = f.readline()

            if codec == b"P2\n":
                self.read_p2(pgm_name)
            elif codec == b'P5\n':
                self.read_p5(pgm_name)
            else:
                raise Exception(f"Incorrect format of PGM: {codec}")

        elif self.map_ext == ".png":
            self.read_png()
            # raise NotImplementedError
        else:
            raise ImportError("Map extension is not understood")

        self.obs_map = np.zeros_like(self.map_img)
        print(f"Map size: {self.width * self.resolution}, {self.height * self.resolution}")

    def read_png(self):
        map_img_path = "maps/" + self.map_name + self.map_ext
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        # self.map_img = img.imread(self.map_name+self.map_ext)
        # self.map_img = np.array(self.map_img)
        self.height = self.map_img.shape[1]
        self.width = self.map_img.shape[0]

    #TODO: flip the pgm files that are read top to bottom
    def read_p2(self, pgm_name):
        print(f"Reading P2 maps")
        with open(pgm_name, 'r') as f:
            lines = f.readlines()

        # This ignores commented lines
        for l in list(lines):
            if l[0] == '#':
                lines.remove(l)
        # here,it makes sure it is ASCII format (P2)
        codec = lines[0].strip()

        # Converts data to a list of integers
        data = []
        for line in lines[1:]:
            data.extend([int(c) for c in line.split()])

        data = (np.array(data[3:]),(data[1],data[0]),data[2])
        self.width = data[1][1]
        self.height = data[1][0]

        data = np.reshape(data[0],data[1])

        self.map_img = data
    
    def read_p5(self, pgm_name):
        print(f"Reading P5 maps")
        with open(pgm_name, 'rb') as pgmf:
            assert pgmf.readline() == b'P5\n'
            comment = pgmf.readline()
            # comment = pgmf.readline()
            #TODO: update this to new format in the python package
            wh_line = pgmf.readline().split()
            (width, height) = [int(i) for i in wh_line]
            depth = int(pgmf.readline())
            assert depth <= 255

            raster = []
            for y in range(height):
                row = []
                for y in range(width):
                    row.append(ord(pgmf.read(1)))
                raster.append(row)
            
        self.height = height
        self.width = width
        self.map_img = np.array(raster)        

    def find_centerline(self, show=True):
        self.dt = ndimage.distance_transform_edt(self.map_img)
        dt = np.array(self.dt) 

        d_search = 0.5 
        n_search = 11
        dth = (np.pi * 4/5) / (n_search-1)

        # makes a list of search locations
        search_list = []
        for i in range(n_search):
            th = -np.pi/2 + dth * i
            x = -np.sin(th) * d_search
            y = np.cos(th) * d_search
            loc = [x, y]
            search_list.append(loc)

        pt = start = np.array([self.conf.sx, self.conf.sy])
        self.cline = [pt]
        th = self.conf.stheta - np.pi/2
        while lib.get_distance(pt, start) > d_search or len(self.cline) < 10 and len(self.cline) < 200:
            vals = []
            self.search_space = []
            for i in range(n_search):
                d_loc = lib.transform_coords(search_list[i], -th)
                search_loc = lib.add_locations(pt, d_loc)

                self.search_space.append(search_loc)

                x, y = self.xy_to_row_column(search_loc)
                val = dt[y, x]
                vals.append(val)

            ind = np.argmax(vals)
            d_loc = lib.transform_coords(search_list[ind], -th)
            pt = lib.add_locations(pt, d_loc)
            self.cline.append(pt)

            if show:
                self.plot_raceline_finding()

            th = lib.get_bearing(self.cline[-2], pt)
            print(f"Adding pt: {pt}")

        self.cline = np.array(self.cline)
        self.N = len(self.cline)
        print(f"Raceline found")
        if show:
            self.plot_raceline_finding(True)

    def plot_raceline_finding(self, wait=False):
        plt.figure(1)
        plt.clf()
        plt.imshow(self.dt, origin='lower')

        for pt in self.cline:
            s_x, s_y = self.xy_to_row_column(pt)
            plt.plot(s_x, s_y, '+', markersize=16)

        for pt in self.search_space:
            s_x, s_y = self.xy_to_row_column(pt)
            plt.plot(s_x, s_y, 'x', markersize=12)


        plt.pause(0.001)

        if wait:
            plt.show()

    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        return c, r

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return xs, ys

    def find_nvecs(self):
        N = self.N
        track = self.cline

        nvecs = []
        # new_track.append(track[0, :])
        nvec = lib.theta_to_xy(np.pi/2 + lib.get_bearing(track[0, :], track[1, :]))
        nvecs.append(nvec)
        for i in range(1, len(track)-1):
            pt1 = track[i-1]
            pt2 = track[min((i, N)), :]
            pt3 = track[min((i+1, N-1)), :]

            th1 = lib.get_bearing(pt1, pt2)
            th2 = lib.get_bearing(pt2, pt3)
            if th1 == th2:
                th = th1
            else:
                dth = lib.sub_angles_complex(th1, th2) / 2
                th = lib.add_angles_complex(th2, dth)

            new_th = th + np.pi/2
            nvec = lib.theta_to_xy(new_th)
            nvecs.append(nvec)

        nvec = lib.theta_to_xy(np.pi/2 + lib.get_bearing(track[-2, :], track[-1, :]))
        nvecs.append(nvec)

        self.nvecs = np.array(nvecs)

    def set_true_widths(self):
        # nvecs = self.nvecs
        # tx = self.cline[:, 0]
        # ty = self.cline[:, 1]

        # stp_sze = 0.01
        # sf = 0.5 # safety factor
        # nws, pws = [], []
        # for i in range(self.N):
        #     pt = [tx[i], ty[i]]
        #     nvec = nvecs[i]

        #     j = stp_sze
        #     s_pt = lib.add_locations(pt, nvec, j)
        #     while not self.check_scan_location(s_pt):
        #         j += stp_sze
        #         s_pt = lib.add_locations(pt, nvec, j)
        #     pws.append(j*sf)

        #     j = stp_sze
        #     s_pt = lib.sub_locations(pt, nvec, j)
        #     while not self.check_scan_location(s_pt):
        #         j += stp_sze
        #         s_pt = lib.sub_locations(pt, nvec, j)
        #     nws.append(j*sf)

        # nws, pws = np.array(nws), np.array(pws)

        nws = 1.5 * np.ones(len(self.nvecs))
        pws = 1.5 * np.ones(len(self.nvecs))

        self.widths =  np.concatenate([nws[:, None], pws[:, None]], axis=-1)     

    def render_map(self):
        plt.figure(2)
        plt.clf()
        plt.imshow(self.map_img, origin='lower')

        xs, ys = [], []
        for pt in self.cline:
            s_x, s_y = self.xy_to_row_column(pt)
            xs.append(s_x)
            ys.append(s_y)
        plt.plot(xs, ys, linewidth=2)

        ns = self.nvecs 
        ws = self.widths
        l_line = self.cline - np.array([ns[:, 0] * ws[:, 0], ns[:, 1] * ws[:, 0]]).T
        r_line = self.cline + np.array([ns[:, 0] * ws[:, 1], ns[:, 1] * ws[:, 1]]).T

        cx, cy = self.convert_positions(self.cline)
        plt.plot(cx, cy, '--', linewidth=2)
        lx, ly = self.convert_positions(l_line)
        plt.plot(lx, ly, linewidth=1)
        rx, ry = self.convert_positions(r_line)
        plt.plot(rx, ry, linewidth=1)

        wpt_x, wpt_y = self.convert_positions(self.wpts)
        plt.plot(wpt_x, wpt_y, linewidth=2)

        plt.show()

    def check_scan_location(self, pt):
        c, r = self.xy_to_row_column(pt)
        if abs(c) > self.width -2 or abs(r) > self.height -2:
            return True
        if self.dt[c, r] < 0.05:
            return True
        return False

    def save_map(self):
        filename = 'maps/' + self.map_name + '_std.csv'

        track = np.concatenate([self.cline, self.nvecs, self.widths], axis=-1)

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(track)

        print(f"Track Saved in File: {filename}")

    def run_optimisation_no_obs(self):
        n_set = MinCurvatureTrajectory(self.cline, self.nvecs, self.widths)

        deviation = np.array([self.nvecs[:, 0] * n_set[:, 0], self.nvecs[:, 1] * n_set[:, 0]]).T
        self.wpts = self.cline + deviation

        self.vs = Max_velocity(self.wpts, self.conf, False)
        # self.vs = Max_velocity(self.wpts, self.conf, True)

    def save_map_opti(self):
        filename = 'maps/' + self.map_name + '_opti.csv'

        dss, ths = convert_pts_s_th(self.wpts)
        ss = np.cumsum(dss)
        ks = np.zeros_like(ths[:, None]) #TODO: add the curvature

        track = np.concatenate([ss[:, None], self.wpts[:-1], ths[:, None], ks, self.vs], axis=-1)

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(track)

        print(f"Track Saved in File: {filename}")


def MinCurvatureTrajectory(pts, nvecs, ws):
    """
    This function uses optimisation to minimise the curvature of the path
    """
    w_min = - ws[:, 0] * 0.9
    w_max = ws[:, 1] * 0.9
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

    N = len(pts)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)
    th1_f = ca.MX.sym('y1_f', N-1)
    th2_f = ca.MX.sym('y1_f', N-1)
    th1_f1 = ca.MX.sym('y1_f', N-2)
    th2_f1 = ca.MX.sym('y1_f', N-2)

    o_x_s = ca.Function('o_x', [n_f], [pts[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [pts[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [pts[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [pts[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan2(im(th1_f, th2_f),real(th1_f, th2_f))])
    
    get_th_n = ca.Function('gth', [th_f], [sub_cmplx(ca.pi*np.ones(N-1), sub_cmplx(th_f, th_ns[:-1]))])
    d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])

    # objective
    real1 = ca.Function('real1', [th1_f1, th2_f1], [ca.cos(th1_f1)*ca.cos(th2_f1) + ca.sin(th1_f1)*ca.sin(th2_f1)])
    im1 = ca.Function('im1', [th1_f1, th2_f1], [-ca.cos(th1_f1)*ca.sin(th2_f1) + ca.sin(th1_f1)*ca.cos(th2_f1)])

    sub_cmplx1 = ca.Function('a_cpx1', [th1_f1, th2_f1], [ca.atan2(im1(th1_f1, th2_f1),real1(th1_f1, th2_f1))])
    
    # define symbols
    n = ca.MX.sym('n', N)
    th = ca.MX.sym('th', N-1)

    nlp = {\
    'x': ca.vertcat(n, th),
    'f': ca.sumsqr(sub_cmplx1(th[1:], th[:-1])), 
    # 'f': ca.sumsqr(track_length(n)), 
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th)),

                # boundary constraints
                n[0], #th[0],
                n[-1], #th[-1],
            ) \
    
    }

    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(pts[i, 0:2], pts[i+1, 0:2])
        th0.append(th_00)

    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    lbx = list(w_min) + [-np.pi]*(N-1) 
    ubx = list(w_max) + [np.pi]*(N-1) 

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    n_set = np.array(x_opt[:N])
    # thetas = np.array(x_opt[1*N:2*(N-1)])

    return n_set



"""Find the max velocity """
def Max_velocity(pts, conf, show=False):
    mu = conf.mu
    m = conf.m
    g = conf.g
    l_f = conf.l_f
    l_r = conf.l_r
    safety_f = conf.force_f
    f_max = mu * m * g * safety_f
    f_long_max = l_f / (l_r + l_f) * f_max
    max_v = conf.max_v  
    max_a = conf.max_a

    s_i, th_i = convert_pts_s_th(pts)
    th_i_1 = th_i[:-1]
    s_i_1 = s_i[:-1]
    N = len(s_i)
    N1 = len(s_i) - 1

    # setup possible casadi functions
    d_x = ca.MX.sym('d_x', N-1)
    d_y = ca.MX.sym('d_y', N-1)
    vel = ca.Function('vel', [d_x, d_y], [ca.sqrt(ca.power(d_x, 2) + ca.power(d_y, 2))])

    dx = ca.MX.sym('dx', N)
    dy = ca.MX.sym('dy', N)
    dt = ca.MX.sym('t', N-1)
    f_long = ca.MX.sym('f_long', N-1)
    f_lat = ca.MX.sym('f_lat', N-1)

    nlp = {\
        'x': ca.vertcat(dx, dy, dt, f_long, f_lat),
        'f': ca.sum1(dt), 
        'g': ca.vertcat(
                    # dynamic constraints
                    dt - s_i_1 / ((vel(dx[:-1], dy[:-1]) + vel(dx[1:], dy[1:])) / 2 ),
                    # ca.arctan2(dy, dx) - th_i,
                    dx/dy - ca.tan(th_i),
                    dx[1:] - (dx[:-1] + (ca.sin(th_i_1) * f_long + ca.cos(th_i_1) * f_lat) * dt  / m),
                    dy[1:] - (dy[:-1] + (ca.cos(th_i_1) * f_long - ca.sin(th_i_1) * f_lat) * dt  / m),

                    # path constraints
                    ca.sqrt(ca.power(f_long, 2) + ca.power(f_lat, 2)),

                    # boundary constraints
                    # dx[0], dy[0]
                ) \
    }

    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})
    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})

    # make init sol
    v0 = np.ones(N) * max_v/2
    dx0 = v0 * np.sin(th_i)
    dy0 = v0 * np.cos(th_i)
    dt0 = s_i_1 / ca.sqrt(ca.power(dx0[:-1], 2) + ca.power(dy0[:-1], 2)) 
    f_long0 = np.zeros(N-1)
    ddx0 = dx0[1:] - dx0[:-1]
    ddy0 = dy0[1:] - dy0[:-1]
    a0 = (ddx0**2 + ddy0**2)**0.5 
    f_lat0 = a0 * m

    x0 = ca.vertcat(dx0, dy0, dt0, f_long0, f_lat0)

    # make lbx, ubx
    # lbx = [-max_v] * N + [-max_v] * N + [0] * N1 + [-f_long_max] * N1 + [-f_max] * N1
    lbx = [-max_v] * N + [0] * N + [0] * N1 + [-f_long_max] * N1 + [-f_max] * N1
    ubx = [max_v] * N + [max_v] * N + [10] * N1 + [f_long_max] * N1 + [f_max] * N1

    #make lbg, ubg
    lbg = [0] * N1 + [0] * N + [0] * 2 * N1 + [0] * N1 #+ [0] * 2 
    ubg = [0] * N1 + [0] * N + [0] * 2 * N1 + [f_max] * N1 #+ [0] * 2 

    r = S(x0=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    dx = np.array(x_opt[:N])
    dy = np.array(x_opt[N:N*2])
    dt = np.array(x_opt[2*N:N*2 + N1])
    f_long = np.array(x_opt[2*N+N1:2*N + N1*2])
    f_lat = np.array(x_opt[-N1:])

    f_t = (f_long**2 + f_lat**2)**0.5

    # print(f"Dt: {dt.T}")
    # print(f"DT0: {dt[0]}")
    t = np.cumsum(dt)
    t = np.insert(t, 0, 0)
    # print(f"Dt: {dt.T}")
    # print(f"Dx: {dx.T}")
    # print(f"Dy: {dy.T}")

    vs = (dx**2 + dy**2)**0.5

    if show:
        plt.figure(1)
        plt.title("Velocity vs dt")
        plt.plot(t, vs)
        plt.plot(t, dx)
        plt.plot(t, dy)
        plt.legend(['v', 'dx', 'dy'])
        plt.plot(t, np.ones_like(t) * max_v, '--')

        plt.figure(2)
        plt.title("F_long, F_lat vs t")
        plt.plot(t[:-1], f_long)
        plt.plot(t[:-1], f_lat)
        plt.plot(t[:-1], f_t, linewidth=3)
        plt.plot(t, np.ones_like(t) * f_max, '--')
        plt.plot(t, np.ones_like(t) * -f_max, '--')
        plt.plot(t, np.ones_like(t) * f_long_max, '--')
        plt.plot(t, np.ones_like(t) * -f_long_max, '--')

        plt.legend(['Flong', "f_lat", "f_t"])

        plt.figure(3)
        plt.title("Theta vs t")
        plt.plot(t, th_i)
        # plt.plot(t, np.abs(th_i))

        plt.figure(5)
        plt.title(f"t vs dt")
        plt.plot(t[1:], dt)
        plt.plot(t[1:], dt, '+')
    
    # plt.figure(9)
    # plt.clf()
    # plt.title("F_long, F_lat vs t")
    # plt.plot(t[:-1], f_long)
    # plt.plot(t[:-1], f_lat)
    # plt.plot(t[:-1], f_t, linewidth=3)
    # plt.plot(t, np.ones_like(t) * f_max, '--')
    # plt.plot(t, np.ones_like(t) * -f_max, '--')
    # plt.plot(t, np.ones_like(t) * f_long_max, '--')
    # plt.plot(t, np.ones_like(t) * -f_long_max, '--')
    # plt.legend(['Flong', "f_lat", "f_t"])


    return vs


def convert_pts_s_th(pts):
    N = len(pts)
    s_i = np.zeros(N-1)
    th_i = np.zeros(N-1)
    for i in range(N-1):
        s_i[i] = lib.get_distance(pts[i], pts[i+1])
        th_i[i] = lib.get_bearing(pts[i], pts[i+1])

    return s_i, th_i



if __name__ == "__main__":
    fname = "config_example_map"
    fname = "config_test"
    conf = lib.load_config_namespace(fname)

    pre_map = PreMap(conf)
    pre_map.run_conversion()

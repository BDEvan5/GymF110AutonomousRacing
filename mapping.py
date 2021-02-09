from numpy.core.fromnumeric import clip
from LibFunctions import load_config_namespace
import yaml 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image

import LibFunctions as lib
from scipy import ndimage 

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

    def run_conversion(self):
        self.read_yaml_file()
        self.load_map()

        self.dt = ndimage.distance_transform_edt(self.map_img) * self.resolution

        self.find_centerline(False)
        self.find_nvecs()
        self.set_true_widths()

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

        d_search = 1 
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
        c = int(clip((pt_xy[0] - self.origin[0]) / self.resolution, 0, self.width-1))
        r = int(clip((pt_xy[1] - self.origin[1]) / self.resolution, 0, self.height-1))

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
        nvecs = self.nvecs
        tx = self.cline[:, 0]
        ty = self.cline[:, 1]

        stp_sze = 0.1
        sf = 0.5 # safety factor
        nws, pws = [], []
        for i in range(self.N):
            pt = [tx[i], ty[i]]
            nvec = nvecs[i]

            j = stp_sze
            s_pt = s_pt = lib.add_locations(pt, nvec, j)
            while not self.check_scan_location(s_pt):
                j += stp_sze
                s_pt = lib.add_locations(pt, nvec, j)
            pws.append(j*sf)

            j = stp_sze
            s_pt = s_pt = lib.sub_locations(pt, nvec, j)
            while not self.check_scan_location(s_pt):
                j += stp_sze
                s_pt = lib.sub_locations(pt, nvec, j)
            nws.append(j*sf)

        nws, pws = np.array(nws), np.array(pws)

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
        plt.plot(cx, cy, linewidth=2)
        lx, ly = self.convert_positions(l_line)
        plt.plot(lx, ly, linewidth=1)
        rx, ry = self.convert_positions(r_line)
        plt.plot(rx, ry, linewidth=1)

        plt.show()

    def check_scan_location(self, pt):
        c, r = self.xy_to_row_column(pt)
        if self.map_img[c, r]:
            return True
        return False

if __name__ == "__main__":
    fname = "config_example_map"
    fname = "config_test"
    conf = lib.load_config_namespace(fname)

    pre_map = PreMap(conf)
    pre_map.run_conversion()

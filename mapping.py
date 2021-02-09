from LibFunctions import load_config_namespace
import yaml 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as img

import LibFunctions as lib


class PreMap:
    def __init__(self, conf) -> None:
        self.conf = conf 
        self.map_name = conf.map_name
        self.map_ext = conf.map_ext

        self.origin = None

    def run_conversion(self):
        self.load_map()
        
    def read_yaml_file(self):
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)

        self.yaml_file = dict(documents.items())

        self.resolution = self.yaml_file['resolution']
        self.origin = self.yaml_file['origin']

    def load_map(self):
        self.read_yaml_file()

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

        self.obs_map = np.zeros_like(self.scan_map)
        print(f"Map size: {self.width * self.resolution}, {self.height * self.resolution}")

    def read_png(self):
        self.scan_map = img.imread(self.map_name+self.map_ext)
        self.scan_map = np.array(self.scan_map)
        self.height = self.scan_map.shape[1]
        self.width = self.scan_map.shape[0]

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

        self.scan_map = data
    
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
        self.scan_map = np.array(raster)        




if __name__ == "__main__":
    fname = "config_example_map"
    fname = "config_test"
    conf = lib.load_config_namespace(fname)

    pre_map = PreMap(conf)
    pre_map.run_conversion()

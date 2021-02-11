import numpy as np 
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as img 
from scipy.ndimage import distance_transform_edt as edt


def convert_map(name: str):
    im = plt.imread('maps/' + name)
    dt = edt(im)

    starting_pt = np.array([0, 0])
    width = im.shape[0]
    height = im.shape[1]
    for i in range(width):
        for j in range(height):
            if im[i+1, j+1]
            



convert_map('example_map.png')


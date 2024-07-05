import heapq

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator

from scipy import ndimage
from PIL import Image

import os, glob
import math

path = 'C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/'

# define starting/end point of circular route
start = [391,119]
# itterate over routes to find order
# append correct order into list
routes = []
routes_unordered = []
for filename in glob.glob(os.path.join(path, '*.csv')):
    if "_route" in filename:
        stn_start = filename.split('_(')[1].replace(')','')
        stn_end = filename.split('_(')[2].replace(')_route.csv','')
        str = [int(stn_start.split(', ')[0]),int(stn_start.split(', ')[1])]
        end = [int(stn_end.split(', ')[0]),int(stn_end.split(', ')[1])]
        routes_unordered.append([str,end,filename])

for sheet in routes_unordered:
    if sheet[0] == start:
        routes.append(sheet)
        routes_unordered.remove(sheet)

def match(a,b):
    if a[1]==b[0]:
        return 1
    else:
        return 0

while len(routes_unordered) > 0:
    for sheet in routes_unordered:
        if match(routes[-1],sheet) == 1:
            routes.append(sheet)
            routes_unordered.remove(sheet)
        else:
            print('pass')
            pass
# itterate over ordered routes
# merge into single arrays to describe XY coordinates and time for each step
traverse_x = []
traverse_y = []
traverse_t = []
for stage in routes:
    df = pd.read_csv(stage[2])
    X = df['X'].values
    Y = df['Y'].values
    T = df['T'].values
    if len(traverse_x)==0:
        traverse_x = X
        traverse_y = Y
        traverse_t = T
    else:
        traverse_x = np.concatenate([traverse_x ,X])
        traverse_y = np.concatenate([traverse_y ,Y])
        traverse_t = np.concatenate([traverse_t ,T])   

traverse_d = np.full(shape=len(traverse_t),fill_value=5,dtype=int)
_dd = np.cumsum(traverse_d) # cum distance
_dt = np.cumsum(traverse_t) # cum time

route_coordinates = []
for name in routes:
    route_coordinates.append(name[1])
print(route_coordinates)
print('Total time: '+np.array2string(_dt[-1])+'\t'+'Total dist: '+np.array2string(_dd[-1]))




import os, glob
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator

from pathlib import Path
from typing import Union


#import math
#from scipy import ndimage
#from PIL import Image
#import heapq
#import seaborn as sns

# adapted from blog post by Jude Capachietti
# https://judecapachietti.medium.com/dijkstras-algorithm-for-adjacency-matrix-in-python-a0966d7093e8
def dijkstra_traverse(start_relcoord,end_relcoord,location):

    import heapq

    import pandas as pd
    import numpy as np
    import seaborn as sns

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import MaxNLocator

    from scipy import ndimage
    from PIL import Image

    def find_shortest_paths(graph, start_point, slopemap):
        # initialize graphs to track if a point is visited
        # current calculated distance from start to point
        # and previous point taken to get to current point
        visited = [[False for col in row] for row in graph]
        distance = [[float('inf') for col in row] for row in graph]
        distance[start_point[0]][start_point[1]] = 0

        time = [[float('inf') for col in row] for row in graph]
        time[start_point[0]][start_point[1]] = 0

        prev_point = [[None for col in row] for row in graph]
        n, m = len(graph), len(graph[0])
        number_of_points, visited_count = n * m, 0
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        min_heap = []

        # min_heap item format:
        # (pt's dist from start on this path, pt's row, pt's col)
        heapq.heappush(min_heap, (distance[start_point[0]][start_point[1]], start_point[0], start_point[1]))

        while visited_count < number_of_points:
            current_point = heapq.heappop(min_heap)
            distance_from_start, row, col = current_point
            for direction in directions:
                new_row, new_col = row + direction[0], col + direction[1]
                if -1 < new_row < n and -1 < new_col < m and not visited[new_row][new_col]:
                    
                    if graph[row][col] < 1.5:
                        dist_to_new_point = 1000000000000000000
                    else:
                        rel_slope =(graph[row][col]-graph[new_row][new_col])/5
                        abs_slope = slopemap[row][col]
                        #if np.degrees(np.arctan(rel_slope)) > 20:
                        if abs(graph[row][col]-graph[new_row][new_col]) > 1.82: # 20 degrees                           
                            dist_to_new_point = 1000000000000000000
                        else:   
                            #extra_weight = (-4.437*(10**-5)*(abs_slope**3))+(0.004032*(abs_slope**2))-(0.1206*abs_slope)+2.317
                            #extra_weight = 2.317 - extra_weight + 1
                            extra_weight = 1
                            a = 1.58
                            b = -3.04
                            c = 0
                            time_to_new_point = (5/(a*np.exp(b*abs((rel_slope)-c))))/60
                            dist_to_new_point = time_to_new_point * extra_weight
                            dist_to_new_point = distance_from_start + dist_to_new_point

                    if dist_to_new_point < distance[new_row][new_col]:
                        #time[new_row][new_col] = time_to_new_point
                        distance[new_row][new_col] = dist_to_new_point
                        prev_point[new_row][new_col] = (row, col)
                        heapq.heappush(min_heap, (dist_to_new_point, new_row, new_col))
            visited[row][col] = True
            visited_count += 1

        return distance, prev_point

    # import elevation map as tif
    dem_file = '_raster/_expzone_DEM.tif'
    expzone = Image.open(dem_file)
    expzone = np.array(expzone)
    # set all elevations <1.5 as sea
    expzone = np.where(expzone < 1.5, 0, expzone)
    expzone = expzone[::-1,]

    # import slope map as tif
    slope_file = '_raster/_expzone_SLOPE.tif'
    l_slope = Image.open(slope_file)
    l_slope = np.array(l_slope)

    # calculate time map to traverse to raster pixels from starting location
    distance, prev_point = find_shortest_paths(expzone, start_relcoord, l_slope) #Y THEN X
    distance = np.array(distance)
    distance = np.where(distance>240,np.nan,distance)
    # create dataframe from calculated time map
    _x = []
    _y = []
    _t = []
    for iy, ix in np.ndindex(distance.shape):
        _x.append(int(ix))
        _y.append(int(iy))
        _t.append(distance[int(iy)][int(ix)])
    df = pd.DataFrame.from_dict(np.array([_x,_y,_t]).T)
    df.columns = ['X_value','Y_value','Z_value']
    df['X_value'] = pd.to_numeric(df['X_value'])
    df['Y_value'] = pd.to_numeric(df['Y_value'])
    df['Z_value'] = pd.to_numeric(df['Z_value'])
    # reset tif raster coordinates into OS coordinates
    lowerleft = (169455,706855)
    df['X_value'] = ((df['X_value']*5)+lowerleft[0])+2.5
    df['Y_value'] = ((df['Y_value']*5)+lowerleft[1])+2.5

    # export dataframe to csv
    location
    df.to_csv(location+str(start_relcoord)+'.csv', 
                encoding='utf-8', index=False)
    
    # plot time map as a heatmap and export
    pivotted = df.pivot('Y_value','X_value','Z_value')
    fig, ax  = plt.subplots()
    # plot heatmap
    ax = sns.heatmap(pivotted,cmap="viridis",ax=ax)

    # set axes
    ax.invert_yaxis()
    ax.set_aspect('equal')
    # save plot of map
    fig = ax.get_figure()
    fig.set_size_inches(12, 10)
    fig.savefig(location+str(start_relcoord)+'.png',
                    dpi=300,
                    bbox_inches = "tight")

    # plot time map as a heatmap and export
    # show contours on map
    pivotted = df.pivot('Y_value','X_value','Z_value')
    fig, ax  = plt.subplots()
    # plot heatmap
    ax = sns.heatmap(pivotted,cmap="viridis",ax=ax)
    # prepare and plot contours
    levels = np.arange(0, 240, 10)
    smooth_scale = 20
    import skimage.transform as st 
    new_size = tuple([smooth_scale*x for x in (pivotted.to_numpy()).shape])
    z = st.rescale(pivotted.to_numpy(), smooth_scale, mode='constant')
    cntr = ax.contour(np.linspace(0, len(pivotted.columns), len(pivotted.columns)*smooth_scale),
                    np.linspace(0, len(pivotted.index), len(pivotted.index)*smooth_scale),
                    z, levels=levels, 
                    colors='yellow',
                    linewidths=0.5)
    ax.clabel(cntr, inline=True, fontsize=8)

    # set axes
    ax.invert_yaxis()
    ax.set_aspect('equal')
    # save plot of map
    fig = ax.get_figure()
    fig.set_size_inches(12, 10)
    fig.savefig(location+str(start_relcoord)+'contour.png',
                    dpi=300,
                    bbox_inches = "tight")

    def find_shortest_route(prev_point_graph, end_point):
        shortest_path = []
        current_point = end_point
        while current_point is not None:
            shortest_path.append(current_point)
            current_point = prev_point_graph[current_point[0]][current_point[1]]
        shortest_path.reverse()
        return shortest_path

    # find path to a point and export path
    path_x = []
    path_y = []
    path_t = []
    for py, px in find_shortest_route(prev_point, end_relcoord):
        path_t.append(distance[py,px])
        path_x.append(((px*5)+lowerleft[0])+2.5)
        path_y.append(((py*5)+lowerleft[1])+2.5)
    path_t = np.array(path_t) ###
    path_t[1:] -= path_t[:-1].copy() ###
    path_df = pd.DataFrame.from_dict(np.array([path_x,path_y,path_t]).T)
    path_df.columns = ['X','Y','T']
    # export route as .csv
    path_df.to_csv(location+'_'+str(start_relcoord)+'_'+str(end_relcoord)+'_route.csv', 
                    encoding='utf-8', index=False)

    plt.close("all")

    return

# set home holder
home = 'C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output'

def clear_directory_folders(directory_path: Union[str, Path]) -> list:
    """Irreversibly removes all folders (and their content) in the specified
    directory. Doesn't remove files of that specified directory. Returns a
    list with folder paths Python lacks permission to delete."""
    erroneous_paths = []
    for path_location in Path(directory_path).iterdir():
        if path_location.is_dir():
            try:
                shutil.rmtree(path_location)
            except PermissionError:
                erroneous_paths.append(path_location)
    return erroneous_paths
def delete_files_in_directory(directory_path):

    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        print("Error occurred while deleting files.")
        exit()

# create output folder or clear output folder 
puzzle_output = r'C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/traverse_permutations'
if not os.path.exists(puzzle_output):
    os.makedirs(puzzle_output)
else:
   delete_files_in_directory(puzzle_output)
   clear_directory_folders(puzzle_output)
# add full traverse routes folder
traverse_output = r'C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/traverse_permutations/traverses'
os.makedirs(traverse_output)

# define function to itterate over each locality 
def traverse_perm(*args):
    locality_number = len(args)-1
    # define output arrays to capture result of each permutation
    itt_station_indx = []
    itt_station_list = []
    itt_station_time = []
    itt_station_dist = []
    # create list of all possible permutations of localities
    current_itt = 0
    from itertools import permutations 
    perm = permutations(args[1:]) 
    # itterate over each permutation
    for i in list(perm): 
        # define start and end point
        startgrid = [170328,708561]
        current_itt = current_itt+1
        itt_route = np.concatenate([[startgrid],i])
        itt_route = np.concatenate([ itt_route,[startgrid]])

        # create folder for output of permutation
        itt_output = 'C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/traverse_permutations/tp'+str(current_itt)+'/'
        os.makedirs(itt_output)
        # define axes to match topography raster BNG
        lowerleft = (169455,706855)
        # itterate over localities in permutation to generate traverse
        for index,object in enumerate(i):
            if index == 0:
                # start
                ###print('start',object[0],object[1])
                dijkstra_traverse((int((startgrid[1] - 2.5 - lowerleft[1])/5),  int((startgrid[0] - 2.5 - lowerleft[0])/5)),
                             (int((object[1] - 2.5 - lowerleft[1])/5),  int((object[0] - 2.5 - lowerleft[0])/5)),  
                              itt_output)
                _prev = object[0],object[1]
            else:
                # each possible intermediate
                dijkstra_traverse((int((_prev [1] - 2.5 - lowerleft[1])/5),  int((_prev [0] - 2.5 - lowerleft[0])/5)),
                             (int((object[1] - 2.5 - lowerleft[1])/5),  int((object[0] - 2.5 - lowerleft[0])/5)),  
                              itt_output)
                _prev = object[0],object[1]
        # end
        dijkstra_traverse((int((object[1] - 2.5 - lowerleft[1])/5),  int((object[0] - 2.5 - lowerleft[0])/5)),
                     (int((startgrid[1] - 2.5 - lowerleft[1])/5),  int((startgrid[0] - 2.5 - lowerleft[0])/5)),
                      itt_output)

        # consolidate generated paths
        # itterate over routes to find order and append correct order into list
        start_ref = [int((startgrid[1] - 2.5 - lowerleft[1])/5),int((startgrid[0] - 2.5 - lowerleft[0])/5)]

        routes = []
        routes_unordered = []
        # open route files between localities within output folder
        for filename in glob.glob(os.path.join(itt_output, '*.csv')):
            if "_route" in filename:
                ###print(filename)
                stn_start = filename.split('_(')[1].replace(')','')
                stn_end = filename.split('_(')[2].replace(')_route.csv','')
                srt = [int(stn_start.split(', ')[0]),int(stn_start.split(', ')[1])]
                end = [int(stn_end.split(', ')[0]),int(stn_end.split(', ')[1])]
                routes_unordered.append([srt,end,filename])
        # find path from start
        for sheet in routes_unordered:
            if sheet[0] == start_ref:
                routes.append(sheet)
                routes_unordered.remove(sheet)
        def match(a,b):
            if a[1]==b[0]:
                return 1
            else:
                return 0
        # itterate to find order of paths to complete circular traverse
        # create ordered traverse
        while len(routes_unordered) > 0:
            for sheet in routes_unordered:
                if match(routes[-1],sheet) == 1:
                    routes.append(sheet)
                    routes_unordered.remove(sheet)
                else:
                    pass
        # itterate over ordered traverse
        # merge paths into single traverse array to describe XY coordinates for each step
        traverse_x = [] # x coorindate
        traverse_y = [] # y coorindate 
        traverse_t = [] # time for step
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

        # export complete traverse as single spreadsheet
        traverse_output = r'C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/puzzler/traverses'
        traverse = pd.DataFrame({   'X': traverse_x,
                                    'Y': traverse_y,
                                    'T': traverse_t,
                                    'cT': _dt, # cum time
                                    'cD': _dd, # cum distance
                                    
                                })
        # save dataframe
        traverse.to_csv('C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/traverse_permutations/traverses/traverse_tp_'+str(int(current_itt))+'.csv', sep=',')

        # append result of permutation to array
        itt_station_indx.append(int(current_itt))
        itt_station_list.append(np.array_str(itt_route).replace('\n', ' ').replace('\r', ''))
        itt_station_time.append(_dt[-1])
        itt_station_dist.append(_dd[-1])

    # combine output of each permutation into dataframe 
    dataset = pd.DataFrame({    'indx': itt_station_indx,
                                'list': itt_station_list,
                                'time': np.array(itt_station_time),
                                'dist': np.array(itt_station_dist),
                            })
    # save dataframe
    dataset.to_csv('C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/traverse_permutations/output.csv', sep=',')

#start  =     [170054 , 708814]

start  =      [170328,708561] #[170514 , 708549] #beach


locality_A1 =  [170702,707463]#	R2_16a
locality_A2 =  [171118,707404]#	Z6-L1
locality_A3 =  [171199,707626]#	Z7-L3
locality_A4 =  [170790,707257]#	R2_15a
			
locality_B1 =  [170608,708366]#	R2_05
locality_B2 =  [170732,707729]#	Z2-L2
locality_B3 =  [171047,708320]#	Z5-L2
locality_B4 =  [170862,708485]#	Z5-L11
			
locality_C1 =  [170645,708609]#	R3_08
locality_C2 =  [170814,708973]#	R3_07a
locality_C3 =  [170334,709065]#	Z1-L10
locality_C4 =  [170556,709215]#	R3_02
			
#traverse_perm(start,locality_A1,locality_A2,locality_A3,locality_A4)
#traverse_perm(start,locality_B1,locality_B2,locality_B3,locality_B4)
traverse_perm(start,locality_C1,locality_C2,locality_C3,locality_C4)
print('fin')

        
# adapted from blog post by Jude Capachietti
# https://judecapachietti.medium.com/dijkstras-algorithm-for-adjacency-matrix-in-python-a0966d7093e8

import heapq
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator

def find_shortest_paths(graph, start_point):
    # initialize graphs to track if a point is visited
    # current calculated distance from start to point
    # and previous point taken to get to current point
    visited = [[False for col in row] for row in graph]
    distance = [[float('inf') for col in row] for row in graph]
    distance[start_point[0]][start_point[1]] = 0
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
                    dist_to_new_point = 100000000
                else:
                    slope =(graph[row][col]-graph[new_row][new_col])/5
                    if np.degrees(np.arctan(slope)) > 15:
                        dist_to_new_point = 100000000
                    else:
                        dist_to_new_point = (5/((2.2370775*np.exp(-3.5*abs((slope)-0.00923238)))/3.6))/60
                        dist_to_new_point = distance_from_start + dist_to_new_point

                if dist_to_new_point < distance[new_row][new_col]:
                    distance[new_row][new_col] = dist_to_new_point
                    prev_point[new_row][new_col] = (row, col)
                    heapq.heappush(min_heap, (dist_to_new_point, new_row, new_col))
        visited[row][col] = True
        visited_count += 1

    return distance, prev_point

# import elevation map as tif
dem_file = '_raster/_Lunga_DEM.tif'
lunga = Image.open(dem_file)
lunga = np.array(lunga)
# set all elevations <1.5 as sea
lunga = np.where(lunga < 1.5, 0, lunga)
lunga = lunga[::-1,]


start_relcoord =    (397, 118)
end_relcoord   =    (91,370)


# calculate time map to traverse to raster pixels from starting location
distance, prev_point = find_shortest_paths(lunga, start_relcoord) #Y THEN X
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
df.to_csv('_output/da_map_'+str(start_relcoord)+'.csv', 
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
fig.savefig('_output/da_map_'+str(start_relcoord)+'.png',
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
fig.savefig('_output/da_map_'+str(start_relcoord)+'contour.png',
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
for py, px in find_shortest_route(prev_point, end_relcoord):
    path_x.append(((px*5)+lowerleft[0])+2.5)
    path_y.append(((py*5)+lowerleft[1])+2.5)
path_df = pd.DataFrame.from_dict(np.array([path_x,path_y]).T)
path_df.columns = ['X','Y']
# export route as .csv
path_df.to_csv('_output/da_path_'+str(start_relcoord)+'_'+str(end_relcoord)+'_route.csv', 
                encoding='utf-8', index=False)

exit()
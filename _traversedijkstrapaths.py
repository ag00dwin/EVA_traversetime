import heapq

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator

from scipy import ndimage
from PIL import Image

# adapted from blog post by Jude Capachietti
# https://judecapachietti.medium.com/dijkstras-algorithm-for-adjacency-matrix-in-python-a0966d7093e8
def dijkstra_eva(start_coord,end_coord,location):

    # reset tif raster coordinates into OS coordinates
    lowerleft = (169455,706855)
    start_relcoord  = (int((start_coord[1] - 2.5 - lowerleft[1])/5), int((start_coord[0] - 2.5 - lowerleft[0])/5))
    end_relcoord    = (int((end_coord[1]   - 2.5 - lowerleft[1])/5), int((end_coord[0]   - 2.5 - lowerleft[0])/5))

    def find_shortest_paths(graph, start_point):
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

        heapq.heappush(min_heap, (distance[start_point[0]][start_point[1]], start_point[0], start_point[1]))

        while visited_count < number_of_points:
            current_point = heapq.heappop(min_heap)
            distance_from_start, row, col = current_point
            for direction in directions:
                new_row, new_col = row + direction[0], col + direction[1]
                if -1 < new_row < n and -1 < new_col < m and not visited[new_row][new_col]:
                    # distance is a weight of time generated from hiking function
                    if graph[row][col] < 1.5:
                        dist_to_new_point = 1000000000000000000
                    else:
                        rel_slope =(graph[row][col]-graph[new_row][new_col])/5

                        if abs(graph[row][col]-graph[new_row][new_col]) > 1.82: # much be <20 degree slope                            
                            dist_to_new_point = 1000000000000000000
                        else:   
                            # application of hiking function
                            a = 1.58
                            b = -3.04
                            c = 0
                            dist_to_new_point = (5/(a*np.exp(b*abs((rel_slope)-c))))/60
                            dist_to_new_point = distance_from_start + dist_to_new_point

                    if dist_to_new_point < distance[new_row][new_col]:
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

    # calculate time map to traverse to raster pixels from starting location
    distance, prev_point = find_shortest_paths(expzone, start_relcoord)
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
    df['X_value'] = ((df['X_value']*5)+lowerleft[0])+2.5
    df['Y_value'] = ((df['Y_value']*5)+lowerleft[1])+2.5

    # export dataframe to csv
    df.to_csv(location+str(start_coord)+'.csv', 
                encoding='utf-8', index=False)
    
    ### plot time map as a heatmap and export
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
    fig.savefig(location+str(start_coord)+'.png',
                    dpi=300,
                    bbox_inches = "tight")

    ### plot time map as a heatmap and export
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
    fig.savefig(location+str(start_coord)+'_contour.png',
                    dpi=300,
                    bbox_inches = "tight")

    # derive shortest path between coordinates 
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
    path_df.to_csv(location+str(start_coord)+'_'+str(end_coord)+'_route.csv', 
                    encoding='utf-8', index=False)

    return

lowerleft = (169455,706855)
start     = [170054 , 708814] #EASTING, NORTHING
stationA  = [170702 , 707463] #EASTING, NORTHING

dijkstra_eva((170054,708814),(170702,707463),'_output/_anisotropic mapping/')

exit()


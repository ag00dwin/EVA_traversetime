import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pandas as pd
import os
import glob
from PIL import Image

import geopy.distance

# import elevation map
dem_file = '_raster/_Lunga_DEM.csv'
dem = pd.read_csv(dem_file,sep=',')
# define array variables from DEM
dem_z = dem['VALUE'].values
dem_x = dem['X'].values
dem_y = dem['Y'].values
# create list of coorinate paris
amap_coordinates = np.array(list(zip(dem_x,dem_y)))
# create function to match closest coorindates
def c_index(X,Y):
    distances = np.linalg.norm(amap_coordinates-np.array((X,Y)), axis=1)
    min_index = np.argmin(distances)
    # return [0] index and [1] distance error from it
    return min_index, distances[min_index]

stop_time_total = []

# loop over each EVA traverse
def stop_finder():
    # loop over the list of csv files
    csv_files = glob.glob(os.path.join('_data', "*.csv"))
    for f in csv_files:

        _df  = pd.read_csv(f,sep=',',)
        
        print(f)
        _df = pd.read_csv(f,sep=',', parse_dates=['time'])
        _df = _df.resample('1Min',on='time').mean()
        _df = _df.reset_index()
        _df = _df.interpolate(method='pad') #########

        _df['time'] = (_df['time'].astype('int64')/(10**9))/60

        mapgrid_x = []
        mapgrid_y = []
        mapgrid_z = []
        mapgrid_error = []
        for index, row in _df.iterrows():
            mapgrid_x.append(dem_x[c_index(_df['X'][index],_df['Y'][index])[0]])
            mapgrid_y.append(dem_y[c_index(_df['X'][index],_df['Y'][index])[0]])
            mapgrid_z.append(dem_z[c_index(_df['X'][index],_df['Y'][index])[0]]) 
            mapgrid_error.append(c_index(_df['X'][index],_df['Y'][index])[1])
            
        _df['mapgrid_x'] = mapgrid_x
        _df['mapgrid_y'] = mapgrid_y
        _df['mapgrid_z'] = mapgrid_z

        dx_ = [] # delta distance
        dz_ = [] # delta elevation

        # itterate over every step in the traverse
        for index, row in _df.iterrows():
            if index == 0:
                dx_.append(0)
                dz_.append(0)
            else:
                elev_1 = (_df['mapgrid_z'][index  ])
                elev_2 = (_df['mapgrid_z'][index-1])
                dz = elev_1-elev_2
    
                dz_.append(dz)

                # linear distance
                coords_1 = (_df['latitude'][index  ], _df['longitude'][index  ])
                coords_2 = (_df['latitude'][index-1], _df['longitude'][index-1])
                dx_l = geopy.distance.geodesic(coords_1, coords_2).m
  
                dx_.append(dx_l) 

        # update dataframe with new variables
        _df['dx'] = dx_                 # delta distance
        _df['dx_cum'] = np.cumsum(dx_)  # cumulative distance
        _df['dz'] = dz_                 # delta elevation

        _df = _df[_df['dx']<10]

        xyz = list(zip(_df['mapgrid_x'],_df['mapgrid_y'],_df['time']))
        
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=10, min_samples=3).fit(xyz)
        
        # update dataframe with new variables
        _df['groupings'] = db.labels_
        _df = _df[_df['groupings'] > -1]

        # export dataframe as .csv
        _df.to_csv('_output/_routegroupings'+str(f.split('\\')[1]),
                        encoding='utf-8', index=False)
        
        plt.scatter(_df['X'],_df['Y'],c=_df['groupings'])
        plt.colorbar()
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    
        # itterate over every stop
        working_df = _df
        max_stop_index = np.max(working_df['groupings'])
        for stop in range(0,max_stop_index+1,1):
            #print(stop)
            grouparray = working_df.index[working_df['groupings'] == stop].tolist()
            grouparray = np.array(grouparray)
            stop_time = len(grouparray)
            #if stop_time < 5:
            #    pass
            #else:
            stop_time_total.append(int(stop_time))

        '''
        working_df = _df
        max_stop_index = np.max(working_df['groupings'])
        print(max_stop_index)
        for stop in range(0,max_stop_index+1,1):
            print(stop)
            pre_stop_index = []
            for index, row in working_df.iterrows():
                if working_df['groupings'][index] == stop:
                    break
                else:
                    pre_stop_index.append(float(index))

            pre_stop_index = np.array(pre_stop_index)
            print(pre_stop_index)
            grouparray = working_df.index[working_df['groupings'] == stop].tolist()
            grouparray = np.array(grouparray)
            maxgrouparray = np.max(grouparray)
            print(grouparray)

            working_df = _df[maxgrouparray:]
        '''
        
stop_finder()


stop_values = []
above_max = 0
stop_time_total = np.array(stop_time_total)
for s in stop_time_total:
    if s > 60:
        above_max += 1
    else:
       stop_values.append(s)
stop_values = np.array(stop_values) 
print(above_max,' above 60 mins')

binwidth=3

# https://stackoverflow.com/questions/6352740/matplotlib-label-each-bin
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(stop_values, bins=range(0, 60 + binwidth, binwidth))

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)
# Set the xaxis's tick labels to be formatted with 1 decimal place.
from matplotlib.ticker import FormatStrFormatter
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

# Label the raw counts and the percentages below the x-axis
bin_centers = 0.5 * np.diff(bins) + bins[:-1]
for count, x in zip(counts, bin_centers):
    # Label the raw counts
    ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    # Label the percentages
    percent = '%0.0f%%' % (100 * float(count) / counts.sum())
    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -32), textcoords='offset points', va='top', ha='center')


# Give ourselves some more room at the bottom of the plot
plt.subplots_adjust(bottom=0.15)

#add axis labels
ax.set_ylabel('No. of activity stops')
ax.set_xlabel('time (mins)')

#adjust position of both axis labels
ax.xaxis.set_label_coords(.9, 0)

plt.show() 

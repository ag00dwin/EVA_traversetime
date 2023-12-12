import os
import glob
import geopy.distance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

# import elevation map
dem_file = '_raster/_expzone_DEM.csv'
dem = pd.read_csv(dem_file,sep=',')
# define array variables from DEM
dem_z = dem['VALUE'].values
dem_x = dem['X'].values
dem_y = dem['Y'].values
# create list of coorinate pairs
amap_coordinates = np.array(list(zip(dem_x,dem_y)))
# create function to match to closest coorindates on the elevation map
def c_index(X,Y):
    distances = np.linalg.norm(amap_coordinates-np.array((X,Y)), axis=1)
    min_index = np.argmin(distances)
    # return [0] index and [1] distance error from raster pixel
    return min_index, distances[min_index]


'''---stop_finder---'''

stop_time_total = []
# loop over each EVA traverse
def stop_finder():
    # loop over the list of csv files
    csv_files = glob.glob(os.path.join('_data', "*.csv"))
    for f in csv_files:
        print(f)
        # import traverse file into a dataframe and resample at 1 min intervals        
        _df = pd.read_csv(f,sep=',', parse_dates=['time'])
        _df = _df.resample('1Min',on='time').mean()
        _df = _df.reset_index()
        _df = _df.interpolate(method='pad')
        _df['time'] = (_df['time'].astype('int64')/(10**9))/60
        # find xyz position on elevation raster grid
        # itterate over whole traverse
        mapgrid_x = []
        mapgrid_y = []
        mapgrid_z = []
        mapgrid_error = []        
        for index, row in _df.iterrows():
            mapgrid_x.append(dem_x[c_index(_df['X'][index],_df['Y'][index])[0]])
            mapgrid_y.append(dem_y[c_index(_df['X'][index],_df['Y'][index])[0]])
            mapgrid_z.append(dem_z[c_index(_df['X'][index],_df['Y'][index])[0]]) 
            mapgrid_error.append(c_index(_df['X'][index],_df['Y'][index])[1])
        # append values to dataframe 
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
                # elevation
                elev_1 = (_df['mapgrid_z'][index  ])
                elev_2 = (_df['mapgrid_z'][index-1])
                dz = elev_1-elev_2
                dz_.append(dz)

                # linear distance
                # does not consider elevation change 
                coords_1 = (_df['latitude'][index  ], _df['longitude'][index  ])
                coords_2 = (_df['latitude'][index-1], _df['longitude'][index-1])
                dx_l = geopy.distance.geodesic(coords_1, coords_2).m
                dx_.append(dx_l) 

        # update dataframe with new variables
        _df['dx'] = dx_                 # delta distance
        _df['dx_cum'] = np.cumsum(dx_)  # cumulative distance
        _df['dz'] = dz_                 # delta elevation
        
        # threshold
        # remove pixels where distance travelled >= 10 m 
        _df = _df[_df['dx']<10]
        # zip coordinates and group by cluster
        xyz = list(zip(_df['mapgrid_x'],_df['mapgrid_y'],_df['time']))
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=10, min_samples=3).fit(xyz)
        # update dataframe with group labels
        _df['groupings'] = db.labels_
        # remove all values that were not clustered into a group
        _df = _df[_df['groupings'] > -1]

        # export group dataframe as .csv
        # plot groups 
        _df.to_csv('_output/_routegroupings'+str(f.split('\\')[1]),
                        encoding='utf-8', index=False)
        
        plt.scatter(_df['X'],_df['Y'],c=_df['groupings'])
        plt.colorbar()
        #plt.show()
    
        # itterate over every group
        # find stop time for every group
        working_df = _df
        max_stop_index = np.max(working_df['groupings'])
        for stop in range(0,max_stop_index+1,1):
            grouparray = working_df.index[working_df['groupings'] == stop].tolist()
            grouparray = np.array(grouparray)
            stop_time = len(grouparray)
            stop_time_total.append(int(stop_time))
        
stop_finder()

# find number of stops over 60 mins out of scale for the histogram
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

# plot histogram of stop times
# formating from Joe Kington: https://stackoverflow.com/questions/6352740/matplotlib-label-each-bin
binwidth=3
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(stop_values, bins=range(0, 60 + binwidth, binwidth))
# Set the ticks to be at the edges of the bins and formatted to integers 
ax.set_xticks(bins)
from matplotlib.ticker import FormatStrFormatter
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# Label the counts and the percentages below the x-axis
bin_centers = 0.5 * np.diff(bins) + bins[:-1]
for count, x in zip(counts, bin_centers):
    # Label the counts
    ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -18), textcoords='offset points', va='top', ha='center')
    # Label the percentages
    percent = '%0.0f%%' % (100 * float(count) / counts.sum())
    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -32), textcoords='offset points', va='top', ha='center')
#add axis labels
ax.set_ylabel('No. of activity stops')
ax.set_xlabel('time (mins)')
#adjust position of both axis labels
ax.xaxis.set_label_coords(0.5, -0.15)
# plot histogram
plt.subplots_adjust(bottom=0.15)
plt.show() 

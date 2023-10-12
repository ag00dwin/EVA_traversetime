import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pandas as pd
import os
import glob

# import elevation map as csv
dem_file = '_raster/_lunga_DEM.csv'
dem = pd.read_csv(dem_file,sep=',')
# define array variables from DEM
dem_z = dem['VALUE'].values
dem_x = dem['X'].values
dem_y = dem['Y'].values

# import slope map as csv 
slope_file = '_raster/_lunga_SLOPE.csv'
slope = pd.read_csv(slope_file,sep=',')
# define array variables from SLOPE
slope_z = slope['VALUE'].values
slope_x = slope['X'].values
slope_y = slope['Y'].values

# create list of coorinate pairs from dem
amap_coordinates = np.array(list(zip(dem_x,dem_y)))
# create function to match coorindates to closest coorindates in slope/dem
def c_index(X,Y):
    distances = np.linalg.norm(amap_coordinates-np.array((X,Y)), axis=1)
    min_index = np.argmin(distances)
    # return [0] index and [1] distance difference closest point
    return min_index, distances[min_index]

# loop over the list of csv files in accessible folder
csv_files = glob.glob(os.path.join('C:/Users/Arthu/Desktop/_analogue application/_post mission/EVA data', "*.csv"))
for f in csv_files:
    print(f)
    _df = pd.read_csv(f,sep=',', parse_dates=['time'])
    _df = _df.resample('1Min',on='time').mean()
    _df = _df.reset_index()
    _df.interpolate()
    
    # for each traverse iterval, match with elevation and slope   
    # copy values and append to dataframe    
    mapgrid_x = []
    mapgrid_y = []
    mapgrid_z = []
    mapgrid_error = []
    mapgrid_slope = []
    for index, row in _df.iterrows():
        mapgrid_x.append(dem_x[c_index(_df['X'][index],_df['Y'][index])[0]])
        mapgrid_y.append(dem_y[c_index(_df['X'][index],_df['Y'][index])[0]])
        mapgrid_z.append(dem_z[c_index(_df['X'][index],_df['Y'][index])[0]]) 
        mapgrid_error.append(c_index(_df['X'][index],_df['Y'][index])[1])
        mapgrid_slope.append(slope_z[c_index(t_df['X'][index],_df['Y'][index])[0]]) 
    _df['mapgrid_x'] = mapgrid_x
    _df['mapgrid_y'] = mapgrid_y
    _df['mapgrid_z'] = mapgrid_z
    _df['mapgrid_error'] = mapgrid_error
    _df['mapgrid_slope'] = mapgrid_slope

    # itterate over every interval in the traverse for dx
    _dx = []
    import geopy.distance
    for index, row in t_df.iterrows():
        if index == 0:
            # starting point
            _dx.append(0)
        else:
            try:
                coords_1 = (t_df['latitude'][index  ], t_df['longitude'][index  ])
                coords_2 = (t_df['latitude'][index-1], t_df['longitude'][index-1])
                dx = geopy.distance.geodesic(coords_1, coords_2).m
            except:
                dx = 0
            _dx.append(dx)

    # add dx to dataframe
    t_df['dx'] = _dx 
    # threshold by minimum distance
    t_df = t_df[t_df['dx']>=5]
    t_df = t_df.reset_index()

    # itterate over every interval in the traverse
    # calculate recorded time for interval and compare to model
    # model a function of average slope for defined dx distance
    true_time       = []
    relslope_time   = []
    absslope_time   = []

    # itterate over every interval in the traverse
    from datetime import datetime
    for index, row in t_df.iterrows():   
        if index == 0:
            # starting point
            true_time.append(0)
            relslope_time.append(0)
            absslope_time.append(0)
        else:
            elev_1 = (t_df['mapgrid_z'][index  ])
            elev_2 = (t_df['mapgrid_z'][index-1])
            dz = elev_2-elev_1

            t1 = t_df['time'][index  ]
            t2 = t_df['time'][index-1]
            delta_t = t1 - t2
            true_time.append(delta_t.total_seconds())
            
            # expoential model for relative slope
            relslope_time.append(t_df['dx'][index]/(((1.90178465*np.exp(-abs(2.4607634)*abs((dz/t_df['dx'][index])))))/3.6)) 
            # polynomial model for absolute slope
            absslope = np.degrees(np.arctan(dz/t_df['dx'][index]))
            absslope_time.append(t_df['dx'][index]/((   (-4.437*(10**-5)*(absslope**3))+
                                                        ( 0.004032*(absslope**2))+
                                                        (-0.1206*(absslope))+
                                                        2.317)/3.6)) 

    # update dataframe with time estimates for each interval of traverse
    t_df['truetime_inva'] = true_time       
    t_df['relstime_inva'] = relslope_time 
    t_df['absstime_inva'] = absslope_time  
    
    ################

    # remove time intervals greater than 1 min 
    t_df = t_df[t_df['truetime_inva']<=60]

    t_df['truetime'] = np.cumsum(t_df['truetime_inva'].values,axis=0) 
    t_df['relstime'] = np.cumsum(t_df['relstime_inva'].values,axis=0) 
    t_df['absstime'] = np.cumsum(t_df['absstime_inva'].values,axis=0) 

    t_df['distance'] = np.cumsum(t_df['dx'].values,axis=0)  

    fig, ax = plt.subplots(figsize=(12.5, 6))
    ax.plot(t_df['distance'].values,t_df['truetime'].values,label='actual time')
    ax.plot(t_df['distance'].values,t_df['relstime'].values,label='rel slope time')
    ax.plot(t_df['distance'].values,t_df['absstime'].values,label='abs slope time')
    plt.legend()
    plt.show()

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import elevation map as csv
dem_file = '_raster/_expzone_DEM.csv'
dem = pd.read_csv(dem_file,sep=',')
# define array variables from DEM
dem_z = dem['VALUE'].values
dem_x = dem['X'].values
dem_y = dem['Y'].values

# import slope map as csv 
slope_file = '_raster/_expzone_SLOPE.csv'
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

# define walk speed via toblers hiking function as a vector
# https://en.wikipedia.org/wiki/Tobler%27s_hiking_function
def walkspeed(dz,dx):
    # dz = elevation difference
    # dx = distance
    W = (6*np.exp(-3.5*abs((dz/dx)+0.05)))*(3/5)
    S = np.degrees(np.arctan(dz/dx))
    return W, S

# run model to: 
# (1) fit exponential curve to relative slope
# (2) fit polynomial to absolute slope
def walk_slope_model():

    # define cumulative sum variables
    all_speed    = []
    all_slope    = []
    all_absslope = []

    # loop over the list of csv files in accessible folder
    csv_files = glob.glob(os.path.join('_data', "*.csv"))

    for f in csv_files:
        # resample data to one minute intervals
        _df = pd.read_csv(f,sep=',', parse_dates=['time'])

        _df = _df.resample('1Min',on='time').mean()
        _df = _df.reset_index()
        # interpolate (default: linear)
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
            mapgrid_slope.append(slope_z[c_index(_df['X'][index],_df['Y'][index])[0]]) 
        _df['mapgrid_x'] = mapgrid_x
        _df['mapgrid_y'] = mapgrid_y
        _df['mapgrid_z'] = mapgrid_z
        _df['mapgrid_error'] = mapgrid_error
        _df['mapgrid_slope'] = mapgrid_slope

        # for each interval in the traverse calaculate properties
        import geopy.distance
        dx_ = [] # delta distance
        dxa_= [] # detla distance (actual with slope)
        dz_ = [] # detla elevation
        dt_ = [] # slope angle
        dat_= [] # absolute slope angle
        ds_ = [] # speed
        
        # itterate over every interval in the traverse
        for index, row in _df.iterrows():
            if index == 0:
                # starting point
                dx_.append(0)
                dxa_.append(0)
                dz_.append(0)
                dt_.append(0)
                dat_.append(0)
                ds_.append(0)
            else:
                try:
                    coords_1 = (_df['latitude'][index  ], _df['longitude'][index  ])
                    coords_2 = (_df['latitude'][index-1], _df['longitude'][index-1])
                    dx = geopy.distance.geodesic(coords_1, coords_2).m
                except:
                    dx = 0

                elev_1 = (_df['mapgrid_z'][index  ])
                elev_2 = (_df['mapgrid_z'][index-1])
                
                dz = elev_2-elev_1
                dz_.append(dz)
                dt_.append(np.degrees(np.arctan(dz/dx)))
                dat_.append(_df['mapgrid_slope'][index])

                # actual distance
                if dz>0:
                    theta = abs(np.arctan(dz/dx))
                    dxa = dx/(np.cos(theta)) 
                else:
                    dxa = dx

                dx_.append(dx) 
                dxa_.append(dxa)
                ds_.append(dxa/16.667)

        # update dataframe with new properties for each interval of traverse
        _df['dx']     = dx_           # delta distance
        _df['dxa']    = dxa_          # delta distance (actual with slope)
        _df['dx_cum'] = np.cumsum(dxa_) # cumulative distance (actual with slope)
        _df['dz']     = dz_           # delta elevation
        _df['dt']     = dt_           # relative slope 
        _df['dat']    = dat_          # absolute slope angle
        _df['ds']     = ds_           # speed (with actual distance)

        # stats for traverse 
        #print(f+' | traverse length = '+str(np.max(_df['dx_cum'])))
        # threshold by minimum distance
        # print stats of threshold
        _df_movement    = _df[_df['dxa']>=5]
        _df_stationary  = _df[_df['dxa']<5]
        #print(f+' | points | time of traverse: '+str(len(_df[_df['dxa']>=0])))
        #print(f+' | points | stationary: '+str(len(_df_stationary))+'| movement: '+str(len(_df_movement)))
        percentage_movement = round((len(_df_movement)/len(_df[_df['dxa']>=0]))*100,3)
        percentage_stationary = round((len(_df_stationary)/len(_df[_df['dxa']>=0]))*100,3)
        #print(f+' | movement% = '+str(round(percentage_movement,2))+'| stationary% = '+str(round(percentage_stationary,2)))
        print('GPSday','length(m)','time(mins)','time_still(mins)','time_still(%)','time_move(mins)','time_move(%)')
        print(f,str(round(np.max(_df['dx_cum']),2)),str(len(_df[_df['dxa']>=0])),str(len(_df_stationary)),percentage_stationary,str(len(_df_movement)),percentage_movement)
        
        # define minimum distance as cellsize in dem/slope raster
        _df = _df[_df['dxa']>=5]
        _df.dropna()

        # append traverse properties to master lists
        for i_1 in _df['ds'].values:    # speed
            all_speed.append(i_1)   
        for i_2 in _df['dt'].values:    # relative slope 
            all_slope.append(i_2)   
        for i_3 in _df['dat'].values:   # absolute slope angle
            all_absslope.append(i_3)   

    # turn master list values into numpy array
    all_speed       = np.array(all_speed)
    all_slope       = np.array(all_slope)
    all_absslope    = np.array(all_absslope)
    all_speed       = np.nan_to_num(all_speed)
    all_slope       = np.nan_to_num(all_slope)
    all_absslope    = np.nan_to_num(all_absslope)

    # define plot space
    fig = plt.figure(figsize=(12.5, 6))
    grid = plt.GridSpec(1, 6, hspace=0.4, wspace=1.5)
    axs0 = fig.add_subplot(grid[0:1, 0:3])
    axs1 = fig.add_subplot(grid[0:1, 3:6])

    # plot LEFT
    axs0.scatter(all_absslope,all_speed)
    # turn all values into histogram
    speed_per_absdegree        = []
    speed_per_absdegree_std    = []
    degree_absnum = []
    for asl in np.arange(0,60,2):
        # create bounds for each histogram bar
        sl_lower = asl
        sl_upper = asl+2
        # itterate over master values and assign to historgram bar
        degree_absvalues = []
        _speed = all_speed
        for index, value in enumerate(all_absslope):
            if sl_lower <= value < sl_upper:
                degree_absvalues.append    (_speed[index])
        if not degree_absvalues:
            degree_absnum.append           (0)
            speed_per_absdegree.append     (0)
            speed_per_absdegree_std.append (0)
        else:
            degree_absnum.append           (len(degree_absvalues))
            speed_per_absdegree.append     (np.mean(degree_absvalues))
            speed_per_absdegree_std.append (2*np.std(degree_absvalues))  
    # plot histogram of values as mean +/- std
    axs0.errorbar(  x=np.arange(0,60,2),
                    y=speed_per_absdegree,
                    yerr=speed_per_absdegree_std,

                    color='black',
                    ecolor='black',

                    capsize=2, 
                    capthick=1,
                    elinewidth=1,
                    markeredgewidth=1,

                    fmt='x',
                    label='Mean Speed [2° int] ±2s'
                    )
    # format axis
    axs0.set_xlabel('slope (degrees)')
    axs0.set_ylabel('speed (km/hr)')
    axs0.set_xlim([0,60])
    axs0.set_ylim([-0.25,5])
    # plot best polynomial best fit
    # print equation
    from sympy import Symbol,expand
    model = np.poly1d(np.polyfit(all_absslope, all_speed, 3))
    print(model)
    axs0.plot(np.arange(0,90,2), model(np.arange(0,90,2)), 'Orange',
                    label = 'best fit: a=%5.6f, b=%5.6f, c=%5.6f, d=%5.6f' % tuple(model))

    # plot RIGHT
    axs1.scatter(all_slope,all_speed)
    # turn all values into histogram
    speed_per_degree        = []
    speed_per_degree_std    = []
    degree_num = []
    for sl in np.arange(-20.5,20.5,2):
        # create bounds
        sl_lower = sl
        sl_upper = sl+2
        # itterate over master values and assign to historgram bar
        degree_values = []
        _speed = all_speed
        for index, value in enumerate(all_slope):
            if sl_lower <= value < sl_upper:
                degree_values.append    (_speed[index])
        if not degree_values:
            degree_num.append           (0)
            speed_per_degree.append     (0)
            speed_per_degree_std.append (0)
        else:
            degree_num.append           (len(degree_values))
            speed_per_degree.append     (np.mean(degree_values))
            speed_per_degree_std.append (2*np.std(degree_values))  
    # plot histogram of values as mean +/- std
    axs1.errorbar(  x=np.arange(-20.5,20.5,2),
                    y=speed_per_degree,
                    yerr=speed_per_degree_std,

                    color='black',
                    ecolor='black',

                    capsize=2, 
                    capthick=1,
                    elinewidth=1,
                    markeredgewidth=1,

                    fmt='x',
                    label='Mean Speed [2° int] ±2s'
                    )
    # plot shape of walkspeed 
    # using toblers hiking function model
    speed = []
    slope = []
    for h in np.arange(-10,10,0.05):
        s1,s2 = walkspeed(h,5)
        speed.append(s1)
        slope.append(s2)
    axs1.plot(slope,speed,c='red',label="Tobler's hiking function")
    # format axis
    axs1.set_xlabel('slope vector (degrees)')
    axs1.set_ylabel('speed (km/hr)')
    axs1.set_xlim([-65,65])
    axs1.set_ylim([-0.25,5])

    # run expoentuial fitting of function to data
    # define function based on toblers hiking function
    def walkspeed_fit(slope,a,b):
        # dz = elevation difference
        # dx = distance
        dzdx = np.tan(np.radians(slope))
        W = (a*np.exp(-abs(b)*abs((dzdx))))
        return W
    # fit function to data
    # print equation
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(walkspeed_fit, all_slope, all_speed)
    print(popt)
    # plot best fit
    axs1.plot(np.arange(-50,50,0.05), walkspeed_fit(np.arange(-50,50,0.05), *popt), 'Orange',
         label='best fit: a=%5.3f, b=%5.3f' % tuple(popt))

    # set legends for LEFT and RIGHT
    axs0.legend(fontsize=8,loc='upper right',frameon=False)
    axs1.legend(fontsize=8,loc='upper right',frameon=False)

    axs0.annotate('No. points= '+str(len(_df)),xy=(40,4))
    axs1.annotate('No. points= '+str(len(_df)),xy=(20,4))

    fig.savefig('_output/slope_modelling.png',
                dpi=300,
                bbox_inches = "tight")
    
    plt.show()

walk_slope_model()
exit()

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

# import slope (gradient) map as csv 
slope_file = '_raster/_expzone_SLOPE.csv'
slope = pd.read_csv(slope_file,sep=',')
# define array variables from SLOPE (gradient)
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
# (1) fit general hiking function to slope
# (2) fit polynomial to gradient
# works for a specific interval which data is resampled to
def walk_slope_model(ouput_folder,time_interval):

    # define cumulative sum variables
    all_speed    = []
    all_slope    = []
    all_gradient = []

    # loop over the list of csv files in accessible folder
    csv_files = glob.glob(os.path.join('_data', "*.csv"))

    for f in csv_files:
        # resample data to defined interval
        _df = pd.read_csv(f,sep=',', parse_dates=['time'])
        _df = _df.resample(str(time_interval)+'s',on='time').mean()
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
        dz_ = [] # detla elevation
        dt_ = [] # slope angle
        dat_= [] # absolute slope angle
        ds_ = [] # speed
        
        # itterate over every interval in the traverse
        for index, row in _df.iterrows():
            if index == 0:
                # starting point
                dx_.append(0)
                dz_.append(0)
                dt_.append(0)
                dat_.append(0)
                ds_.append(0)
            else:
                # convert lat/long into BNG grid coordinates
                import math
                coords_1 = _df['X'][index  ], _df['Y'][index  ]
                coords_2 = _df['X'][index-1], _df['Y'][index-1]
                dx = math.dist(coords_1,coords_2)
                # calculate difference in elevtion
                elev_1 = (_df['mapgrid_z'][index  ])
                elev_2 = (_df['mapgrid_z'][index-1])
                dz = elev_2-elev_1

                dz_.append(dz)
                dt_.append(np.degrees(np.arctan(dz/dx)))
                dat_.append(_df['mapgrid_slope'][index])
                dx_.append(dx) 
                ds_.append((dx/time_interval)*3.6)

        # update dataframe with new properties for each interval of traverse
        _df['dx']     = dx_           # delta distance
        _df['dz']     = dz_           # delta altitude
        _df['dt']     = dt_           # slope 
        _df['dat']    = dat_          # gradient 
        _df['ds']     = ds_           # speed 

        # define minimum distance as cellsize in dem/slope raster
        _df = _df[_df['dx']>=5]
        _df.dropna()

        # append traverse properties to master lists
        for i_1 in _df['ds'].values:    # speed
            all_speed.append(i_1)   
        for i_2 in _df['dt'].values:    # slope 
            all_slope.append(i_2)   
        for i_3 in _df['dat'].values:   # gradient
            all_gradient.append(i_3)   

    # turn master list values into numpy array
    all_speed       = np.array(all_speed)
    all_slope       = np.array(all_slope)
    all_gradient    = np.array(all_gradient)
    all_speed       = np.nan_to_num(all_speed)
    all_slope       = np.nan_to_num(all_slope)
    all_gradient    = np.nan_to_num(all_gradient)

    # define plot space
    fig = plt.figure(figsize=(12.5, 6))
    grid = plt.GridSpec(1, 6, hspace=0.4, wspace=1.5)
    axs0 = fig.add_subplot(grid[0:1, 0:3])
    axs1 = fig.add_subplot(grid[0:1, 3:6])

    # plot LEFT
    axs0.scatter(all_gradient,all_speed)
    # turn all values into histogram
    speed_per_absdegree        = []
    speed_per_absdegree_std    = []
    degree_absnum = []
    for asl in np.arange(0,30,1):
        # create bounds for each histogram bar
        sl_lower = asl
        sl_upper = asl+1
        # itterate over master values and assign to historgram bar
        degree_absvalues = []
        _speed = all_speed
        for index, value in enumerate(all_gradient):
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
    axs0.errorbar(  x=np.arange(0,30,1),
                    y=speed_per_absdegree,
                    yerr=speed_per_absdegree_std,

                    color='black',
                    ecolor='black',

                    capsize=2, 
                    capthick=1,
                    elinewidth=1,
                    markeredgewidth=1,

                    fmt='x',
                    label='Mean Speed [1° int] ±2s'
                    )
    # format axis
    axs0.set_xlabel('Gradient (degrees)')
    axs0.set_ylabel('velcoity (km/hr)')
    axs0.set_xlim([-2,40])
    axs0.set_ylim([-0.25,5])
    # plot best polynomial best fit
    # print equation
    from sympy import Symbol,expand
    model = np.poly1d(np.polyfit(all_gradient, all_speed, 3))
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
    
    # export histogram 
    hist_df = pd.DataFrame.from_dict(np.array([np.arange(-20.5,20.5,2),speed_per_degree,speed_per_degree_std]).T)
    hist_df.columns = ['slope','velocity','2std']
    # export route as .csv
    hist_df.to_csv(ouput_folder+str(time_interval)+'.csv', 
                    encoding='utf-8', index=False)
    
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
    axs1.plot(slope,speed,c='red',label="THF",linestyle='--')
    # format axis
    axs1.set_xlabel('Slope (degrees)')
    axs1.set_ylabel('velocity (km/hr)')
    axs1.set_xlim([-65,65])
    axs1.set_ylim([-0.25,5])

    # run expoentuial fitting of function to data
    # define function based on toblers hiking function
    def walkspeed_fit(slope,a,b,c):
        # dz = elevation difference
        # dx = distance
        dzdx = np.tan(np.radians(slope))
        W = (a*np.exp(-abs(b)*abs((dzdx)+c)))
        return W
    # fit function to data
    # print equation
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(walkspeed_fit, all_slope, all_speed,
                            p0=[1.34,-3.21,-0.00055], 
                            maxfev=5000)
    # plot best fit
    axs1.plot(np.arange(-50,50,0.05), walkspeed_fit(np.arange(-50,50,0.05), *popt), 'Orange',
         label='G-THF Best fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

    # coefficient_of_dermination
    from sklearn.metrics import r2_score
    coefficient_of_dermination = r2_score(all_speed, walkspeed_fit(all_slope, *popt))

    print(str(len(all_speed)), time_interval, round(coefficient_of_dermination,3), popt)

    # set legends for LEFT and RIGHT
    axs0.legend(fontsize=8,loc='upper right',frameon=False)
    axs1.legend(fontsize=8,loc='upper right',frameon=False)
    # annotate plots
    axs0.annotate('No. points= '+str(len(all_speed)),xy=(40,4))
    axs1.annotate('No. points= '+str(len(all_speed)),xy=(20,4))
    axs1.annotate('R2= '+str(round(coefficient_of_dermination,3)),xy=(20,3.75))
    # save plots
    plt.savefig(ouput_folder+str(time_interval)+'.png')
    plt.savefig(ouput_folder+str(time_interval)+'.pdf')
    plt.close("all")

# run for each sampling interval
# from 10 second to 300 seconds at 10 second gaps
for secs in reversed(range(10,305,10)):
    walk_slope_model(
        '_output/_traverse walkspeed/',
        secs,
                    )

exit()

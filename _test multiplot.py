import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''' find upvert downvert pace'''
'''
csv_file = pd.read_csv('C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/speed_second_windows.csv'
                       ,sep=',')
                       



for index,value in enumerate(csv_file['a']):
    a = csv_file['a'][index]
    b = csv_file['b'][index]
    c = csv_file['c'][index]

    def vertpace(S):
        dzdx = np.tan(np.deg2rad(S))
        P = a*np.exp((-1*b)*(abs(dzdx)+c))
        P = abs(P*(1/dzdx))
        
        return P
                    
    # find local minimal
    # downslope
    x = []
    slope = []
    for _s in np.arange(-30,-2,0.05):
        x.append(_s)
        slope.append(vertpace(_s))
    plt.plot(x,slope,c='red')
    
    x = []
    slope = []
    for _s in np.arange(2,30,0.05):
        x.append(_s)
        slope.append(vertpace(_s)) 
    plt.plot(x,slope,c='blue')  
    
plt.show()
exit()
'''

###
###
###
# average error in fit to map
###
###
###




###
###
###
# average time recording
###
###
###

from datetime import datetime

# loop over the list of csv files in accessible folder
csv_files = glob.glob(os.path.join('C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traversetracks/_complete tracks', "*.csv"))


count = 0 
for f in csv_files: 
    _df = pd.read_csv(f,sep=',', parse_dates=['time']) 
    _time = _df['time'].values

    def f(x):
        return int(x)/1000000000
    def array_for(x):
        return np.array([f(xi) for xi in x])

    diff = np.diff(array_for(_time ))

    if count == 0:
        time_intervals = diff
        count +=1
    else:
        time_intervals = np.concatenate((time_intervals,diff))
        count +=1

print(np.mean(time_intervals))
print(np.median(time_intervals))

plt.hist(time_intervals,bins=range(0,100,1))
plt.show()





###
###
###
# location accuracy 
###
###
###

# loop over the list of csv files in accessible folder
csv_files = glob.glob(os.path.join('C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traversetracks/_complete tracks', "*.csv"))

count = 0 
for f in csv_files:    
    if count == 0: 
        _df = pd.read_csv(f,sep=',', parse_dates=['time'])
        count +=1
    else:
        _df = pd.concat([_df,pd.read_csv(f,sep=',', parse_dates=['time'])])


horizontal_accuracy_median                          = np.median(_df['horizontal_accuracy'])
horizontal_accuracy_q75, horizontal_accuracy_q25    = np.percentile(_df['horizontal_accuracy'], [75 ,25])

vertical_accuracy_median                            = np.median(_df['vertical_accuracy'])
vertical_accuracy_q75, vertical_accuracy_q25        = np.percentile(_df['vertical_accuracy'], [75 ,25])

print(  horizontal_accuracy_median,
        horizontal_accuracy_q75,
        horizontal_accuracy_q25, len(_df))
print(  vertical_accuracy_median,
        vertical_accuracy_q75,
        vertical_accuracy_q25, len(_df))



# Plot the dataframe
ax = _df[['horizontal_accuracy', 'vertical_accuracy']].plot(kind='box', title='boxplot')

# Display the plot
plt.show()


exit()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

data = 'C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/speed_second_windows.csv'
data = pd.read_csv(data,sep=',')

def walkspeed_fit(slope,a,b,c):
    # dz = elevation difference
    # dx = distance
    dzdx = np.tan(np.radians(slope))
    W = (a*np.exp(-abs(b)*abs((dzdx)+c)))
    return W



# define plot space
fig, host = plt.subplots(figsize=(12.5, 6), layout='constrained')
ax2 = host.twinx()
ax3 = host.twinx()

host.set_xlim(0, 2)
host.set_ylim(0, 2)
ax2.set_ylim(0, 4)
ax3.set_ylim(1, 65)



fig, ax1 = plt.subplots(figsize=(12.5, 6))
ax1.set_xlabel('Average Window (s)') 

color = 'black'
ax1.set_ylabel('b-value', color = color) 
ax1.plot(data['window'].values, data['b'].values, color = color) 
ax1.tick_params(axis ='y', labelcolor = color) 
 
# Adding Twin Axes to plot using dataset_2
ax2 = ax1.twinx() 
 
color = 'red'
ax2.set_ylabel('R2', color = color) 
ax2.plot(data['window'].values, data['r2'].values, color = color) 
ax2.tick_params(axis ='y', labelcolor = color) 
 
# Show plot
plt.show()





'''
vertpace_min = []
vertpace_max = []
for indx,val in enumerate(_a):
    a = _a[indx]
    b = _b[indx]
    c = _c[indx]

    # find local minimal
    # downslope
    _v = np.arange(-3,-0.2,0.05)
    _vp = []
    _vs = []
    for _v in _v:
        _vp.append(vertpace(_v,5)[0])
        _vs.append(vertpace(_v,5)[1])
    downslope_min = _vs[np.argmin(_vp)]

    # upslope
    _v = np.arange(0.2,3,0.05)
    _vp = []
    _vs = []
    for _v in _v:
        _vp.append(vertpace(_v,5)[0])
        _vs.append(vertpace(_v,5)[1])
    upslope_min = _vs[np.argmin(_vp)]

    print(data['window'].values[indx],upslope_min,downslope_min)
'''



'''
# define plot space
fig = plt.figure(figsize=(12.5, 6))
grid = plt.GridSpec(1, 6, hspace=0.4, wspace=1.5)
axs0 = fig.add_subplot(grid[0:1, 0:3])
axs1 = fig.add_subplot(grid[0:1, 3:6])
axs1.set_xlabel('Slope (degrees)')
axs1.set_ylabel('velocity (km/hr)')
axs1.set_xlim([-65,65])
axs1.set_ylim([-0.25,2.2])


start = 0
stop = 0.8
number_of_lines= len(_a)
cm_subsection = np.linspace(start, stop, number_of_lines) 
colours = [ cm.inferno(x) for x in cm_subsection ]
    
for indx,val in enumerate(_a):
    axs1.plot(  np.arange(-50,50,0.05), 
                walkspeed_fit(np.arange(-50,50,0.05), _a[indx],_b[indx],_c[indx]),
                c=colours[indx],
                label=str(data['window'].values[indx]))

plt.legend(prop={'size': 6},handlelength=1)
plt.show()
'''

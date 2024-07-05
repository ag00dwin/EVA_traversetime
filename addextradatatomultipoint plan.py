
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, constrained_layout=True,figsize=(4,6)) 
indx = 0
plotarea = [ax1,ax2,ax3]

def myround(x, base=1):
    return base * round(x/base)

path  = 'C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/ouputs'
for (root, dirs, file) in os.walk(path):
    for f in file:
        if '.csv' in f:
            output_compilations = pd.read_csv(path+'/'+f)

            all_time = output_compilations['time'].values
            all_dist = output_compilations['dist'].values
            all_dpz  = output_compilations['dZp'].values

            # plot all data
            plotarea[indx].scatter(all_dist,all_dpz,c=all_time)
            plotarea[indx].set_xlabel('Distance (m)')
            plotarea[indx].set_ylabel('Elevation Gain (m)')
        
            if ((int(all_time.max())))-(int(all_time.min()))>20:
                ticks = np.linspace(int(all_time.min()), int((all_time.max())), 10, endpoint=True)
            else:
                ticks = np.around(np.linspace(int(all_time.min()), int((all_time.max())), 5, endpoint=True),decimals=0)
    
            
            fig.colorbar(plotarea[indx].scatter(all_dist,all_dpz,c=all_time),
                         ax=plotarea[indx],fraction=0.046, pad=0.04,ticks=ticks)

            # plot most efficient 
            print(output_compilations['indx'][np.argmin(all_time)],f,np.argmin(all_time),np.argmin(all_dist),np.argmin(all_dpz))
            plotarea[indx].scatter(all_dist[np.argmin(all_time)],all_dpz[np.argmin(all_time)],
                                    s=100,
                                    facecolors='none', 
                                    edgecolors='r')


            indx +=1
      
            
   
            # normalise
            # for index,value in enumerate(all_time):

plt.show()

'''
#review output of multipoint and correct

import os
import pandas as pd
import numpy as np

# import elevation map as csv
dem_file = '_raster/_expzone_DEM.csv'
dem = pd.read_csv(dem_file,sep=',')
# define array variables from DEM
dem_z = dem['VALUE'].values
dem_x = dem['X'].values
dem_y = dem['Y'].values
# create list of coorinate pairs from dem
amap_coordinates = np.array(list(zip(dem_x,dem_y)))
# create function to match coorindates to closest coorindates in slope/dem
def c_index(X,Y):
    distances = np.linalg.norm(amap_coordinates-np.array((X,Y)), axis=1)
    min_index = np.argmin(distances)
    # return [0] index and [1] distance difference closest point
    return min_index, distances[min_index]

path  = 'C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/traverse_permutations_C/traverses'

newpath = 'C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/traverse_permutations_C/traverses_updated'
if not os.path.exists(newpath):
    os.makedirs(newpath)

all_index   = []
all_dzp     = []

for (root, dirs, file) in os.walk(path):
    for f in file:
        if '.csv' in f:
            route_index = (f.split('_')[-1]).replace('.csv','')
            _df = pd.read_csv(path+'/'+f)

            mapgrid_z = []  # elevation of each point
            for index, row in _df.iterrows():
                mapgrid_z.append(dem_z[c_index(_df['X'][index],_df['Y'][index])[0]])
            
            diff_z      = []
            diff_z_plus = []
            diff_z_neg  = []
            for ind,val in enumerate(mapgrid_z):
                if ind == 0:
                    diff_z.append(0)
                    diff_z_plus.append(0)
                    diff_z_neg.append(0)
                else:
                    diff_z.append(mapgrid_z[ind]-mapgrid_z[ind-1])
                    if (mapgrid_z[ind]-mapgrid_z[ind-1]) > 0:
                        diff_z_plus.append(diff_z_plus[-1]+mapgrid_z[ind]-mapgrid_z[ind-1])
                        diff_z_neg.append(diff_z_neg[-1])
                    else:
                        diff_z_plus.append(diff_z_plus[-1])
                        diff_z_neg.append(diff_z_neg[-1]+(mapgrid_z[ind]-mapgrid_z[ind-1]))       

            _df['Z']    = np.array(mapgrid_z)
            _df['dZ']   = np.array(diff_z)
            _df['dZp']  = np.array(diff_z_plus)
            _df['dZn']  = np.array(diff_z_neg)

            _df.to_csv(newpath+'/'+f)

            all_index.append(route_index)
            all_dzp.append(np.max(np.array(diff_z_plus)))

new_data = pd.DataFrame({   'indx': np.array(all_index),
                            'dZp': np.array(all_dzp),
                            })
new_data.sort_values('indx')

output_df = pd.read_csv('C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/traverse_permutations_C/output.csv')

output_df['indx_2'] = new_data['indx']
output_df['dZp']    = new_data['dZp']

# save dataframe
output_df.to_csv('C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/EVA_traverse_time_output/traverse_permutations_C/output_updated.csv', sep=',')

'''
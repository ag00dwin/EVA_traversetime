import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

'''
# read the image of a plant seedling as grayscale from the outset
plant_seedling = iio.imread('C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH/Traverse Research/_surface_rougness tiff.TIF')

# convert the image to float dtype with a value range from 0 to 1
#plant_seedling = ski.util.img_as_float(plant_seedling)

# create the histogram
histogram, bin_edges = np.histogram(plant_seedling, bins=100, range=(0, 1))

# configure and draw the histogram figure
fig, ax = plt.subplots()
ax.set_title("Grayscale Histogram")
ax.set_xlabel("grayscale value")
ax.set_ylabel("pixel count")
#ax.set_xlim([0.005, 1])  # <- named arguments do not work here
#ax.set_ylim([0, 1000])  # <- named arguments do not work here

ax.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()
'''
from matplotlib.image import imread
image = imread('C:/Users/Arthu/Desktop/_SPACE HEALTH RESEARCH\Traverse Research/_surface_rougness tiff.tif')
print(image)
print(np.mean(image))
print(np.median(image))

exit()

# read the image of a plant seedling as grayscale from the outset
plant_seedling = iio.imread('C:/Users/Arthu/Desktop/EVA_traversetime/_raster/_expzone_slope.tif')
plant_seedling[plant_seedling==0] = np.nan

total_len = np.count_nonzero(~np.isnan(plant_seedling))

_zero   = [] #0-5
_five   = [] #5-10
_ten    = [] #10-15
_fift   = [] #15-20
_twen   = [] #20+

_len = []



for r in plant_seedling:
    for i in r:
        if i is np.nan:
            pass 
        else:
            _len.append(i)
            if i > 20:
                _twen.append(i)
            elif i > 15:
                _fift.append(i) 
            elif i > 10:
                _ten.append(i) 
            elif i > 5:
                _five.append(i) 
            else:
                _zero.append(i)

total_len = len(_len)

print((len(_zero)/total_len)*100,'0-5')
print((len(_five)/total_len)*100,'5-10')
print((len(_ten)/total_len)*100,'10-15')
print((len(_fift)/total_len)*100,'15-20')
print((len(_twen)/total_len)*100,'20+')
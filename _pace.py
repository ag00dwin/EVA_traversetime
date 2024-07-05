import matplotlib.pyplot as plt
import numpy as np

a = 1.58
b = -3.04
c = 0

# define plot space
fig = plt.figure(figsize=(12.5, 6))
grid = plt.GridSpec(1, 6, hspace=0.4, wspace=1.5)
axs0 = fig.add_subplot(grid[0:1, 0:3])
axs1 = fig.add_subplot(grid[0:1, 3:6])
# format axis
axs0.set_xlabel('slope vector (degrees)')
axs0.set_ylabel('pace (hr/km)')

axs1.set_xlabel('slope vector (degrees)')
axs1.set_ylabel('vert pace (hr/km)')
# plot hiking function
def pace(dz,dx):
    P = a*np.exp((-1*b)*(abs(dz/dx)+c))
    S = np.degrees(np.arctan(dz/dx))
    return P, S
def vertpace(dz,dx):
    P = a*np.exp((-1*b)*(abs(dz/dx)+c))
    P = abs(P*(dx/dz))
    S = np.degrees(np.arctan(dz/dx))
    return P, S


pacer = []
slope = []
for h in np.arange(-3,3,0.05):
    s1,s2 = pace(h,5)
    pacer.append(s1)
    slope.append(s2)
axs0.plot(slope,pacer,c='red')

vpacer = []
slope = []
for h in np.concatenate((np.arange(-3,-0.3,0.05),np.arange(0.3,3,0.05))):
    s1,s2 = vertpace(h,5)
    vpacer.append(s1)
    slope.append(s2)
axs1.plot(slope,vpacer,c='red')
# find local minimal
# downslope
_v = np.arange(-3,-0.2,0.05)
_vp = []
_vs = []
for _v in _v:
    _vp.append(vertpace(_v,5)[0])
    _vs.append(vertpace(_v,5)[1])
print('downslope_min: ',_vs[np.argmin(_vp)])
axs1.scatter(_vs[np.argmin(_vp)],np.min(_vp),c='black')
axs1.annotate(  str(round(_vs[np.argmin(_vp)],2)),
                xy          =(_vs[np.argmin(_vp)], np.min(_vp)), 
                xytext      =(_vs[np.argmin(_vp)], np.min(_vp)+1), )

# upslope
_v = np.arange(0.2,3,0.05)
_vp = []
_vs = []
for _v in _v:
    _vp.append(vertpace(_v,5)[0])
    _vs.append(vertpace(_v,5)[1])
print('upslope_min: ',_vs[np.argmin(_vp)])
axs1.scatter(_vs[np.argmin(_vp)],np.min(_vp),c='black')
axs1.annotate(  str(round(_vs[np.argmin(_vp)],2)),
                xy          =(_vs[np.argmin(_vp)], np.min(_vp)), 
                xytext      =(_vs[np.argmin(_vp)], np.min(_vp)+1), )
plt.show()
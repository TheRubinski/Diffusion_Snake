import numpy as np
from scipy import ndimage

a = np.array([[0, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 1, 1, 0],
              [1, 1, 0, 0]])

k = np.array([[0,1,0],
              [1,0,1],
              [0,1,0]])



print(a*ndimage.convolve(a, k, mode='constant', cval=0.0)) # constant padding with 0.0





b = np.array([[0,1,2],
              [3,4,5],
              [6,7,8]])


uk = b
spline = k

smid=slice(1,-1);sup=slice(2,None);sdown=slice(None,-2)             # XXX right/ left missing ???
neighbourslices=[(sup,smid),(sdown,smid),(smid,sup),(smid,sdown)]   # drunter, drüber, rechts, links


print(neighbourslices)

uk = puk = np.pad(uk, pad_width=1, constant_values=0)#padded uk
# uk=sum(uk[j] for j in neighbourslices)


print(uk)

for j in neighbourslices:
    print(j)
    print(uk[j])
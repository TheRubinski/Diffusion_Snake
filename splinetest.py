import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt


x,y=zip(*[(1,1),(1,0),(0,0),(0,1)])
x=list(x)
y=list(y)
x.append(x[0])
y.append(y[0])
# append the starting x,y coordinates


# fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
# is needed in order to force the spline fit to pass through all the input points.
tck, u = interpolate.splprep([x, y], s=0, per=1)

# evaluate the spline fits for 1000 evenly spaced distance values
xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

# plot the result

plt.plot(x, y, 'or')
plt.plot(xi, yi, '-b')
plt.show()
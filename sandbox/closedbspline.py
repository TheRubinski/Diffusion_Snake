import numpy as np
from scipy import interpolate
# Example control points

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cicleBspline import Spline



c = np.array([[0, 0], [1, 1], [2, -1], [3, 0]])
def generate_points_in_circle(num_points,radius=1,center=[(0,0)]):
    theta = np.linspace(0, 2 * np.pi, num_points,endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack((x, y), axis=-1)*radius+center

n=4
c=generate_points_in_circle(n,(4,1))+np.random.rand(n,2)*0.4
# Degree of the B-spline
k = 2
#[-0.75 -0.5  -0.25  0.    0.25  0.5   0.75  1.    1.25  1.5   1.75]
# Create the knot vector
#num_control_points = len(c)
num_knots = n +k*2+1

# Create a uniform knot vector for an open curve
#t = np.linspace(0, 1, num_knots)
#print(t)
t = np.linspace(-k/n, 1+k/n, num_knots)
print(t)
#print(t)
#print((c[-k:],c,c[:k]))
c_closed=np.vstack((c,c[:k]))
#print(c_closed)


# Create the closed B-spline curve
C = interpolate.BSpline(t, c_closed,k,extrapolate= "periodic")
#C = BSpline()
print(C(1))

# Evaluate the curve from t=0 to t=1
t_values = np.linspace(0, 1, 100,endpoint=False)
curve_points = C(t_values)


# Plot the curve
import matplotlib.pyplot as plt
plt.plot(curve_points[:,0], curve_points[:,1], label='Closed B-spline Curve', alpha=0.5)
plt.scatter(c[:,0], c[:,1], c='red', label='Control Points')
plt.scatter(curve_points[:,0], curve_points[:,1], c="b", alpha=0.5)
print(C(0),C(1))
plt.legend()
plt.show()
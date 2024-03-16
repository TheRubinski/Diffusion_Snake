import numpy as np
from scipy import interpolate
# Example control points



from bsplineclass import Spline




c = np.array([[0, 0], [1, 1], [2, -1], [3, 0]])
def generate_points_in_circle(num_points,radius=1,center=[(0,0)]):
    theta = np.linspace(0, 2 * np.pi, num_points,endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack((x, y), axis=-1)*radius+center

n=20
c=generate_points_in_circle(n)+np.random.rand(n,2)*0.4
# Degree of the B-spline
k = 3

C = Spline(c=c,k=k)
#C = BSpline()


# Evaluate the curve from t=0 to t=1
t_values = np.linspace(0, 1, 100,endpoint=False)
curve_points = C.spline(t_values)


# Plot the curve
import matplotlib.pyplot as plt
plt.plot(curve_points[:,0], curve_points[:,1], label='Closed B-spline Curve', alpha=0.5)
plt.scatter(c[:,0], c[:,1], c='red', label='Control Points')
plt.scatter(curve_points[:,0], curve_points[:,1], c="b", alpha=0.5)

plt.legend()
plt.show()
import numpy as np
from scipy import interpolate
# Example control points
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

# Create the knot vector
num_control_points = len(c)
num_knots = num_control_points +k+1

# Create a uniform knot vector for an open curve
t = np.linspace(0, 1, num_knots)
t=np.concatenate((t, t[:k+1]))
# Adjust the control points for a closed curve
#c_closed = np.concatenate((c, c[:k+1]))
c_closed=np.vstack((c,(c[0],)))

# Create the closed B-spline curve
#C = interpolate.BSpline(t, c_closed,k,extrapolate= "periodic")
C = interpolate.BSpline(t, c_closed,k,extrapolate="periodic")


# Evaluate the curve from t=0 to t=1
t_values = np.linspace(0, 1, 100)
curve_points = C(t_values)

# Plot the curve
import matplotlib.pyplot as plt
plt.plot(curve_points[:,0], curve_points[:,1], label='Closed B-spline Curve', alpha=0.5)
plt.scatter(c[:,0], c[:,1], c='red', label='Control Points')
plt.legend()
plt.show()
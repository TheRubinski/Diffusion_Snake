# import numpy as np
# from bsplineclass import Spline


# def generate_points_in_circle(num_points,radius=1,center=[(0,0)]):
#     theta = np.linspace(0, 2 * np.pi, num_points,endpoint=False)
#     x = np.cos(theta)
#     y = np.sin(theta)
#     return np.stack((x, y), axis=-1)*radius+center

# C=Spline(generate_points_in_circle(10))
# print(C.spline.t)




import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
np.set_printoptions(linewidth = 300)
# Sample data
x_data = np.linspace(0, 10, 10)
y_data = np.sin(x_data)

# Define degree of the B-spline
k = 3

# Create B-spline interpolation
bspline = make_interp_spline(x_data, y_data, k=k)

# Construct design matrix
design_matrix = bspline.design_matrix(x_data, bspline.t, k)
print(design_matrix.toarray())
# Evaluate spline
spline_values = design_matrix.dot(bspline.c)

# Plot original data points
plt.scatter(x_data, y_data, label='Data')
print(bspline.c)
print(y_data)

# Plot spline
plt.plot(x_data, spline_values, label='B-Spline')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('B-Spline Interpolation')
plt.legend()
plt.grid(True)
plt.show()
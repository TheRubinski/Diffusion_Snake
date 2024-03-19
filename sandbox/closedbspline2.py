import numpy as np
from scipy import interpolate
# Example control points



from bsplineclass import Spline









def wrap_array(arr, wrap_length):
    # Split the array into two parts
    # first_part = arr[:-wrap_length].copy()
    # second_part = arr[-wrap_length:]
    # first_part[:len(second_part)]+=second_part

    first_part = arr[:,:-wrap_length].copy()
    second_part = arr[:,-wrap_length:]
    #print(first_part)
    #print(second_part)
    second_length = second_part.shape[1]
    #print(first_part[:,:second_length])
    first_part[:,:second_length] += second_part

    return first_part

# Example usage
arr = np.array([[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10]])
wrap_length = 2
wrapped_arr = wrap_array(arr, wrap_length)
print(wrapped_arr)



c = np.array([[0, 0], [1, 1], [2, -1], [3, 0]])
def generate_points_in_circle(num_points,radius=1,center=[(0,0)]):
    theta = np.linspace(0, 2 * np.pi, num_points,endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack((x, y), axis=-1)*radius+center



n=10
c=generate_points_in_circle(n)#+np.random.rand(n,2)*0.1
# Degree of the B-spline
k =4

C = Spline(c=c,k=k)
#C = BSpline()


# Evaluate the curve from t=0 to t=1



offset=-(k-1)*(1/len(c)/2)
spaced=np.linspace(offset, 1+offset, len(c),endpoint=False)%1
print(spaced)
m=C.designmatrix(spaced,wrap=True)
print(m)
print(np.linalg.inv(m))



t_values = np.linspace(0, 1, 10000)
curve_points = C.spline(t_values)
curve_points2 = C.spline(spaced)

# Plot the curve
import matplotlib.pyplot as plt
plt.plot(curve_points[:,0], curve_points[:,1], label='Closed B-spline Curve', alpha=0.5)
plt.scatter(c[:,0], c[:,1], c='red', label='Control Points')
plt.scatter(curve_points2[:,0], curve_points2[:,1], c="b", alpha=0.5)

plt.legend()
plt.show()
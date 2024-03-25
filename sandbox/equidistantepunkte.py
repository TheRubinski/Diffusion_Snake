
import numpy as np
from scipy import interpolate
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cicleBspline import Spline



def generate_points_in_circle_bad(num_points,radius=1,center=[(0,0)]):
    theta = np.linspace(0, 1, num_points,endpoint=False)**2
    theta*=2 * np.pi
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack((x, y), axis=-1)*radius+center



c=generate_points_in_circle_bad(100)#+np.random.rand(n,2)*0.1
# Degree of the B-spline
k =2

C = Spline(c,k=k)
t_values = np.linspace(0, 1, 1000)
curve_points = C.spline(t_values)
#print(np.linalg.norm(curve_points[0]-curve_points[-1]))# first and last are he same d.h. shoild be 0


#distsum=0
#dists=[0]
#for i in range(len(curve_points)-1):
#    distsum+=np.linalg.norm(curve_points[i]-curve_points[i+1])
#    dists.append(distsum)
#dists=np.array(dists)/distsum#normalize dists from 0 to 1

#chatgptcode
distances = np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1)
cumulative_distances = np.cumsum(distances)
normalized_distances = cumulative_distances / cumulative_distances[-1]
normalized_distances = np.insert(normalized_distances, 0, 0)
dists=normalized_distances

#newspacings=np.linspace(0, 1, 100,endpoint=False)
#points=[]
#for index,x in zip(np.searchsorted(dists,newspacings),newspacings):
#    x0,x1=dists[index-1],dists[index]
#    y0,y1=curve_points[index-1],curve_points[index]
#    #linear interpolation
#    y=(y0*(x1-x)+y1*(x-x0))/(x1-x0)# 
#    points.append(y)

#this does the same as the loop
x=np.linspace(0, 1, 100,endpoint=False)
index=np.searchsorted(dists,x)
x0,x1=dists[index-1],dists[index]
y0,y1=curve_points[index-1],curve_points[index]
y=(y0*(x1-x)[:,None]+y1*(x-x0)[:,None])/(x1-x0)[:,None]# linear interpolation
points=y

#print(np.linalg.norm(points[0]-points[-1]))#should not be 0



print(points)









# Plot the curve
import matplotlib.pyplot as plt
plt.plot(curve_points[:,0], curve_points[:,1], label='Closed B-spline Curve', alpha=0.5)
plt.scatter(c[:,0], c[:,1], c='red', label='Control Points')
#plt.scatter(curve_points2[:,0], curve_points2[:,1], c="b", alpha=0.5)
plt.scatter(*points.T, c="b")
plt.legend()
plt.show()
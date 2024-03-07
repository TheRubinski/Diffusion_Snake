import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from skimage.draw import polygon
from matplotlib import animation

def drawspline(spline,canvas=None,steps=1000):
    if canvas is None:
        canvas = np.zeros((100, 100), bool)
    xi,yi = spline(np.linspace(0, 1, steps)).T

    rr, cc = polygon(xi[:-1],yi[:-1], canvas.shape)
    #canvas[rr,cc] = 1
    canvas[cc,rr] = 1
    return canvas,xi,yi


def makespline(points):
    #points=np.array(points)
    points=np.vstack((points,(points[0],)))
    spline = interpolate.make_interp_spline(np.linspace(0, 1, points.shape[0]),points, k=3,bc_type="periodic")
    
    return spline




pointnum=5
points= np.random.uniform(10,90, size=(pointnum,2))
pointssoll=np.copy(points)

pointsplt,=plt.plot([],[], 'or')#plot points
splineplt,=plt.plot([], [], '-b')#plot spline
imgplt=plt.imshow(np.zeros((100, 100), bool),vmin=0,vmax=1,animated=True)


def animate(f):

    for i in range(len(points)):
        current_point = points[i]
        goal_point = pointssoll[i]
        direction = goal_point - current_point
        if np.linalg.norm(direction) < 1:
            pointssoll[i] = np.random.uniform(10,90, size=2)
        else:
            points[i] += 1 * direction / np.linalg.norm(direction)

    x,y=np.array(points).T
    pointsplt.set_data(x, y)
    
    
    spline=makespline(points.astype(int))#here are somtimes exceptions if points are to close or something
    img,xi,yi=drawspline(spline,steps=1000)

    splineplt.set_data(xi, yi)
    imgplt.set_array(img)

    #[[72.3394304  76.3429846 ]
    # [50.52473444 27.2649546 ]
    # [50.99123604 27.93682006]
    # [81.459163   57.33716789]
    # [54.21997677 24.46773367]]

    return [imgplt,pointsplt,splineplt]

 
anim = animation.FuncAnimation(plt.gcf(), animate, interval=10,cache_frame_data=False,blit=True)
plt.show()
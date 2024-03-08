import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from skimage.draw import polygon,polygon_perimeter
from matplotlib import animation

def drawspline(spline,canvas=None,steps=1000):
    if canvas is None:
        canvas = np.zeros((100, 100), int)
    xi,yi = spline(np.linspace(0, 1, steps)).T

    px,py=xi[:-1],yi[:-1]
    rr, cc = polygon(px,py, canvas.shape)
    #canvas[rr,cc] = 1
    canvas[cc,rr] = 2
    rr, cc = polygon_perimeter(px,py, canvas.shape)
    canvas[cc,rr] = 1
    return canvas,xi,yi


def makespline(points):
    #points=np.array(points)
    points=np.vstack((points,(points[0],)))
    spline = interpolate.make_interp_spline(np.linspace(0, 1, points.shape[0]),points, k=3,bc_type="periodic")
    
    return spline

def polygon_direction(points):
    direction_sum = 0
    num_points = len(points)
    for i in range(num_points):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % num_points]  # Wrap around to the first point for the last edge
        direction_sum += (x2 - x1) * (y2 + y1)
    if direction_sum > 0:
        return 1#"Clockwise"
    elif direction_sum < 0:
        return -1#"Counter-Clockwise"
    else:
        return 0#"Collinear"


pointnum=5
points= np.random.uniform(10,90, size=(pointnum,2))
pointssoll=np.copy(points)

fig=plt.figure()
pointsplt,=plt.plot([],[], 'or')#plot points
splineplt,=plt.plot([], [], '-b')#plot spline
#cplt,=plt.plot([],[], 'og')#plot spline.c
normalplt=plt.quiver(*points.T,[0]*len(points),[0]*len(points),angles='xy')
#cplt2,=plt.plot([], [], '-g')#plot spline
imgplt=plt.imshow(np.zeros((100, 100), int),vmin=0,vmax=2,animated=True)



def animate(f):

    for i in range(len(points)):
        current_point = points[i]
        goal_point = pointssoll[i]
        direction = goal_point - current_point
        if np.linalg.norm(direction) < 1:
            pointssoll[i] = np.random.uniform(10,90, size=2)
        else:
            points[i] += 1 * direction / np.linalg.norm(direction)


    
    
    spline=makespline(points)#here are somtimes exceptions if points are to close or something
    img,xi,yi=drawspline(spline,steps=1000)

    splineplt.set_data(xi, yi)
    imgplt.set_array(img)

    x,y=points.T
    pointsplt.set_data(x, y)

    #ps=spline(np.linspace(0,1,len(points),endpoint=False))
    dx,dy=spline(np.linspace(0,1,len(points),endpoint=False),1).T
    r=polygon_direction(points)
    normalplt.set_offsets(points)
    normalplt.set_UVC(-dy*r,dx*r)



    #cplt2.set_data(x,y)
    #[[72.3394304  76.3429846 ]
    # [50.52473444 27.2649546 ]
    # [50.99123604 27.93682006]
    # [81.459163   57.33716789]
    # [54.21997677 24.46773367]]
    return [imgplt,pointsplt,splineplt,normalplt]
    #return [imgplt,pointsplt,splineplt,cplt,cplt2]

 
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()
import numpy as np
from scipy import interpolate
from skimage.draw import polygon,polygon_perimeter


class Spline:
    def __init__(self, points, k=3):
        self.k = k
        self.setpoints(points)
    
    def setpoints(self, points):
        self.points=np.array(points)
        closedpoints=np.vstack((self.points,(self.points[0],)))
        self.spline = interpolate.make_interp_spline(np.linspace(0, 1, closedpoints.shape[0]),closedpoints, k=self.k,bc_type="periodic")

    def set_c(self, points):
        self.spline.c = points

    def designmatrix(self, points):
        # In each row of the design matrix all the basis elements are evaluated at the certain point (first row - x[0], …, last row - x[-1]).
        return self.spline.design_matrix(points, self.spline.t, self.k).toarray()

    @staticmethod
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

    def draw(self,canvas=None,steps=1000,drawinside=True):
        if canvas is None:
            canvas = np.zeros((100, 100), int)
        xi,yi = self.spline(np.linspace(0, 1, steps)).T

        px,py=xi[:-1],yi[:-1]
        if drawinside:
            rr, cc = polygon(px,py, canvas.shape)
            canvas[cc,rr] = 2
        rr, cc = polygon_perimeter(px,py, canvas.shape)
        canvas[cc,rr] = 1
        return canvas,xi,yi
    
    def get_masks(self,canvas=None,steps=1000):
        if canvas is None:
            in_mask = np.zeros((100, 100), int)
        else:
            in_mask = np.zeros((canvas.shape), int)
        spline_mask = np.copy(in_mask)

        xi,yi = self.spline(np.linspace(0, 1, steps)).T
        px,py=xi[:-1],yi[:-1]
        rr, cc = polygon(px,py, in_mask.shape)               # in + also some on spline
        in_mask[cc,rr] = 1
        rr, cc = polygon_perimeter(px,py, in_mask.shape)     # only spline curve
        spline_mask[cc,rr] = 1
        in_mask = np.logical_and(in_mask, np.logical_not(spline_mask))  # remove elements also in spline 
        out_mask = np.logical_not(np.logical_and(in_mask, spline_mask))
        return in_mask, out_mask, spline_mask
    
    def get_2masks(self,canvas=None,steps=1000):
        if canvas is None:
            in_mask = np.zeros((100, 100), int)
        else:
            in_mask = np.zeros((canvas.shape), int)

        xi,yi = self.spline(np.linspace(0, 1, steps)).T
        px,py=xi[:-1],yi[:-1]
        rr, cc = polygon(px,py, in_mask.shape)               # in + also some on spline
        in_mask[cc,rr] = 1

        out_mask = np.logical_not(in_mask)
        return in_mask, out_mask
    
    def normals(self, points=None):
        if points is None:
            points = np.linspace(0,1,len(self.points),endpoint=False)
        dx,dy=self.spline(points,1).T
        r=Spline.polygon_direction(self.points)
        return -dy*r,dx*r
    
    

if __name__=="__main__":
    from matplotlib import pyplot as plt
    from matplotlib import animation
    pointnum=4
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


        
        
        spline=Spline(points)
        img,xi,yi=spline.draw(steps=1000)

        splineplt.set_data(xi, yi)
        imgplt.set_array(img)

        x,y=points.T
        pointsplt.set_data(x, y)

        normalplt.set_offsets(points)
        normalplt.set_UVC(*spline.normals())


        return [imgplt,pointsplt,splineplt,normalplt]


    
    anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
    plt.show()
# This script shall contain the final demo for presentation in the end ...
import numpy as np
from src.cicleBspline import Spline
from skimage import io, color
from matplotlib import pyplot as plt
from src.diffusionSnake import DiffusionSnake


# XXX Config 1 - working good for demo
# image_path = './sample_images/snail1_gray.png'
# v = 0.01 
# n_points = 100        # number of controllpoints for spline
# alph=0.9              # learning rate

# Config
image_path = './sample_images/snail1_gray.png'
v = 0.01
n_points = 100  # number of controllpoints for spline
alph=0.9        # learning rate 

# Setup
ds = DiffusionSnake(image_path, v, n_points, alph, mode="simple", respace=True)
u,x,y=ds.draw()

# Plot/ Animate
from matplotlib import animation
fig=plt.figure()
uplt=plt.imshow(u)
Cplt,=plt.plot(x,y,"-b")
print_step = plt.text(.05, .99, "Step: 0", ha='left', va='top')
cplt,=plt.plot(*ds.C.c.T,"or")          # plot controllpoints


def animate(frame):
    print_step.set_text(f"Step: {ds.n_step}")
    
    for i in range(10):
        ds.step()
    u,x,y=ds.draw()

    uplt.set_array(u.T * ds.f.T)
    Cplt.set_data(x,y)
    cplt.set_data(*ds.C.spline(np.linspace(0,1,100)).T) # plot controllpoints


    return[uplt,Cplt,print_step,cplt]

animate(0)
animate(1)
plt.pause(0.5)
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()
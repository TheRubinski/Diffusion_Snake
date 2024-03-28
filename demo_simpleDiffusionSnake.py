# This script shall contain the final demo for presentation in the end ...
import numpy as np
from matplotlib import pyplot as plt
from src.diffusionSnake import DiffusionSnake


# XXX Config 1 - working good for demo
# image_path = './sample_images/snail1_gray.png'
# v = 0.01 
# n_points = 100        # number of controllpoints for spline
# alph=0.9              # learning rate

# Config
image_path = './sample_images/artificial/100/rect_1.png'
v = 0.01
n_points = 100  # number of controllpoints for spline
alph=0.2        # learning rate 

# Setup
ds = DiffusionSnake(image_path, v, n_points, alph, mode="simple", respace=True)





# Plot/ Animate
u,x,y=ds.draw()
f = ds.f

from matplotlib import animation
fig, axes = plt.subplots(1,2)
plt.subplot(1,2,1)
uplt=plt.imshow(u)
print_step = plt.text(.05, .99, "Step: 0", ha='left', va='top')
Cplt,=plt.plot(x,y,"-b", label='Spline')
cplt,=plt.plot(*ds.C.c.T,"or", label='c-points')          # plot controllpoints

plt.subplot(1,2,2)
fplt=plt.imshow(f.T)
C2plt,=plt.plot(x,y,"-b", label='Spline')
axes[0].set_title('u-function')
axes[1].set_title('Input Image')

a1=axes[0].legend()
a2=axes[1].legend()


def animate(frame):
    print_step.set_text(f"Step: {ds.n_step}")
    
    for i in range(1):# range(10) -> ca 5x speedup
        ds.step()
    u,x,y=ds.draw()

    #uplt.set_array((u * ds.f).T)         # overlay u and f
    uplt.set_array(u.T)
    Cplt.set_data(x,y)
    C2plt.set_data(x,y)
    cplt.set_data(*ds.C.spline(np.linspace(0,1,100)).T) # plot controllpoints
    return[uplt,Cplt,C2plt,print_step,cplt,a1,a2]

animate(0)
animate(1)
plt.pause(0.5)
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()
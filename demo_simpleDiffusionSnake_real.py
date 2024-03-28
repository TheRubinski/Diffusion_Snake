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
# ./sample_images/real/1200/ is 1200 x 1200 pixels ./sample_images/real/900 is 900 x 900 pixels ...
# 
# image_path = './sample_images/real/1200/switch_1.png'       # 1200 x 1200 pixels very slow, but working
# image_path = './sample_images/real/100/hand_1.png'          # somehow working --> goes out of bounds
image_path = './sample_images/real/300/hand_1.png'          # somehow working
image_path = './sample_images/real/100/hand_2.png'          # poorly working, but a hard one, does what it should
image_path = './sample_images/real/300/hand_2.png'          # poorly working, but a hard one, does what it should
image_path = './sample_images/real/300/bee.png'             # working good


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
#cplt,=plt.plot(*ds.C.c.T,"or")          # plot controllpoints


def animate(frame):
    global u,C,step,print_step
    print_step.set_text(f"Step: {ds.n_step}")
    
    for i in range(10): ds.step()   # do 10 steps in each frame
    u,x,y=ds.draw()

    uplt.set_array(u.T * ds.f.T)    # overlay u and f
    Cplt.set_data(x,y)
    #cplt.set_data(*ds.C.spline(np.linspace(0,1,100)).T) # plot controllpoints

    return[uplt,Cplt,print_step]#,cplt]

animate(0)
animate(1)
plt.pause(0.5)
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()
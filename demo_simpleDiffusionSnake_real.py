from matplotlib import pyplot as plt
from src.diffusionSnake import DiffusionSnake
import numpy as np


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
image_path = './sample_images/real/100/hand_1.png'          # somehow working

v = 0.01
n_points = 100  # number of controllpoints for spline
alph=0.9        # learning rate 

# Setup
ds = DiffusionSnake(image_path, v, n_points, alph, mode="simple", respace=True)
u,x,y=ds.draw()

# Plot/ Animate
f = ds.f

from matplotlib import animation
fig, axes = plt.subplots(1,2)
plt.subplot(1,2,1)
uplt=plt.imshow(u)
print_step = plt.text(.05, .99, "Step: 0", ha='left', va='top')
Cplt,=plt.plot(x,y,"-b", label='Spline')

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
    return[uplt,Cplt,C2plt,print_step,a1,a2]

animate(0)
animate(1)
plt.pause(0.5)
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()
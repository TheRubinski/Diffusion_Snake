# This script shall contain the final demo for presentation in the end ...
import numpy as np
from src.cicleBspline import Spline
from skimage import io, color
from matplotlib import pyplot as plt
from src.diffusionSnake import DiffusionSnake
import time

# n_steps = 1000  # max iterations  # TODO: implement
# eps = 1e-4      # for convergence # TODO: implement

# # Config - Example Two: Converges somehow stable in 400 steps
image_path = './sample_images/artificial/100/snail1.png' # snail1.png'
image_path = './sample_images/artificial/100/rect_1.png'
lambd, v= 3, 0.03  # Parameters for Diffusion Snake
n_points = 50  # number of controllpoints for spline
alph=0.3        # learning rate 
u_iter=12
tau=0.25


# image_path = './sample_images/real/1200/switch_1.png'       # 1200 x 1200 pixels very slow, but working
# image_path = './sample_images/real/100/hand_1.png'          # somehow working --> goes out of bounds
# image_path = './sample_images/real/300/hand_1.png'          # somehow working
# image_path = './sample_images/real/100/hand_2.png'          # poorly working, but a hard one, does what it should
# image_path = './sample_images/real/300/hand_2.png'          # poorly working, but a hard one, does what it should

# image_path = './sample_images/real/100/bee.png'             # working
# lambd, v= 3, 0.01  # Parameters for Diffusion Snake
# n_points = 50  # number of controllpoints for spline
# alph=0.3        # learning rate 
# u_iter=1
# tau=0.25

image_path = './sample_images/real/300/bee.png'             # working 
lambd, v= 15, 0.01  # Parameters for Diffusion Snake
n_points = 100  # number of controllpoints for spline
alph=0.9        # learning rate 
u_iter=12
tau=0.25


# Setup
ds = DiffusionSnake(image_path, v, n_points, alph, respace=True, mode="full", lambd=lambd, u_iterations=u_iter, tau=tau)
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
    
    for i in range(10):#ca 5x speedup
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
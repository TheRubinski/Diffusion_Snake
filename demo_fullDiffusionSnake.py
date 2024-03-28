# This script shall contain the final demo for presentation in the end ...
import numpy as np
from src.cicleBspline import Spline
from skimage import io, color
from matplotlib import pyplot as plt
from src.diffusionSnake import DiffusionSnake
import time

# n_steps = 1000  # max iterations  # TODO: implement
# eps = 1e-4      # for convergence # TODO: implement

# # Config - Example One: Converges somehow stable in 1400 steps
# image_path = './sample_images/rect_1.png'
# lambd, v= 7, 0.5  # Parameters for Diffusion Snake
# n_points = 100  # number of controllpoints for spline
# alph=0.1        # learning rate 
# u_iter=12
# tau=0.2

# # Config - Example Two: Converges somehow stable in 400 steps
image_path = './sample_images/snail1.png' # snail1.png'
lambd, v= 3, 0.03  # Parameters for Diffusion Snake
n_points = 50  # number of controllpoints for spline
alph=0.3        # learning rate 
u_iter=12
tau=0.25

# Config - Example Two: Converges somehow stable in 400 steps
image_path = './sample_images/rect_1.png'
lambd, v= 7, 0.1  # Parameters for Diffusion Snake
n_points = 50  # number of controllpoints for spline
alph=0.2        # learning rate 
u_iter=12
tau=0.25

# Setup
ds = DiffusionSnake(image_path, v, n_points, alph, respace=True, mode="full", lambd=lambd, u_iterations=u_iter, tau=tau)
u,x,y=ds.draw()


# Plot/ Animate
from matplotlib import animation
fig=plt.figure()
uplt=plt.imshow(u)
Cplt,=plt.plot(x,y,"-b")
print_step = plt.text(.05, .99, "Step: 0", ha='left', va='top')
#cplt,=plt.plot(*ds.C.c.T,"or")          # plot controllpoints

delta_w_neu=0
step=0
def animate(frame):
    print_step.set_text(f"Step: {ds.n_step}")
    for i in range(10):
        ds.step()
    u,x,y=ds.draw()
    uplt.set_array(u.T)#ds.f.T)         # overlay u and f
    Cplt.set_data(x,y)
    #cplt.set_data(*ds.C.spline(np.linspace(0,1,100)).T) # plot controllpoints


    return[uplt,Cplt,print_step]#,cplt]

animate(0)
animate(1)
plt.pause(0.5)
anim = animation.FuncAnimation(fig, animate, interval=1,cache_frame_data=False,blit=True)
plt.show()
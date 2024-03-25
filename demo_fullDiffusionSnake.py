# This script shall contain the final demo for presentation in the end ...
import numpy as np
from src.cicleBspline import Spline
from skimage import io, color
from matplotlib import pyplot as plt
from src.diffusionSnake import DiffusionSnake


# Config
image_path = './sample_images/snail1_gray.png'
n_steps = 1000  # max iterations
eps = 1e-4      # for convergence
lambd, v=5,0.1  # Parameters for Diffusion Snake. labda is not needed for simple mode
n_points = 100  # number of controllpoints for spline
alph=0.7        # learning rate 


# Setup
ds = DiffusionSnake(image_path, lambd, v, n_points, alph, mode="full")
u,x,y=ds.draw()


# Plot/ Animate
from matplotlib import animation
fig=plt.figure()
uplt=plt.imshow(u)
Cplt,=plt.plot(x,y,"-b")
print_step = plt.text(1,5,"Step: 0")
#cplt,=plt.plot(*controllpoints.T,"or")

delta_w_neu=0
step=0


def animate(frame):
    global u,C,step, print_step,controllpoints
    # if step == n_steps: # XXX Maybe use pauseResume.py for this

    print_step.set_text(f"Step: {step}") # = plt.text(1,5,"Step: "+str(step))

    ds.step()
    u,x,y=ds.draw()

    uplt.set_array(ds.f.T)
    Cplt.set_data(x,y)
    #cplt.set_data(*controllpoints.T)

    step += 1
    return[uplt,Cplt,print_step]

animate(0)
animate(1)
plt.pause(0.5)
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()
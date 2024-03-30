import numpy as np
from src.cicleBspline import Spline
from skimage import io, color
from matplotlib import pyplot as plt
from src.diffusionSnake import DiffusionSnake
import time






# Demo Full Diffusion Snake
## Examples 1: Artificial Images: 
lambd, v= 7, 0.1  # blur-factor, factor for spline lengths punishment
n_points = 50     # number of controllpoints for spline
alph=0.2          # learning rate 
u_iter=12         # u-function iterations per snake-step 
tau=0.25          # stepsize for u-terations
init_size = 0.3   # initial spline radius in parts per image-size




# XXX: Bitte jeweils einkommentieren

# Some Converges stable, but a bit shaky with 100x100 px
image_path = './sample_images/artificial/100/rect_1.png'      # approx 500 steps         # XXX show
image_path = './sample_images/artificial/100/snail1.png'      # approx 5000 steps        # XXX show
# image_path = './sample_images/artificial/300/roundish2_300.png'



# Example Parameter Problems: Some do not converge depending a lot on parameters. Cremers had simliar problems Compare Fig 2.4, 2.5
# image_path = './sample_images/artificial/200/snail200.png'
# n_points, u_iter = 100, 1
# init_size = 0.3               # --> not nice
# init_size = 0.4               # --> takes long/ gets stuck
# init_size = 0.1               # --> better? (slow, goes half the way until 7000 steps)
# n_points = 200                # --> now loops






## Examples 2: Real Images:
lambd, v= 7, 0.1  # blur-factor, factor for spline lengths punishment
n_points = 50     # number of controllpoints for spline
alph=0.2          # learning rate 
u_iter=2          # u-function iterations per snake-step 
tau=0.25          # stepsize for u-terations
init_size = 0.3   # initial spline sioze in parts per image-size


image_path = './sample_images/real/100/switch_1.png'          # simple working good   # XXX show
# image_path = './sample_images/real/300/switch_1.png'          # ... but bigger get slower 
# image_path = './sample_images/real/600/switch_1.png'          # ... and slower
# image_path = './sample_images/real/1200/switch_1.png'         # ... and slower





# lambd, v = 12, 0.05 # 0.2 slow, but working
# alpha = 0.9
# n_points = 100
# u_iter=1
# image_path = './sample_images/real/100/hand_1.png'          # somehow working, slow
# image_path = './sample_images/real/300/hand_1.png'          # somehow working, even slower
# image_path = './sample_images/real/100/hand_2.png'          # poorly working, but a hard one, does what it should
# image_path = './sample_images/real/300/hand_2.png'          # poorly working, but a hard one, does what it should






image_path = './sample_images/real/100/bee.png'             # approx 2000 steps       # XXX show
u_iter=12
lambd, v= 15, 0.01  
n_points = 30 
alph=0.5
init_size = 0.2      

# image_path = './sample_images/real/300/bee.png'             # approx 2000 steps
# n_points = 50 
# alph=0.9   








# Setup
ds = DiffusionSnake(image_path, v, n_points, alph, respace=True, mode="full", lambd=lambd, u_iterations=u_iter, tau=tau, init_size=init_size)
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
fplt=plt.imshow(f.T, cmap='gray', vmin=0, vmax=1)
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
import numpy as np
from matplotlib import pyplot as plt
from src.diffusionSnake import DiffusionSnake





# Demo Simple Diffusion Snake
# Examples 1: Artificial images: nice and fast  
# image_path = './sample_images/artificial/100/snail1_gray.png'       # nice and fast on 100 x 100 pixels       
image_path = './sample_images/artificial/100/labyrinth_1.png'       # also good on more complex shape           #  XXX: Show -> not so nice with init radius size/5
image_path = './sample_images/artificial/300/snail300.png'            # working very good on 300 x 300 pixels   #  XXX: Show


# Examples 2: Real Images
# image_path= './sample_images/real/300/bee.png'                        # also very good, a little slow in the beginning (300 x 300)              # XXX: Show
# image_path= './sample_images/real/300/hand_1.png'                     # hard one, still ok, good in the beginning                               # XXX: Show, init: size/5
# image_path= './sample_images/real/300/hand_2.png'                     # hardest one. Finds darkest area in image, but not only the hand
# image_path= './sample_images/real/1200/switch_1.png'                  # very slow on big images (1800 steps here for 1200 x 1200 pixels)
v = 0.01
n_points = 100  # number of controllpoints for spline
alph=0.9        # learning rate 
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

plt.subplot(1,2,2)
fplt=plt.imshow(f.T)
C2plt,=plt.plot(x,y,"-b", label='Spline')
axes[0].set_title('u-function')
axes[1].set_title('Input Image')

a1=axes[0].legend()
a2=axes[1].legend()


def animate(frame):
    print_step.set_text(f"Step: {ds.n_step}")
    
    for i in range(10):# range(10) -> ca 5x speedup
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
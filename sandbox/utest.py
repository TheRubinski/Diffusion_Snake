

import io
import requests
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from bsplineclass import Spline
from math import sqrt
x = requests.get('https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png')

stream = io.BytesIO(x.content)
img = Image.open(stream)

#img.show()
f=np.array(img)

plt.imshow(f, interpolation='nearest')
plt.show()

#print(np.info(f))

spline=Spline([(100,29),(117,100),(15,104),(41,27)])
mask,*_=spline.draws(np.zeros(f.shape,np.uint8))
plt.imshow(mask, interpolation='nearest')
plt.show()

uk=np.copy(f)
w=mask!=1
max_row, max_col = f.shape
t=1/4
l=10
def N(position):
    row, col = position
    neighbors = []
    if row > 0:neighbors.append((row - 1, col))  # Top neighbor
    if row < max_row - 1:neighbors.append((row + 1, col))  # Bottom neighbor
    if col > 0:neighbors.append((row, col - 1))  # Left neighbor
    if col < max_col - 1:neighbors.append((row, col + 1))  # Right neighbor
    return neighbors

fig=plt.figure()
imgplt=plt.imshow(f)



def animate(frame):
    global uk
    ukn=np.zeros(f.shape,np.float64)

    for i in range(max_row):
        for j in range(max_col):
            pos=i,j
            neighbours=N(pos)
            ukn[pos]=((1-t*sum(sqrt(w[pos]*w[npos])for npos in neighbours))*uk[pos]
                    +t*sum(sqrt(w[pos]*w[npos])*uk[npos]for npos in neighbours)
                    )

    ukn+=t/l**2*f
    ukn/=(1+t/l**2)

    uk=ukn
    imgplt.set_array(uk)
    print(frame)
    return [imgplt]
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()


import io
import requests
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from bsplineclass import Spline
from math import sqrt





# original_array = np.array([[1, 2, 3],
#                           [4, 5, 6],
#                           [7, 8, 9]])

# padding_size = 1
# padded_array = np.pad(original_array, pad_width=padding_size, mode='constant', constant_values=0)


# print(padded_array)
# smid=slice(1,-1)
# sup=slice(2,None)
# sdown=slice(None,-2)
# neighbourslices=[(sup,smid),(sdown,smid),(smid,sup),(smid,sdown)]
# for j in neighbourslices:
#     print(padded_array[j])





x = requests.get('https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png')

stream = io.BytesIO(x.content)
img = Image.open(stream)

#img.show()
f=np.array(img)

plt.imshow(f, interpolation='nearest')
plt.show()

#print(np.info(f))

spline=Spline([(100,29),(117,100),(15,104),(41,27)])
mask,*_=spline.draw(np.zeros(f.shape,np.uint8))
plt.imshow(mask, interpolation='nearest')
plt.show()

uk=np.copy(f)
#uk=np.zeros(f.shape,np.uint8)
w=mask!=1
max_row, max_col = f.shape
t=1/4
l=10


fig=plt.figure()
imgplt=plt.imshow(f)

pw = np.pad(w, pad_width=1, constant_values=0)

smid=slice(1,-1)
sup=slice(2,None)
sdown=slice(None,-2)
neighbourslices=[(sup,smid),(sdown,smid),(smid,sup),(smid,sdown)]


def animate(frame):
    global uk
    for i in range(50):
        puk = np.pad(uk, pad_width=1, constant_values=0)#padded uk

        uk=(
            (1-t*sum(np.sqrt(w*pw[j])for j in neighbourslices))*uk
            +t*sum(np.sqrt(w*pw[j])*puk[j]for j in neighbourslices)#u_k=u_(k+1)
            +t/l**2*f
            )/(1+t/l**2)


    imgplt.set_array(uk)
    print(frame*50)
    return [imgplt]
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()


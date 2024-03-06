# just some random snippets 

import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import requests
import io
from PIL import Image
from scipy import interpolate


### load and show images from path
path = "./sample_images"
for f in os.listdir(path):
  im = cv.imread(os.path.join(path, f))
  im = np.asarray(im)
  plt.imshow(im)
  plt.show()



### load and show images from url
x = requests.get('https://cdn.pixabay.com/photo/2012/04/18/00/07/silhouette-of-a-man-36181_960_720.png')
#x = requests.get('https://img.freepik.com/premium-vector/fire-flames-firefighter-silhouette-set-vector-transparent-background_733316-1.jpg?w=1380')  # vlt einzlene
#x = requests.get('https://c8.alamy.com/compfr/ba5r97/silhouette-d-un-arbre-ba5r97.jpg')                                           # nicht gut zum testen
#x = requests.get('https://cdn.vectorstock.com/i/preview-1x/40/10/tree-silhouette-on-white-background-vector-43684010.webp')
#x = requests.get('https://i.stack.imgur.com/IqpIS.png')
#x = requests.get('https://cdn.vectorstock.com/i/preview-1x/40/10/tree-silhouette-on-white-background-vector-43684010.webp')
#print(x.encoding)
#print(x.text)
stream = io.BytesIO(x.content)
img = Image.open(stream)

#img.show()
data=np.array(img)
print(data.shape)
if len(data.shape)==3:
  imagemap=np.sum(data,axis=-1)
imagemap=(imagemap<np.mean(imagemap)).astype(float)
#imagemap should now be a bitmap:)
#print(imagemap)
plt.imshow(imagemap, interpolation='nearest')
plt.show()


### splines
x = [23, 24, 24, 25, 25]
y = [13, 12, 13, 12, 13]

# append the starting x,y coordinates


# fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
# is needed in order to force the spline fit to pass through all the input points.
tck, u = interpolate.splprep([x, y], s=0, per=True)

# evaluate the spline fits for 1000 evenly spaced distance values
xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

# plot the result
fig, ax = plt.subplots(1, 1)
ax.plot(x, y, 'or')
ax.plot(xi, yi, '-b')
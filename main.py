import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt


# just load and show images 
path = "./sample_images"    # all images here are 100 x 100 px
for f in os.listdir(path):
  im = cv.imread(os.path.join(path, f))
  im = np.asarray(im)
  plt.imshow(im)
  plt.show()


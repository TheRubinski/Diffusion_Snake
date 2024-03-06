import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import requests
import io
from PIL import Image
from scipy.interpolate import CubicSpline


def overlay(image, contour):
    return image


# XXX use quadratic B-spline curves (? 2.23, p.24 )


# Load image file

# load and show multiple images from path
# path = "./sample_images"
# for f in os.listdir(path):
#   im = cv.imread(os.path.join(path, f))
#   im = np.asarray(im)


# single image for test-purpose
im = cv.imread(".\sample_images\circle.png")
im = np.asarray(im)
plt.imshow(im)
plt.show()


# generate spline-curve aka C

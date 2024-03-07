# This script shall contain the final demo for presentation in the end ...

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline


def overlay(image, contour):
    return image


def circle_spline(n_nodes=100, degree=2, scale=2):
    """
    degree: of spline, so 2 is quadratic, 3 is qubic
    scale: of cicle in radius
    """
    theta = 2 * np.pi * np.linspace(0, 1, n_nodes)
    y = np.c_[np.cos(theta), np.sin(theta)]         # y are the input datapoints. Here = def cycle
    cs = make_interp_spline(theta, y, k=degree)
    cs.c = scale *cs.c
    return cs 


def get_masks(spline, image):
    # TODO Ruben
    return in_mask, out_mask, conture_mask


# Load image file

## load and show multiple images from path
# path = "./sample_images"
# for f in os.listdir(path):
#   im = cv.imread(os.path.join(path, f))
#   im = np.asarray(im)

# load single image for test-purpose
im = cv.imread(".\sample_images\circle.png")
im = np.asarray(im)
plt.imshow(im)
plt.show()


# generate spline-curve aka C
scale = 5
n_nodes = 10
cs = circle_spline(n_nodes=n_nodes, degree=2, scale=1)       # XXX use quadratic B-spline curves (? 2.23, p.24 )
cs.c = scale * cs.c                                          # XXX You can simply scale the spline by scaling cs.c XXX this is approx the same as scale * y befor generating spline


# TODO Ruben: Move spline to image
# Plot spline
theta = 2 * np.pi * np.linspace(0, 1, n_nodes)
y = np.c_[np.cos(theta), np.sin(theta)]         # y are the input datapoints. Here = def cycle

# Plot
xs = 2 * np.pi * np.linspace(0, 1, 100)         # get (x,y) values from spline curve for plotting
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(y[:, 0], y[:, 1], 'o', label='data')
ax.plot(cs.c[:,0], cs.c[:,1], 'o', label='controlpoints')
ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
ax.axes.set_aspect('equal')
ax.legend(loc='upper right')
plt.show()



# https://docs.scipy.org/doc/scipy/tutorial/interpolate/splines_and_polynomials.html#tutorial-interpolate-ppoly
# use spl.integrate() to integrate spline
# and use spl.derivative() 
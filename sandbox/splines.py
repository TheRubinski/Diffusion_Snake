import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import requests
import io
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline



# # generate Cycle with cubic spline
# theta = 2 * np.pi * np.linspace(0, 1, 5)
# y = np.c_[np.cos(theta), np.sin(theta)]
# cs = CubicSpline(theta, y, bc_type='periodic')
# print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
# # ds/dx=0.0 ds/dy=1.0
# xs = 2 * np.pi * np.linspace(0, 1, 100)
# fig, ax = plt.subplots(figsize=(6.5, 4))
# ax.plot(y[:, 0], y[:, 1], 'o', label='data')
# ax.plot(np.cos(xs), np.sin(xs), label='true')
# ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
# ax.axes.set_aspect('equal')
# ax.legend(loc='center')
# plt.show()


# generate non-cubic B-splines 
# x = np.linspace(0, 3/2, 7)
# y = np.sin(np.pi*x)
# bspl = make_interp_spline(x, y, k=3)    # was k=3

# der = bspl.derivative()      # a BSpline representing the derivative
# xx = np.linspace(0, 3/2, 51)
# plt.plot(xx, bspl(xx), '--', label=r'$\sin(\pi x)$ approx')
# plt.plot(x, y, 'o', label='data')
# plt.plot(xx, der(xx)/np.pi, '--', label='$d \sin(\pi x)/dx / \pi$ approx')
# plt.legend()
# plt.show()



# # quadratic spline cycle
# def cycle_spline(n_nodes=100, degree=2, scale=2):
#     # np.linspace(0, 1, n_nodes)                    # [0.   0.25 0.5  0.75 1.  ]
#     theta = 2 * np.pi * np.linspace(0, 1, n_nodes)  # [0.         1.57079633 3.14159265 4.71238898 6.28318531]
#     y = np.c_[np.cos(theta), np.sin(theta)]
#     y = scale * y
#     print(y)
#     cs = make_interp_spline(theta, y, k=degree)     # removed bc_type='periodic', since it made problems
#     # print(cs.c)
#     #print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
#     # ds/dx=0.0 ds/dy=1.0
#     xs = 2 * np.pi * np.linspace(0, 1, 100)
#     fig, ax = plt.subplots(figsize=(6.5, 4))
#     ax.plot(y[:, 0], y[:, 1], 'o', label='data')
#     ax.plot(np.cos(xs), np.sin(xs), label='true')
#     ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
#     ax.axes.set_aspect('equal')
#     ax.legend(loc='center')
#     plt.show()

# cycle_spline(n_nodes=100, degree=3, scale=100)


# # quadratic spline cycle XXX working, but not positive
# def cycle_spline(n_nodes=100, degree=2, scale=2):
#     theta = 2 * np.pi * np.linspace(0, 1, n_nodes)
#     y = np.c_[np.cos(theta), np.sin(theta)]         # y are the input datapoints. Here = def cycle
#     cs = make_interp_spline(theta, y, k=degree)
#     print(theta)
#     print(cs.t)                                     # cs.t are the knots of the spline, constructed by first and last element of theta and len(theta)
#     print(y)
#     print(cs.c)                                     
#     # cs.c are the coefficients in b-spline basis 
#     # see https://docs.scipy.org/doc/scipy/tutorial/interpolate/splines_and_polynomials.html#design-matrices-in-the-b-spline-basis 

#     cs.c = scale * cs.c      # XXX You can simply scale the spline by scaling cs.c XXX this is approx the same as scale * y befor generating spline

#     # Plot
#     xs = 2 * np.pi * np.linspace(0, 1, 100)         # get (x,y) values from spline curve for plotting
#     fig, ax = plt.subplots(figsize=(6.5, 4))
#     ax.plot(y[:, 0], y[:, 1], 'o', label='data')
#     ax.plot(cs.c[:,0], cs.c[:,1], 'o', label='controlpoints')
#     ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
#     ax.axes.set_aspect('equal')
#     ax.legend(loc='upper right')
#     plt.show()

# cycle_spline(n_nodes=10, degree=2, scale=5)




def circle_spline(n_nodes=100, degree=2, scale=2):
    """
    degree: of spline, so 2 is quadratic, 3 is qubic
    scale: of cicle in radius
    """
    theta = 2 * np.pi * np.linspace(0, 1, n_nodes)
    y = np.c_[np.cos(theta), np.sin(theta)]         # y are the input datapoints. Here = def cycle
    cs = make_interp_spline(theta, y, k=degree)
    print(theta)
    print(cs.t)                                     # cs.t are the knots of the spline, constructed by first and last element of theta and len(theta)
    print(y)
    print(cs.c)                                     
    # cs.c are the coefficients in b-spline basis 
    # see https://docs.scipy.org/doc/scipy/tutorial/interpolate/splines_and_polynomials.html#design-matrices-in-the-b-spline-basis 

    cs.c = scale * cs.c      # XXX You can simply scale the spline by scaling cs.c XXX this is approx the same as scale * y befor generating spline
    # XXX NEXT: find a way to move the circle (simply by moving cs.c ??)
    # XXX AFTER THAT: add circle to image

    # Plot
    xs = 2 * np.pi * np.linspace(0, 1, 100)         # get (x,y) values from spline curve for plotting
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(y[:, 0], y[:, 1], 'o', label='data')
    ax.plot(cs.c[:,0], cs.c[:,1], 'o', label='controlpoints')
    ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
    ax.axes.set_aspect('equal')
    ax.legend(loc='upper right')
    plt.show()

circle_spline(n_nodes=10, degree=2, scale=5)
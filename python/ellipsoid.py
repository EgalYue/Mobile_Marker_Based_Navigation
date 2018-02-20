#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 14.02.18 20:29
@File    : ellipsoid.py
@author: Yue Hu
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import random, cos, sin, sqrt, pi, linspace, deg2rad, meshgrid
from matplotlib import cm

def get_semi_axes_abc(covariance_matrix,cl):
    """
    Get a, b, c of the ellipsoid, they are half the length of the principal axes.
    """
    # Confidence level, we consider only n = 3
    if cl == 0.25:
        z_square = 1.21253
    elif cl == 0.5:
        z_square = 2.36597
    elif cl == 0.75:
        z_square = 4.10834
    elif cl == 0.95:
        z_square = 7.81473
    elif cl == 0.97:
        z_square = 8.94729
    elif cl == 0.99:
        z_square = 11.3449
    else:
        z_square = 0
        print "Error: Wrong confidence level!!!"

    # theta1, theta2, theta3= np.linalg.eig(covariance_matrix) # eigenvalues of covariance matrix
    theta,vec= np.linalg.eig(covariance_matrix) # eigenvalues of covariance matrix
    # print theta
    a = sqrt(theta[0] * z_square)
    b = sqrt(theta[1] * z_square)
    c = sqrt(theta[2] * z_square)
    return a,b,c


def ellipsoid_Volume(a,b,c):
    """
    The volume bounded by the ellipsoid: V = 4/3 * PI * a * b * c
    """
    V = (4.0 / 3.0) * pi * a * b * c
    return V


def drawEllipsoid(a,b,c,xp,yp,zp):
    """
    Here is how you can do it via spherical coordinates
    because a,b,c are half the length of the principal axes.
    xp,yp,zp are the center of ellipsoid
    """
    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')


    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = a * np.outer(np.cos(u), np.sin(v)) + xp
    y = b * np.outer(np.sin(u), np.sin(v)) + yp
    z = c * np.outer(np.ones_like(u), np.cos(v)) + zp

    # Plot:
    # ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')
    # plt.contour(x, y, z,[0])

    # Need to set the forth parameter and the offset value
    ax.contour(x, y, z, [xp], zdir='x', offset=xp,cmap=cm.coolwarm)
    ax.contour(x, y, z, [yp], zdir='y', offset=yp, cmap=cm.coolwarm)
    ax.contour(x, y, z, [zp], zdir='z', offset=zp, cmap=cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(a, b, c)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    plt.show()

def drawAllEllipsoid(a, b, c):
    """
    Here is how you can do it via spherical coordinates
    because a,b,c are []
    """
    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    for i in range(0,len(a)):
        x = a[i] * np.outer(np.cos(u), np.sin(v))
        y = b[i] * np.outer(np.sin(u), np.sin(v))
        z = c[i] * np.outer(np.ones_like(u), np.cos(v))

        # Plot:
        # TODO set color
        if i == 1:
            color = "b"
        elif i == 2:
            color = "r"
        else:
            color = "g"
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color)

        # Adjustment of the axes, so that they all have the same span:
        max_radius = max(a[i], b[i], c[i])
        for axis in 'xyz':
            getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    plt.show()


#=============================Test=========================================

drawEllipsoid(4,9,16,2,5,3)
# drawAllEllipsoid([4,0],[9,9],[0,16])
# print ellipsoid_Volume(3,4,5)
cov = np.array([[2.70877234e-03,4.02577695e-04,1.07777226e-06],[4.02577695e-04, 3.36981672e-04, -1.08499644e-06],[1.07777226e-06, -1.08499644e-06, 2.50353898e-05]])
cov = np.array([[1,0,0],[0,2,0],[0,0,3]])
get_semi_axes_abc(cov,0.95)
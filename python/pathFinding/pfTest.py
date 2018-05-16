#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 28.04.18 08:40
@File    : pfTest.py
@author: Yue Hu
"""
# --------------!!! important!!!----------
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'wx'
#-----------------------------------------
from pylab import *
import matplotlib.pyplot as plt
from mayavi import mlab

x_min = 0
y_min = 0
x_max = 100
y_max = 100
X, Y = mgrid[x_min:x_max + 1:1, y_min: y_max + 1:1]
print X.shape

def gaussian_obstacle(X, Y, x_obs, y_obs, size_robot, size_obstacle):
    # We extend the size of the obstacle with the size of the robot (border expansion)
    ext_size = size_robot + size_obstacle
    sigma_x = (ext_size / 2) / 2.3548
    sigma_y = (ext_size / 2) / 2.3548
    theta = 0
    A = 100  # Weight of the Gaussian
    Z = zeros_like(X)
    a = cos(theta) ** 2 / 2 / sigma_x ** 2 + sin(theta) ** 2 / 2 / sigma_y ** 2
    b = -sin(2 * theta) / 4 / sigma_x ** 2 + sin(2 * theta) / 4 / sigma_y ** 2
    c = sin(theta) ** 2 / 2 / sigma_x ** 2 + cos(theta) ** 2 / 2 / sigma_y ** 2
    Z[:] = Z[:] + A * exp(- (a * (X - x_obs) ** 2 + 2 * b * (X - x_obs) * (Y - y_obs) + c * (Y - y_obs) ** 2))
    return Z


def calculate_vector_field(Z):
    U, V = gradient(Z, 1, 1)
    U = -U
    V = -V
    return U, V


Z = gaussian_obstacle(X, Y, 50, 50, 10, 10)
print "Z",Z
U, V = calculate_vector_field(Z)
print "U",U.shape


def get_velocity_command(U, V, x_robot, y_robot):
    return U[x_robot, y_robot], V[x_robot, y_robot]


Vx_robot, Vy_robot = get_velocity_command(U, V, 40, 40)
print "Vx_robot, Vy_robot",Vx_robot,Vy_robot

# # Plot
# mlab.figure(size=(800, 600))
# mlab.quiver3d(X, Y, zeros_like(X), U, V, zeros_like(U))
# mlab.mesh(X, Y, Z)
# mlab.show()
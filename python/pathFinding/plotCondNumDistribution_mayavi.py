#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 16.05.18 20:48
@File    : plotCondNumDistribution_mayavi.py
@author: Yue Hu
"""
# --------------!!! important!!!----------
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'wx'
#-----------------------------------------
from pylab import *
import matplotlib.pyplot as plt
from mayavi import mlab
import os


# Read accuracy matrix
cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../pathFinding/accuracyMatrix.txt', cur_path)
f = open(new_path, 'r')
l = [map(float, line.split(' ')) for line in f]
accuracy_mat = np.asarray(l)  # convert to matrix : 30 x 60


def calculate_vector_field(Z):
    U, V = gradient(Z, 1, 1)
    U = -U
    V = -V
    return U, V



x_min = 0
y_min = 0
x_max = 60
y_max = 30
X, Y = mgrid[x_min:x_max:1, y_min: y_max:1]
print X.shape
print Y.shape

U, V = calculate_vector_field(accuracy_mat.T)
print U.shape
print V.shape

mlab.figure(size=(800, 600))
mlab.quiver3d(X, Y, zeros_like(X), U, V, zeros_like(U), line_width=3, scale_factor=6)
# mlab.mesh(X, Y, accuracy_mat.T)
mlab.show()

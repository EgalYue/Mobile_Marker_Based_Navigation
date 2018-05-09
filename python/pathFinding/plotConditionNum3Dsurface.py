#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 27.04.18 12:32
@File    : plotConditionNum3Dsurface.py
@author: Yue Hu

Plot the condition number distribution like Mountain
"""

from __future__ import division  # set / as float!!!!
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sys

sys.path.append("..")

import numpy as np
import os  # Read matrix form file

cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../pathFinding/accuracyMatrix.txt', cur_path)
f = open(new_path, 'r')
l = [map(float, line.split(' ')) for line in f] # Read float data
accuracy_mat = np.asarray(l)  # convert to matrix

reso = 0.1
row = accuracy_mat.shape[0]
row_num = int(row*reso)
col = accuracy_mat.shape[1]
col_num = int(col*reso)

Y = np.arange(0, row_num, reso)
X = np.arange(0, col_num, reso)
print "X.shape",X.shape
print "Y.shape",Y.shape
X, Y = np.meshgrid(X, Y)
print "X.shape",X.shape
print "Y.shape",Y.shape
Z = accuracy_mat
print "Z.shape",Z.shape

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('Y: World coordinate')
ax.set_ylabel('X: World coordinate')
ax.set_zlabel('Condition number')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)

plt.show()


"""
    fig = plt.figure()
    ax = Axes3D(fig)
    reso = 0.1
    row = accuracy_mat.shape[0]
    row_num = int(row * reso)
    col = accuracy_mat.shape[1]
    col_num = int(col * reso)
    Y = np.arange(0, row_num, reso)
    X = np.arange(0, col_num, reso)
    X, Y = np.meshgrid(X, Y)
    ax.set_xlabel('Y: World coordinate')
    ax.set_ylabel('X: World coordinate')
    ax.set_zlabel('Total potential')
    ax.plot_surface(X,Y,data.T+accuracy_mat, rstride=1, cstride=1, cmap=cm.viridis)
    plt.show()
"""
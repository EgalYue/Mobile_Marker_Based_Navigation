#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 27.04.18 12:32
@File    : plotTest.py
@author: Yue Hu
"""

from __future__ import division  # set / as float!!!!
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sys

sys.path.append("..")

import Rt_matrix_from_euler_zyx as R_matrix_from_euler_zyx
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t
import numpy as np
import decimal
from scipy.linalg import expm, rq, det, inv
from vision.camera import Camera
from vision.plane import Plane
from solve_pnp import pose_pnp
import cv2
import error_functions as ef

import os  # Read matrix form file
cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../CondNum_TheoryValidation_newAccMat/accuracyMatrix.txt', cur_path)
f = open(new_path, 'r')
l = [map(float, line.split(' ')) for line in f] # Read float data
accuracy_mat = np.asarray(l)  # convert to matrix

row = accuracy_mat.shape[0]
col = accuracy_mat.shape[1]

Y = np.arange(0, row, 1)
X = np.arange(0, col, 1)
print "X.shape",X.shape
print "Y.shape",Y.shape
X, Y = np.meshgrid(X, Y)
Z = accuracy_mat

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('Y')
ax.set_ylabel('Z')
ax.set_zlabel('Condition number')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)

plt.show()
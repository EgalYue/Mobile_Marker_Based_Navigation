#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 20.02.18 17:58
@File    : test.py
@author: Yue Hu
"""

import covariance_mat_sphericalCoordinate as cms
from vision.camera import Camera
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t
from vision.circular_plane import CircularPlane
import numpy as np

def test_f():



    alpha = 0
    belt = 0
    r = 1
    input = [0.0,0.0,0.0]
    input[0] = alpha
    input[1] = belt
    input[2] = r

    cam = Camera()
    cam.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    cam.set_width_heigth(640, 480)
    cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(0), 0))
    cam.set_t(0., 0., 0.5, 'world')
    pl = CircularPlane(origin=np.array([0., 0., 0.]), normal=np.array([0, 0, 1]), radius=0.15, n=4)
    x1 = round(pl.radius * np.cos(np.deg2rad(45)), 3)
    y1 = round(pl.radius * np.sin(np.deg2rad(45)), 3)
    objectPoints_square = np.array(
        [[x1, -x1, -x1, x1],
         [y1, y1, -y1, -y1],
         [0., 0., 0., 0.],
         [1., 1., 1., 1.]])

    new_objectPoints = np.copy(objectPoints_square)

    cms.f(input,cam.K,new_objectPoints)





#======================Test========================
test_f()



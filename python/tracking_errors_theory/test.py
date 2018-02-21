#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 20.02.18 17:58
@File    : test.py
@author: Yue Hu
"""
import sys
sys.path.append("..")
import covariance_mat_sphericalCoordinate as cms
from vision.camera import Camera
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t
from vision.circular_plane import CircularPlane
import numpy as np
import ellipsoid as ellipsoid
import display_cov_mat as dvm


"""
   Test the functions in covariance_mat_sphericalCoordinate.py
"""

def test_f():
    """
    Test f(input, K, obj_point)
    :return:
    """


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


def test_covariance_alpha_belt_r():
    """
    Test covariance_alpha_belt_r(cam, new_objectPoints)
    :return:
    """
    pl = CircularPlane(origin=np.array([0., 0., 0.]), normal=np.array([0, 0, 1]), radius=0.15, n=4)
    x1 = round(pl.radius * np.cos(np.deg2rad(45)), 3)
    y1 = round(pl.radius * np.sin(np.deg2rad(45)), 3)
    objectPoints_square = np.array(
        [[x1, -x1, -x1, x1],
         [y1, y1, -y1, -y1],
         [0., 0., 0., 0.],
         [1., 1., 1., 1.]])

    new_objectPoints = np.copy(objectPoints_square)

    # ------------------------------Z fixed, study X Y-----------------------------------------
    cams_Zfixed = []
    for x in np.linspace(-0.5, 0.5, 5):
        for y in np.linspace(-0.5, 0.5, 5):
            cam1 = Camera()
            cam1.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
            cam1.set_width_heigth(640, 480)
            # TODO  WE have not set R yet
            # print 'cam1.R',cam1.R
            cam1.set_t(x, y, 1.3, frame='world')
            cams_Zfixed.append(cam1)


    xInputs = []
    yInputs = []
    volumes = []
    for cam in cams_Zfixed:
        t = cam.get_world_position()
        xInputs.append(t[0])
        yInputs.append(t[1])

        cov_mat = cms.covariance_alpha_belt_r(cam, new_objectPoints)
        a, b, c = ellipsoid.get_semi_axes_abc(cov_mat, 0.95)
        v = ellipsoid.ellipsoid_Volume(a, b, c)
        volumes.append(v)

    dvm.displayCovVolume_Zfixed3D(xInputs, yInputs, volumes)





#======================Test========================
# test_f()
test_covariance_alpha_belt_r()



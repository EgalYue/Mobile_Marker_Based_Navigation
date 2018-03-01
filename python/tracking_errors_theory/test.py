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
    Z fixed, study X Y-
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
        cam_tem = cam.clone()
        valid = cms.validCam(cam_tem,new_objectPoints)
        if valid:
            t = cam.get_world_position()
            xInputs.append(t[0])
            yInputs.append(t[1])

            cov_mat = cms.covariance_alpha_belt_r(cam, new_objectPoints)
            a, b, c = ellipsoid.get_semi_axes_abc(cov_mat, 0.95)
            v = ellipsoid.ellipsoid_Volume(a, b, c)
            volumes.append(v)

    dvm.displayCovVolume_Zfixed3D(xInputs, yInputs, volumes)


def test2_covariance_alpha_belt_r():
    """
    Test covariance_alpha_belt_r(cam, new_objectPoints)
    X Y fixed, Z changed
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

    # -----------------------------X Y fixed, Z changed-----------------------------
    cams_XYfixed = []
    for i in np.linspace(0.5, 2, 5):
        cam1 = Camera()
        cam1.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
        cam1.set_width_heigth(640, 480)
        cam1.set_R_axisAngle(1.0, 0.0, 0.0, np.deg2rad(180.0))
        cam1.set_t(0, 0, i, frame='world')
        # 0.28075725, -0.23558331, 1.31660688
        cams_XYfixed.append(cam1)

    zInputs = []
    volumes = []
    ellipsoid_paras = np.array([[0], [0], [0], [0], [0], [0]]) # a,b,c,x,y,z
    for cam in cams_XYfixed:
        cam_tem = cam.clone()
        valid = cms.validCam(cam_tem,new_objectPoints)
        if valid:
            t = cam.get_world_position()
            zInputs.append(t[2])
            print "Height",t[2]

            cov_mat = cms.covariance_alpha_belt_r(cam, new_objectPoints)
            a, b, c = ellipsoid.get_semi_axes_abc(cov_mat, 0.95)
            display_array = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],dtype=float)
            display_array[0:3,0] = a, b, c
            display_array[3:6,0] = np.copy(t[0:3])
            ellipsoid_paras = np.hstack((ellipsoid_paras, display_array))

            v = ellipsoid.ellipsoid_Volume(a, b, c)
            print "volumn",v
            volumes.append(v)

    dvm.displayCovVolume_XYfixed3D(zInputs, volumes)
    # print "ellipsoid_paras\n",ellipsoid_paras
    # ellipsoid.drawAllEllipsoid(ellipsoid_paras) # Draw the 3D ellipsoid

def test3_covariance_alpha_belt_r():
    """
    Y fixed, study X Z  like a half circle above the marker
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
    # ------------------------------Y fixed, study X Z  like a half circle -----------------------------------------
    cams_Yfixed = []
    for angle in np.linspace(np.deg2rad(0), np.deg2rad(180), 20):
        x = np.cos(angle)
        z = np.sin(angle)
        cam1 = Camera()
        cam1.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
        cam1.set_width_heigth(640, 480)
        # TODO  WE have not set R yet
        # print 'cam1.R',cam1.R
        cam1.set_t(x, 0, z, frame='world')
        cams_Yfixed.append(cam1)

    xInputs = []
    zInputs = []
    volumes = []
    for cam in cams_Yfixed:
        cam_tem = cam.clone()
        valid = cms.validCam(cam_tem,new_objectPoints)
        if valid:
            t = cam.get_world_position()
            xInputs.append(t[0])
            zInputs.append(t[2])
            print "x = ",t[0]
            print "z = ",t[2]

            cov_mat = cms.covariance_alpha_belt_r(cam, new_objectPoints)
            a, b, c = ellipsoid.get_semi_axes_abc(cov_mat, 0.95)
            v = ellipsoid.ellipsoid_Volume(a, b, c)
            print "v",v
            volumes.append(v)

    dvm.displayCovVolume_Zfixed3D(xInputs, zInputs, volumes)



def test_calculate_camRt_from_alpha_belt_r():
    print cms.calculate_camRt_from_alpha_belt_r(np.deg2rad(45),np.deg2rad(45),1)




def convert_Cartesian_To_Spherical():
    """
    convert Cartesian coordinate system of camera to Spherical coordinate system
    :param cam:
    :return: rad
    """
    # Get the world position of cam
    world_position = [1,1,1]
    x = world_position[0]
    y = world_position[1]
    z = world_position[2]

    r = np.sqrt(x * x + y * y + z * z,dtype=np.float32)
    if r ==0:
        alpha = np.deg2rad(0.0)
    else:
        alpha = np.arccos(y / r, dtype=np.float32)


    if x == 0:
        belt = np.deg2rad(90.0)
    else:
        belt = np.arctan(y / x, dtype=np.float32)
    print "r",r
    print "alpha",np.rad2deg(alpha)
    print "belt",np.rad2deg(belt)


    x = r * np.cos(belt) * np.sin(alpha)
    y = r * np.sin(belt) * np.sin(alpha)
    z = r * np.cos(alpha)
    print "x",x
    print "y",y
    print "z",z

    return alpha,belt,r

#======================Test========================
# test_f()
# test_calculate_camRt_from_alpha_belt_r()
# convert_Cartesian_To_Spherical()
# test_covariance_alpha_belt_r()
test2_covariance_alpha_belt_r()
# test3_covariance_alpha_belt_r()
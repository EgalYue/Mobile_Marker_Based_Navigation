#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 05.05.18 14:13
@File    : compute_condNumOfCamPose.py
@author: Yue Hu
"""
from __future__ import division # set / as float!!!!
import numpy as np
import sys
sys.path.append("..")
from vision.camera import Camera
from vision.plane import Plane
import hT_gradient as gd
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t

# ----------------------------------Initialization-----------------------
plane_size = (0.3, 0.3)
plane = Plane(origin=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), size=plane_size, n=(2, 2))
plane.set_origin(np.array([0, 0, 0]))
plane.uniform()
objectPoints = plane.get_points()
new_objectPoints = np.copy(objectPoints)
# print "new_objectPoints",new_objectPoints
# ------------------------------------------------------------------------
normalized = True


def getCondNum_camPoseInMarker(y,z):
    """
    Compute the condition number of camera position in marker coordinate
    :param y: camera potion in marker coordinate
    :param z: camera potion in marker coordinate
    :return:
    """

    ## CREATE A SIMULATED CAMERA
    cam = Camera()
    cam.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    cam.set_width_heigth(640, 480)

    cam.set_t(-0, -y, -z)
    cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0.0, 0, 0))
    cam.look_at([0, 0, 0])

    radius = np.sqrt(y * y + z * z )
    angle = np.rad2deg(np.arccos(y / radius))
    cam.set_radius(radius)
    cam.set_angle(angle)
    objectPoints = np.copy(new_objectPoints)
    imagePoints = np.array(cam.project(objectPoints, False))

    if ((imagePoints[0, :] < cam.img_width) & (imagePoints[0, :] > 0) & (imagePoints[1, :] < cam.img_height) & (
        imagePoints[1, :] > 0)).all():
        input_list = gd.extract_objectpoints_vars(objectPoints)
        input_list.append(np.array(cam.K))
        input_list.append(np.array(cam.R))
        input_list.append(cam.t[0, 3])
        input_list.append(cam.t[1, 3])
        input_list.append(cam.t[2, 3])
        input_list.append(cam.radius)
        # TODO normalize points!!!
        condNum = gd.matrix_condition_number_autograd(*input_list, normalize = normalized)

    return condNum

#------------------------------ Test -------------------------------
# print getCondNum_camPoseInMarker(0,1.2)

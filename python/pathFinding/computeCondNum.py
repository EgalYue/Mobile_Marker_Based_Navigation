#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 12.05.18 18:01
@File    : computeCondNum.py
@author: Yue Hu
"""
from __future__ import division  # set / as float!!!!
import sys
sys.path.append("..")
import numpy as np
import decimal
from scipy.linalg import expm, rq, det, inv
from vision.camera import Camera
from vision.plane import Plane
import Rt_matrix_from_euler_zyx as R_matrix_from_euler_zyx
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t
from solve_pnp import pose_pnp
import cv2
import error_functions as ef
import CondNumTheoryValidation.hT_gradient as gd

# ----------------------- Basic Infos ---------------------------------------------------
# ----------------------- Marker object points -----------------------------------------
plane_size = (0.3, 0.3)
plane = Plane(origin=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), size=plane_size, n=(2, 2))
plane.set_origin(np.array([0, 0, 0]))
plane.uniform()
objectPoints = plane.get_points()
new_objectPoints = np.copy(objectPoints)
# --------------------------------------------------------------------------------------

def cellCenterPosition(path, grid_reso):
    """
    Get the exact position of each cell in the real world
    Based on Real World Coordinate System [30,60]
    :param pos:
    :param grid_step:
    :return:
    """
    real_path = np.eye(2, 1, dtype=float)
    length = path.shape[1]
    for i in range(length):
        x_str = str(path[0][i] * grid_reso + grid_reso / 2)
        y_str = str(path[1][i] * grid_reso + grid_reso / 2)
        x = float(decimal.Decimal(x_str))
        y = float(decimal.Decimal(y_str))
        xy = np.eye(2, 1, dtype=float)
        xy[0, 0] = x
        xy[1, 0] = y
        real_path = np.hstack((real_path, xy))
    real_path = real_path[:, 1:]
    return real_path

def getCameraPosInWorld(T_WC):
    """
    Get the camear position in the real world cordinate system
    :param T:
    :return:
    """
    t = np.dot(inv(T_WC), np.array([0, 0, 0, 1]))
    cam_x = t[0]
    cam_y = t[1]
    return np.array([cam_x, cam_y]).reshape(2, 1)

def getCameraPosInMarker(T_MC):
    """
    Get the camear position in the real world cordinate system
    YZ
    :param T:
    :return:
    """
    t = np.dot(inv(T_MC), np.array([0, 0, 0, 1]))
    cam_x = t[0]
    cam_y = t[1]
    cam_z = t[2]
    return np.array([cam_y, cam_z]).reshape(2, 1)

def marker_set_t(x, y, z, R, frame='world'):
    """
    Set t of Marker in the world coordinate system
    :param x:
    :param y:
    :param z:
    :param R:
    :param frame:
    :return:
    """
    t = np.eye(4)
    if frame == 'world':
        marker_world = np.array([x, y, z, 1]).T
        marker_t = np.dot(R, -marker_world)
        t[:3, 3] = marker_t[:3]
        return t
    else:
        t[:3, 3] = np.array([x, y, z])
        return t

def getMarkerTransformationMatrix(width, height, grid_reso):
    """
    Assume that we create a [30,60] matrix for the area of marker and the center of marker is between 29 and 30.
    The Real World Coordinate System is at [0,0].
    The positive direction of X_W-axis points to 0->30
    The positive direction of Y_W-axis points to 0->30

    The positive direction of Z_M-axis(Marker) is same as the positive direction of X_W-axis
    The positive direction of Y_M-axis(Marker) is same as the negative direction of Y_W-axis
    :return:
    """
    alpha = np.deg2rad(180.0)
    beta = np.deg2rad(-90.0)
    gamma = np.deg2rad(0.0)
    R = R_matrix_from_euler_zyx.R_matrix_from_euler_zyx(alpha, beta, gamma)
    x = 0.0
    y = width / 2.0 * grid_reso
    z = 0.0
    t = marker_set_t(x, y, z, R, frame='world')
    T = np.dot(t, R)  # Transformation matrix between World and Marker coordinate systems
    return T

def getT_MC_and_Rt_errors(T_WM, pos_world, Rmat_error_loop, tvec_error_loop):
    pos_world_homo = np.array([pos_world[0], pos_world[1], 0, 1])
    pos_marker = np.dot(T_WM, pos_world_homo)
    # print "pos_marker\n", pos_marker

    # Create an initial camera on the center of the world
    cam = Camera()
    f = 800
    cam.set_K(fx=f, fy=f, cx=320, cy=240)  # Camera Matrix
    cam.img_width = 320 * 2
    cam.img_height = 240 * 2
    cam = cam.clone()
    cam.set_t(pos_marker[0], pos_marker[1], pos_marker[2], 'world')
    cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0.0, 0, 0))
    # print "cam.R\n",cam.R
    cam.look_at([0, 0, 0])
    # print "cam.Rt after look at\n",cam.R
    objectPoints = np.copy(new_objectPoints)
    imagePoints = np.array(cam.project(objectPoints, False))
    new_imagePoints = np.copy(imagePoints)
    # -----------------------------------------------------------------
    new_imagePoints_noisy = cam.addnoise_imagePoints(new_imagePoints, mean=0, sd=4)
    # print "new_imagePoints_noisy\n",new_imagePoints_noisy
    debug = False
    # TODO  cv2.SOLVEPNP_DLS, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE
    pnp_tvec, pnp_rmat = pose_pnp(new_objectPoints, new_imagePoints_noisy, cam.K, debug, cv2.SOLVEPNP_ITERATIVE, False)

    # Calculate errors
    Cam_clone_cv2 = cam.clone_withPose(pnp_tvec, pnp_rmat)
    tvec_error, Rmat_error = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, Cam_clone_cv2.get_tvec(), pnp_rmat)
    Rmat_error_loop.append(Rmat_error)
    tvec_error_loop.append(tvec_error)

    # print "cam.get_world_position()\n",cam.get_world_position()
    t = np.eye(4)
    t[:3, 3] = pnp_tvec[:3]
    T_MC = np.dot(t, pnp_rmat)
    return T_MC

def getCondNum_camPoseInRealWord(x_w, y_w, grid_reso, width, height):
    """
    Compute the condition number of camera position in real world coordinate
    :param x_w: camera potion in real world coordinate
    :param y_w: camera potion in real world coordinate
    :param width: gird 60
    :param height: grid 30
    :return:
    """
    # width = int(grid_width/ grid_reso)
    # height = int(grid_height/ grid_reso)
    T_WM = getMarkerTransformationMatrix(width, height, grid_reso)
    pos_world_homo = np.array([x_w, y_w, 0, 1])
    pos_marker = np.dot(T_WM, pos_world_homo)

    ## CREATE A SIMULATED CAMERA
    cam = Camera()
    cam.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    cam.set_width_heigth(640, 480)

    cam.set_t(pos_marker[0], pos_marker[1], pos_marker[2], 'world')
    cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0.0, 0, 0))
    cam.look_at([0, 0, 0])

    radius = np.sqrt(pos_marker[0]**2 + pos_marker[1]**2 + pos_marker[2]**2)
    angle = np.rad2deg(np.arccos(pos_marker[1] / radius))
    cam.set_radius(radius)
    cam.set_angle(angle)
    objectPoints = np.copy(new_objectPoints)
    imagePoints = np.array(cam.project(objectPoints, False))

    condNum = np.inf # undetected region set as 0
    if ((imagePoints[0, :] < cam.img_width) & (imagePoints[0, :] > 0) & (imagePoints[1, :] < cam.img_height) & (
            imagePoints[1, :] > 0)).all():
        input_list = gd.extract_objectpoints_vars(objectPoints)
        input_list.append(np.array(cam.K))
        input_list.append(np.array(cam.R))
        input_list.append(cam.t[0, 3])
        input_list.append(cam.t[1, 3])
        input_list.append(cam.t[2, 3])
        input_list.append(cam.radius)
        # TODO normalize points!!!   set normalize as default True
        condNum = gd.matrix_condition_number_autograd(*input_list, normalize=True)

    return condNum


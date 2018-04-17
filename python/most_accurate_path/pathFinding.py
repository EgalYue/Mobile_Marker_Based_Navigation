#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 16.04.18 12:50
@File    : pathFinding.py
@author: Yue Hu
"""
from __future__ import division # set / as float!!!!
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
import plotPath as plotPath

import os  # Read matrix form file

# -----------------------Basic Infos---------------------------------------------------
homography_iters = 1000 # TODO iterative

# -----------------------marker object points-----------------------------------------
plane_size = (0.3, 0.3)
plane = Plane(origin=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), size=plane_size, n=(2, 2))
plane.set_origin(np.array([0, 0, 0]))
plane.uniform()
objectPoints = plane.get_points()
new_objectPoints = np.copy(objectPoints)
# --------------------------------------------------------------------------

cell_length = 0.1  # The length of each cell is 0.1m. Each cell of matrix is 0.1m x 0.1m.
width = 60
height = 30
cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../CondNum_TheoryValidation_newAccMat/accuracyMatrix.txt', cur_path)
f = open(new_path, 'r')
l = [map(int, line.split(' ')) for line in f]
accuracy_mat = np.asarray(l)  # convert to matrix


# ====================================================================================


def cellCenterPosition(path, grid_step):
    """
    Get the exact position from the center of each cell
    Based on Real World Coordinate System [30,60]
    :param pos:
    :param grid_step:
    :return:
    """
    real_path = np.eye(2, 1, dtype=float)
    length = path.shape[1]
    for i in range(length):
        x_str = str(path[0][i] * grid_step + grid_step / 2)
        y_str = str(path[1][i] * grid_step + grid_step / 2)
        x = float(decimal.Decimal(x_str))
        y = float(decimal.Decimal(y_str))
        xy = np.eye(2, 1, dtype=float)
        xy[0, 0] = x
        xy[1, 0] = y
        real_path = np.hstack((real_path, xy))
    real_path = real_path[:, 1:]
    return real_path


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


def getMarkerTransformationMatrix(width, height, cell_length):
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
    y = width / 2.0 * cell_length
    z = 0.0
    t = marker_set_t(x, y, z, R, frame='world')
    T = np.dot(t, R)  # Transformation matrix between World and Marker coordinate systems
    return T


def moveTo(current_pos, next_pos):
    """
    Compute the distances need to move
    :param current_pos:
    :param next_pos:
    :return:
    """
    current_x = current_pos[0]
    current_y = current_pos[1]
    next_x = next_pos[0]
    next_y = next_pos[1]
    x_step = next_x - current_x
    y_step = next_y - current_y
    return np.array([x_step, y_step]).reshape(2, 1)


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


def getT_MC(T_WM, pos_world):
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

    # TODO Iterative for image points???
    #----------------------------------------------------------------
    #new_imagePoints_noisy = np.zeros((3,4))
    # for j in range(homography_iters):
    #     new_imagePoints_noisy = new_imagePoints_noisy + cam.addnoise_imagePoints(new_imagePoints, mean=0, sd=4)
    # new_imagePoints_noisy = new_imagePoints_noisy / homography_iters
    #-----------------------------------------------------------------
    new_imagePoints_noisy = cam.addnoise_imagePoints(new_imagePoints, mean=0, sd=4)
    # print "new_imagePoints_noisy\n",new_imagePoints_noisy
    debug = False
    # TODO  cv2.SOLVEPNP_DLS, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE
    pnp_tvec, pnp_rmat = pose_pnp(new_objectPoints, new_imagePoints_noisy, cam.K, debug, cv2.SOLVEPNP_ITERATIVE, False)
    # print "cam.get_world_position()\n",cam.get_world_position()
    t = np.eye(4)
    t[:3, 3] = pnp_tvec[:3]
    T_MC = np.dot(t, pnp_rmat)
    # T_MC = cam.Rt # TODO  Real T_MC
    return T_MC


# ===================================================================================
def main():
    # --------------------Test for a simple path----------------------------------------
    #                    A[26,18] -> B[26,22]                                          -
    #                    A - - - B                                                     -
    # -----------------------------------------------------------------------------------
    # accu_path = accuracy_mat[4,16:21]
    # print "-- accu_path --:\n",accu_path

    path = np.array([[26, 26, 26, 26, 26],
                     [18, 19, 20, 21, 22]])
    path = np.array([[4, 4, 4, 4, 4],
                     [16, 17, 18, 19, 20]])
    fix_path = cellCenterPosition(path, cell_length)
    length = fix_path.shape[1]
    T_WM = getMarkerTransformationMatrix(width, height, cell_length)

# ------------------------ Initialization---------------------
    cam_pos_real_current = np.array([2.65, 1.85]).reshape(2, 1)
    cam_pos_real_current = np.array([0.45, 1.65]).reshape(2, 1)
    cam_pos_measured_current = cam_pos_real_current
    move_dis = moveTo(cam_pos_measured_current, fix_path[:, 1])

    # ------------------------- Algorithmus -----------------------------
    # real_path = np.eye(2, 1, dtype=float)
    # real_path[0, 0] = cam_pos_real_current[0, 0]
    # real_path[1, 0] = cam_pos_real_current[1, 0]
    # measured_path = np.eye(2, 1, dtype=float)
    # measured_path[0, 0] = cam_pos_measured_current[0, 0]
    # measured_path[1, 0] = cam_pos_measured_current[1, 0]
    # for i in range(1, length):
    #     T_MC = getT_MC(T_WM, fix_path[:, i])
    #     # camPosInMarker = getCameraPosInMarker(T_MC)
    #     # print "camPosInMarker\n",camPosInMarker
    #     T_WC = np.dot(T_MC, T_WM)
    #     cam_pos_measured_current = getCameraPosInWorld(T_WC)
    #     measured_path = np.hstack((measured_path, cam_pos_measured_current))
    #
    #     # Update camera current real position
    #     cam_pos_real_current = cam_pos_real_current + move_dis  # this move_dis is the previous value
    #     real_path = np.hstack((real_path, cam_pos_real_current))
    #
    #     # Update move_dis
    #     if i == length - 1:
    #         move_dis = np.array([0.0, 0.0]).reshape(2, 1)
    #     else:
    #         move_dis = moveTo(cam_pos_measured_current, fix_path[:, i + 1])
    # print "-- fix_path --:\n", fix_path
    # print "-- measured_path --:\n", measured_path
    # print "-- real_path --:\n", real_path
    # #-----------------------------Plot-----------------------------------------------
    # plotPath.plotPath(fix_path,real_path,measured_path)

    #TODO Iterative for path points???
    fix_path_mean = np.zeros((2,length))
    real_path_mean = np.zeros((2,length))
    measured_path_mean = np.zeros((2,length))
    for j in range(homography_iters):
        # ------------------------ Initialization---------------------
        cam_pos_real_current = np.array([2.65, 1.85]).reshape(2, 1)
        cam_pos_real_current = np.array([0.45, 1.65]).reshape(2, 1)
        cam_pos_measured_current = cam_pos_real_current
        move_dis = moveTo(cam_pos_measured_current, fix_path[:, 1])

        real_path = np.eye(2, 1, dtype=float)
        real_path[0, 0] = cam_pos_real_current[0, 0]
        real_path[1, 0] = cam_pos_real_current[1, 0]
        measured_path = np.eye(2, 1, dtype=float)
        measured_path[0, 0] = cam_pos_measured_current[0, 0]
        measured_path[1, 0] = cam_pos_measured_current[1, 0]

        for i in range(1, length):
            T_MC = getT_MC(T_WM, fix_path[:, i])
            # camPosInMarker = getCameraPosInMarker(T_MC)
            # print "camPosInMarker\n",camPosInMarker
            T_WC = np.dot(T_MC, T_WM)
            cam_pos_measured_current = getCameraPosInWorld(T_WC)
            measured_path = np.hstack((measured_path, cam_pos_measured_current))

            # Update camera current real position
            cam_pos_real_current = cam_pos_real_current + move_dis  # this move_dis is the previous value
            real_path = np.hstack((real_path, cam_pos_real_current))

            # Update move_dis
            if i == length - 1:
                move_dis = np.array([0.0, 0.0]).reshape(2, 1)
            else:
                move_dis = moveTo(cam_pos_measured_current, fix_path[:, i + 1])
        # print "measured_path",measured_path
        measured_path_mean = measured_path_mean +  measured_path
        real_path_mean = real_path_mean + real_path
        fix_path_mean = fix_path_mean + fix_path

    measured_path_mean = measured_path_mean/homography_iters
    real_path_mean = real_path_mean/homography_iters
    fix_path_mean = fix_path_mean/homography_iters
    print "-- fix_path_mean --:\n", fix_path_mean
    print "-- measured_path_mean --:\n", measured_path_mean
    print "-- real_path_mean --:\n", real_path_mean
    #-----------------------------Plot-----------------------------------------------
    plotPath.plotPath(fix_path_mean,real_path_mean,measured_path_mean)


# =================================================================================================

if __name__ == '__main__':
    main()

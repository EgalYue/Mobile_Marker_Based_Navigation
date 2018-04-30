#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 28.04.18 21:16
@File    : comare_TwoPaths.py
@author: Yue Hu

Compare the error(accuracy) of two paths, there two paths maybe one using A* and another using Potential field based on condition number
"""

from __future__ import division  # set / as float!!!!
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
import plotPath as plotPath
import os  # Read matrix form file
import matplotlib.pyplot as plt
import potential_field_planning as pfp
import A_star as Astar
from A_star import Node

# -----------------------Basic Infos---------------------------------------------------
homography_iters = 1000 # TODO iterative for cam pose of each step
error_iters = 10        # TODO iterative for distance error

# -----------------------marker object points-----------------------------------------
plane_size = (0.3, 0.3)
plane = Plane(origin=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), size=plane_size, n=(2, 2))
plane.set_origin(np.array([0, 0, 0]))
plane.uniform()
objectPoints = plane.get_points()
new_objectPoints = np.copy(objectPoints)
# --------------------------------------------------------------------------

cell_length = 0.1  # The length of each cell is 0.1m. Each cell of matrix is 0.1m x 0.1m.
robot_radius = 0.5
width = 60
height = 30
cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../CondNum_TheoryValidation_newAccMat/accuracyMatrix.txt', cur_path)
f = open(new_path, 'r')
l = [map(float, line.split(' ')) for line in f]
accuracy_mat = np.asarray(l)  # convert to matrix


def cellCenterPosition(path, grid_size):
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
        x_str = str(path[0][i] * grid_size + grid_size / 2)
        y_str = str(path[1][i] * grid_size + grid_size / 2)
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

    # TODO Iterative for image points???
    # ----------------------------------------------------------------
    # new_imagePoints_noisy = np.zeros((3,4))
    # for j in range(homography_iters):
    #     new_imagePoints_noisy = new_imagePoints_noisy + cam.addnoise_imagePoints(new_imagePoints, mean=0, sd=4)
    # new_imagePoints_noisy = new_imagePoints_noisy / homography_iters
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



# ===================================================================================
'''
def main():
    #------------------------Fix path manually adding only for test------------------------
    # fix_pathMat_list = []
    # #======================================================
    # p1 = np.array([[11, 10, 9, 8, 8, 9, 10, 11],
    #                [36, 37, 38, 39, 40, 41, 42, 43]])
    #
    # p2 = np.array([[11, 11, 11, 11, 11, 11, 11, 11],
    #                [36, 37, 38, 39, 40, 41, 42, 43]])
    # #---------------------------------------------------------
    # fix_pathMat_list.append(p1)
    # fix_pathMat_list.append(p2)
    #
    # fix_path_list = []
    # for path_mat in fix_pathMat_list:
    #     fix_path_mat = cellCenterPosition(path_mat, cell_length)
    #     fix_path_list.append(fix_path_mat)
    #-------------------------------------------------------------------------------------

    # TODO 30
    #----------------------------- Potential field path planning AND A* path planning -------------------------
    # gird : start = (21,20), goal = (21,30)
    paths_pfp = pfp.potentialField(sx = 2.15, sy = 2.05, gx = 2.15, gy = 3.05, ox = [], oy = [], grid_size = 0.1, robot_radius = 0.5, grid_width = 60, grid_height = 30)
    # print "paths_pfp\n",paths_pfp
    paths_Astar = Astar.aStar(startNode = Node(21,20,None,0,0,0), goalNode = Node(21,30,None,0,0,0), d_diagnoal = 14, d_straight = 10, grid_width = 60, grid_height = 30)
    # print "paths_Astar\n",paths_Astar
    fix_path_list = []
    fix_path_list.append(paths_pfp)
    fix_path_list.append(paths_Astar)
    #---------------------------------------------------------------------------------------------------------

    measured_path_list = []

    Rmat_error_list_allPaths = [] # store the R error for all paths
    tvec_error_list_allPaths = [] # store the t error for all paths

    # TODO 29
    allPaths_pos_list = [] # store the  1000 times pos for all steps for all paths

    for fix_path in fix_path_list:
        print "======================LOOP start one time================================="
        # --------------------Test for a simple path----------------------------------------
        #                    A[26,18] -> B[26,22]                                          -
        #                    A - - - B                                                     -
        # -----------------------------------------------------------------------------------
        # accu_path = accuracy_mat[4,16:21]
        # print "-- accu_path --:\n",accu_path

        path_steps = fix_path.shape[1] # The total step of  one path
        T_WM = getMarkerTransformationMatrix(width, height, cell_length)

        # ------------------------ Initialization---------------------
        measured_path = np.zeros((2, 1), dtype=float)


        Rmat_error_list = [] # store the R error for current only one path
        tvec_error_list = [] # store the t error for current only one path

        #TODO 29
        allPos_list = [] # store all the positions of each step ,each step is computed 1000 times

        for i in range(0, path_steps):
            # homography_iters
            cam_pos_measured_current_sum = np.zeros((2, 1), dtype=float)
            # The R errors and t errors
            Rmat_error_loop = []
            tvec_error_loop = []

            # TODO 29
            currentPos = np.zeros((2, 1), dtype=float)  # 2x1000 store current position, is computed 1000 times

            # For each step(each cam position need to compute iterative, obtain mean value)
            for j in range(homography_iters):
                fix_currrentStep = fix_path[:, i] # current step point
                T_MC = getT_MC_and_Rt_errors(T_WM, fix_currrentStep, Rmat_error_loop, tvec_error_loop)
                T_WC = np.dot(T_MC, T_WM)
                cam_pos_measured_current = getCameraPosInWorld(T_WC)
                cam_pos_measured_current_sum = cam_pos_measured_current_sum + cam_pos_measured_current

                # TODO 29
                currentPos = np.hstack((currentPos,cam_pos_measured_current))

            cam_pos_measured_current_mean = cam_pos_measured_current_sum / homography_iters
            cam_pos_measured_current = np.copy(cam_pos_measured_current_mean)
            measured_path = np.hstack((measured_path, cam_pos_measured_current))
            # The R errors and t errors
            Rmat_error_list.append(np.mean(Rmat_error_loop))
            tvec_error_list.append(np.mean(tvec_error_loop))

            # TODO 29
            currentPos = currentPos[:,1:]
            allPos_list.append(currentPos)

        # Because of np.hstack, remove the first column
        measured_path = measured_path[:,1:]
        print "-- fix_path_mean --:\n", fix_path
        print "-- measured_path_mean --:\n", measured_path
        print "======================LOOP end one time================================="
        measured_path_list.append(measured_path)

        Rmat_error_list_allPaths.append(Rmat_error_list)
        tvec_error_list_allPaths.append(tvec_error_list)

        #TODO 29
        allPaths_pos_list.append(allPos_list)

    # ---------------------------- Plot-----------------------------------------------

    plotPath.plotAllPaths(fix_path_list, measured_path_list, Rmat_error_list_allPaths, tvec_error_list_allPaths)
    # plotPath.comparePaths_Gaussian(fix_path_list, measured_path_list)
    plotPath.plotScatterEachStep(allPaths_pos_list)
    # ===================================== End main() ===============================================
'''

def compute_measured_data(fix_path):
    # --------------------Test for a simple path----------------------------------------
    #                    A[26,18] -> B[26,22]                                          -
    #                    A - - - B                                                     -
    # -----------------------------------------------------------------------------------
    # accu_path = accuracy_mat[4,16:21]
    # print "-- accu_path --:\n",accu_path

    path_steps = fix_path.shape[1]  # The total step of  one path
    T_WM = getMarkerTransformationMatrix(width, height, cell_length)

    # ------------------------ Initialization---------------------
    measured_path = np.zeros((2, 1), dtype=float)

    Rmat_error_list = []  # store the R error for current only one path
    tvec_error_list = []  # store the t error for current only one path

    # TODO 29
    allPos_list = []  # store all the positions of each step ,each step is computed 1000 times
    # TODO 30
    disError = []

    for i in range(0, path_steps):
        # homography_iters
        cam_pos_measured_current_sum = np.zeros((2, 1), dtype=float)
        # The R errors and t errors
        Rmat_error_loop = []
        tvec_error_loop = []

        # TODO 29
        currentPos = np.zeros((2, 1), dtype=float)  # 2x1000 store current position, is computed 1000 times

        # For each step(each cam position need to compute iterative, obtain mean value)
        for j in range(homography_iters):
            fix_currrentStep = fix_path[:, i]  # current step point
            T_MC = getT_MC_and_Rt_errors(T_WM, fix_currrentStep, Rmat_error_loop, tvec_error_loop)
            T_WC = np.dot(T_MC, T_WM)
            cam_pos_measured_current = getCameraPosInWorld(T_WC)
            cam_pos_measured_current_sum = cam_pos_measured_current_sum + cam_pos_measured_current

            # TODO 29
            currentPos = np.hstack((currentPos, cam_pos_measured_current))

        cam_pos_measured_current_mean = cam_pos_measured_current_sum / homography_iters
        cam_pos_measured_current = np.copy(cam_pos_measured_current_mean)
        measured_path = np.hstack((measured_path, cam_pos_measured_current))
        # The R errors and t errors
        Rmat_error_list.append(np.mean(Rmat_error_loop))
        tvec_error_list.append(np.mean(tvec_error_loop))

        # TODO 29
        currentPos = currentPos[:, 1:]
        allPos_list.append(currentPos)

    # Because of np.hstack, remove the first column
    measured_path = measured_path[:, 1:]
    tem_measured = np.square(fix_path - measured_path)
    disError = np.sqrt(tem_measured[0, :] + tem_measured[1, :])
    # print "-- fix_path_mean --:\n", fix_path
    # print "-- measured_path_mean --:\n", measured_path
    return  measured_path, allPos_list, disError

def computeDistanceErrorMeanStd(fix_path):
    stepLength = fix_path.shape[1]
    disErrorList = np.zeros((1, stepLength))

    measured_path_list = np.zeros((1, stepLength))
    for i in range(error_iters):
        # TODO  allPos_list
        measured_path, allPos_list, disError = compute_measured_data(fix_path)

        disErrorList = np.vstack((disErrorList, disError))
        measured_path_list = np.vstack((measured_path_list, measured_path))
    disErrorList = disErrorList[:,1:]
    disErrorMean = np.mean(disErrorList,axis = 0)
    disErrorStd = np.std(disErrorList,axis = 0)

    measured_path_list = measured_path_list[1:,:]
    measured_pathX_list = measured_path_list[0::2,:]
    measured_pathX_mean = np.mean(measured_pathX_list,axis = 0)
    measured_pathY_list = measured_path_list[1::2, :]
    measured_pathY_mean = np.mean(measured_pathY_list,axis = 0)

    measured_path = np.vstack((measured_pathX_mean,measured_pathY_mean))

    return measured_path, allPos_list, disErrorMean, disErrorStd

def main():
    #----------------------------- Potential field path planning AND A* path planning -------------------------
    # gird : start = (21,20), goal = (21,30)
    paths_pfp = pfp.potentialField(sx = 2.15, sy = 2.05, gx = 2.15, gy = 3.05, ox = [], oy = [], grid_size = cell_length, robot_radius = robot_radius, grid_width = width, grid_height = height)
    paths_Astar = Astar.aStar(startNode = Node(21,20,None,0,0,0), goalNode = Node(21,30,None,0,0,0), d_diagnoal = 14, d_straight = 10, grid_width = width, grid_height = height)
    fix_path_list = [] # first is pfp, second is A*
    fix_path_list.append(paths_pfp)
    fix_path_list.append(paths_Astar)
    #---------------------------------------------------------------------------------------------------------
    measured_path_list = []
    # TODO 29
    allPaths_pos_list = [] # store the  1000 times pos for all steps for all paths
    # TODO 30
    disErrorMean_list = []
    disErrorStd_list = []

    for fix_path in fix_path_list:
        print "======================LOOP start one time================================="
        measured_path, allPos_list, disErrorMean, disErrorStd = computeDistanceErrorMeanStd(fix_path)
        print "measured_path\n",measured_path
        print "disErrorStd\n",disErrorStd
        print "disErrorMean\n",disErrorMean
        measured_path_list.append(measured_path)
        # TODO 29
        allPaths_pos_list.append(allPos_list)
        # TODO 30
        disErrorMean_list.append(disErrorMean)
        disErrorStd_list.append(disErrorStd)

        print "======================LOOP end one time================================="

    # ---------------------------- Plot-----------------------------------------------
    plotPath.plotComparePaths(fix_path_list, disErrorMean_list, disErrorStd_list)
    # plotPath.plotScatterEachStep(allPaths_pos_list)
    # ===================================== End main() ===============================================




# Following is: compute the path 1000 times and get the mean value, this idea is not correct! we should compute 1000 times for
#               each points first! and then add it to the whole path
# def main():
#     fix_pathMat_list = []
#     # ======================================================
#     p1 = np.array([[11, 10, 9, 8, 8, 9, 10, 11],
#                    [36, 37, 38, 39, 40, 41, 42, 43]])
#
#     # p2 = np.array([[11, 11, 11, 11, 11, 11, 11, 11],
#     #                [36, 37, 38, 39, 40, 41, 42, 43]])
#     # ---------------------------------------------------------
#     fix_pathMat_list.append(p1)
#     # fix_pathMat_list.append(p2)
#
#     fix_path_list = []
#     for path_mat in fix_pathMat_list:
#         fix_path_mat = cellCenterPosition(path_mat, cell_length)
#         fix_path_list.append(fix_path_mat)
#
#     measured_path_list = []
#
#     Rmat_error_list_allPaths = []  # store the R error for all paths
#     tvec_error_list_allPaths = []  # store the t error for all paths
#     for fix_path in fix_path_list:
#         # --------------------Test for a simple path----------------------------------------
#         #                    A[26,18] -> B[26,22]                                          -
#         #                    A - - - B                                                     -
#         # -----------------------------------------------------------------------------------
#         # accu_path = accuracy_mat[4,16:21]
#         # print "-- accu_path --:\n",accu_path
#
#         path_steps = fix_path.shape[1]  # The total step of  one path
#         T_WM = getMarkerTransformationMatrix(width, height, cell_length)
#
#         measured_path_iters_list = [] # store all the computes, homography_iters = 1000
#         # compute times
#         for j in range(homography_iters):
#
#             measured_path = np.zeros((2, 1), dtype=float)
#             for i in range(0, path_steps):
#
#                 Rmat_error_loop = []
#                 tvec_error_loop = []
#                 T_MC = getT_MC_and_Rt_errors(T_WM, fix_path[:, i], Rmat_error_loop, tvec_error_loop)
#                 T_WC = np.dot(T_MC, T_WM)
#                 cam_pos_measured_current = getCameraPosInWorld(T_WC)
#                 measured_path = np.hstack((measured_path, cam_pos_measured_current))
#
#             measured_path = measured_path[:, 1:]
#             measured_path_iters_list.append(measured_path)
#             print "measured_path\n",measured_path
#             # print "measured_path_iters_list\n",measured_path_iters_list
#
#         plotPath.plotPositionError_FillBetween(fix_path,measured_path_iters_list)
#
#
#     print "=====================End========================="






# =============================== Main Entry ==================================================================

if __name__ == '__main__':
    main()
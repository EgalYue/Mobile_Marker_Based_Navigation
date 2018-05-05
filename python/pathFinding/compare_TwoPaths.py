#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 28.04.18 21:16
@File    : compare_TwoPaths.py
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
import os  # Read matrix form file
import matplotlib.pyplot as plt
import potential_field_planning as pfp
import A_star as Astar
import plotPath as plotPath
import plotPath_mayavi as plotPath_mayavi
import CondNumTheoryValidation.hT_gradient as gd


# ----------------------- Basic Infos ---------------------------------------------------
homography_iters = 1000 # TODO iterative for cam pose of each step
error_iters = 10       # TODO iterative for distance error
normalized = True


grid_reso = 0.1  # The length of each cell is 0.1m. Each cell of matrix is 0.1m x 0.1m.
robot_radius = 0.5 # [m]
grid_width = 6 # default 6[m]
grid_height = 3 # default 3[m]
cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../pathFinding/accuracyMatrix.txt', cur_path)
f = open(new_path, 'r')
l = [map(float, line.split(' ')) for line in f]
accuracy_mat = np.asarray(l)  # convert to matrix

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


def getCondNum_camPoseInRealWord(x_w, y_w):
    """
    Compute the condition number of camera position in real world coordinate
    :param x_w: camera potion in real world coordinate
    :param y_w: camera potion in real world coordinate
    :return:
    """
    width = int(grid_width/ grid_reso)
    height = int(grid_height/ grid_reso)
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
        condNum = gd.matrix_condition_number_autograd(*input_list, normalize=normalized)

    return condNum
# ===================================================================================

def compute_measured_data(fix_path):
    # --------------------Test for a simple path----------------------------------------
    #                    A[26,18] -> B[26,22]                                          -
    #                    A - - - B                                                     -
    # -----------------------------------------------------------------------------------
    # accu_path = accuracy_mat[4,16:21]
    # print "-- accu_path --:\n",accu_path

    path_steps = fix_path.shape[1]  # The total step of  one path
    width = int(grid_width/ grid_reso)
    height = int(grid_height/ grid_reso)
    T_WM = getMarkerTransformationMatrix(width, height, grid_reso)

    # ------------------------ Initialization---------------------
    measured_path = np.zeros((2, 1), dtype=float)

    Rmat_error_list = []  # store the R error for current only one path
    tvec_error_list = []  # store the t error for current only one path

    # TODO 29
    # allPos_list = []  # store all the positions of each step ,each step is computed 1000 times

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
        # currentPos = currentPos[:, 1:]
        # allPos_list.append(currentPos)

    # Because of np.hstack, remove the first column
    measured_path = measured_path[:, 1:]
    tem_measured = np.square(fix_path - measured_path)
    disError = np.sqrt(tem_measured[0, :] + tem_measured[1, :])
    # print "-- fix_path_mean --:\n", fix_path
    # print "-- measured_path_mean --:\n", measured_path
    # return  measured_path, allPos_list, disError
    # TODO 2
    xyError = np.vstack((measured_path,disError)) # 3 x n path_steps : measured path + distance error
    return  measured_path, disError, xyError, Rmat_error_list, tvec_error_list

def computeDistanceErrorMeanStd(fix_path):
    stepLength = fix_path.shape[1]
    disErrorList = np.zeros((1, stepLength))
    measured_path_list = np.zeros((1, stepLength))
    # TODO 2
    xyError_list = np.zeros((1, stepLength))
    Rmat_error_list_mean_iters = np.zeros((1, stepLength))
    tvec_error_list_mean_iters = np.zeros((1, stepLength))

    for i in range(error_iters):
        # TODO  allPos_list
        # measured_path, allPos_list, disError = compute_measured_data(fix_path)
        measured_path, disError, xyError, Rmat_error_list, tvec_error_list = compute_measured_data(fix_path)
        disErrorList = np.vstack((disErrorList, disError))
        measured_path_list = np.vstack((measured_path_list, measured_path))
        # TODO 2
        xyError_list = np.vstack((xyError_list, xyError))
        Rmat_error_list_mean_iters = np.vstack((Rmat_error_list_mean_iters,Rmat_error_list))
        tvec_error_list_mean_iters = np.vstack((tvec_error_list_mean_iters,tvec_error_list))

    disErrorList = disErrorList[1:,:]
    disErrorMean = np.mean(disErrorList,axis = 0)
    disErrorStd = np.std(disErrorList,axis = 0)
    # TODO 2
    xyError_list = xyError_list[1:,:] # (3*error_iters) x path_steps :  store x,y,distance error of error_iters times
    Rmat_error_list_mean_iters = Rmat_error_list_mean_iters[1:,:]
    Rmat_error_list_mean = np.mean(Rmat_error_list_mean_iters,axis = 0)
    Rmat_error_list_std = np.std(Rmat_error_list_mean_iters,axis = 0)

    tvec_error_list_mean_iters = tvec_error_list_mean_iters[1:,:]
    tvec_error_list_mean = np.mean(tvec_error_list_mean_iters,axis = 0)
    tvec_error_list_std = np.std(tvec_error_list_mean_iters,axis = 0)

    measured_path_list = measured_path_list[1:,:]
    measured_pathX_list = measured_path_list[0::2,:]
    measured_pathX_mean = np.mean(measured_pathX_list,axis = 0)
    measured_pathY_list = measured_path_list[1::2, :]
    measured_pathY_mean = np.mean(measured_pathY_list,axis = 0)

    measured_path = np.vstack((measured_pathX_mean,measured_pathY_mean)) # mean measured path of error_iters times

    # return measured_path, allPos_list, disErrorMean, disErrorStd
    return measured_path, disErrorMean, disErrorStd, xyError_list, Rmat_error_list_mean, tvec_error_list_mean, Rmat_error_list_std, tvec_error_list_std

# def gridPosToRealPos(ix, iy, grid_reso = 0.1):
#     x_real = ix * grid_reso + grid_reso/2
#     y_real = iy * grid_reso + grid_reso/2
#     return x_real,y_real
#
# def realPosTogridPos(x_real, y_real, grid_reso = 0.1):
#     ix = int(round((x_real - grid_reso/2) /grid_reso))
#     iy = int(round((y_real - grid_reso/2) /grid_reso))
#     return ix,iy

def main():
    #----------------------------- Potential field path planning AND A* path planning -------------------------
    # gird : start = (21,20), goal = (21,30)
    # convert position in grid to real    2.15 2.05 2.15 3.05
    # TODO Set the position!!!
    sx_real = 1.55
    sy_real = 2.05
    gx_real = 1.55
    gy_real = 4.05

    paths_pfp = pfp.potentialField(sx = sx_real, sy = sy_real, gx = gx_real, gy = gy_real, ox = [], oy = [], grid_reso = grid_reso, robot_radius = robot_radius, grid_width = grid_width, grid_height = grid_height)
    paths_Astar = Astar.aStar(sx = sx_real, sy = sy_real, gx = gx_real, gy = gy_real, d_diagnoal = 14, d_straight = 10, grid_reso = grid_reso, grid_width = grid_width, grid_height = grid_height)

    fix_path_list = [] # first is pfp, second is A*
    fix_path_list.append(paths_pfp)
    fix_path_list.append(paths_Astar)
    #---------------------------------------------------------------------------------------------------------
    measured_path_list = []
    # TODO 29
    # allPaths_pos_list = [] # store the  1000 times pos for all steps for all paths
    # TODO 30
    disErrorMean_list = []
    disErrorStd_list = []
    # TODO 2
    xyError_list_AllPaths = []  # store xyError_list for all paths
    Rmat_error_mean_list_AllPaths = []
    tvec_error_mean_list_AllPaths = []

    Rmat_error_std_list_AllPaths = []
    tvec_error_std_list_AllPaths = []
    for fix_path in fix_path_list:
        print "======================LOOP start one time================================="
        # measured_path, allPos_list, disErrorMean, disErrorStd = computeDistanceErrorMeanStd(fix_path)
        measured_path, disErrorMean, disErrorStd, xyError_list,  Rmat_error_list_mean_iters, \
        tvec_error_list_mean_iters, Rmat_error_list_std_iters, tvec_error_list_std_iters = computeDistanceErrorMeanStd(fix_path)
        print "fix_path\n",fix_path
        print "measured_path\n",measured_path
        print "disErrorStd\n",disErrorStd
        print "disErrorMean\n",disErrorMean
        measured_path_list.append(measured_path)
        # TODO 29
        # allPaths_pos_list.append(allPos_list)
        # TODO 30
        disErrorMean_list.append(disErrorMean)
        disErrorStd_list.append(disErrorStd)

        # TODO 2
        xyError_list_AllPaths.append(xyError_list)
        Rmat_error_mean_list_AllPaths.append(Rmat_error_list_mean_iters)
        Rmat_error_std_list_AllPaths.append(Rmat_error_list_std_iters)
        tvec_error_mean_list_AllPaths.append(tvec_error_list_mean_iters)
        tvec_error_std_list_AllPaths.append(tvec_error_list_std_iters)
        print "======================LOOP end one time================================="

    # ---------------------------- Plot-----------------------------------------------
    # plotPath.plotComparePaths(fix_path_list, disErrorMean_list, disErrorStd_list, Rmat_error_mean_list_AllPaths, tvec_error_mean_list_AllPaths, Rmat_error_std_list_AllPaths, tvec_error_std_list_AllPaths)

    # plotPath.plotFixedMeasuredFillBetween(fix_path_list, disErrorMean_list)
    # plotPath.plotComparePaths_DisError_3DSurface(xyError_list_AllPaths, grid_reso = grid_reso)
    # plotPath.plotComparePaths_R_error_3DSurface(fix_path_list, Rmat_error_mean_list_AllPaths, grid_reso = grid_reso, width = grid_width, height = grid_height)
    # plotPath.plotComparePaths_t_error_3DSurface(fix_path_list, tvec_error_mean_list_AllPaths, grid_reso = grid_reso, width = grid_width, height = grid_height)
    # ---------------------------- Plot with Mayavi ------------------------------------
    plotPath_mayavi.plotComparePaths_DisError_3DSurface(xyError_list_AllPaths, grid_reso = grid_reso)
    plotPath_mayavi.plotComparePaths_R_error_3DSurface(fix_path_list, Rmat_error_mean_list_AllPaths, grid_reso = grid_reso, width = grid_width, height = grid_height)
    plotPath_mayavi.plotComparePaths_t_error_3DSurface(fix_path_list, tvec_error_mean_list_AllPaths, grid_reso = grid_reso, width = grid_width, height = grid_height)

    # ===================================== End main() ===============================================

# =============================== Main Entry ==================================================================

if __name__ == '__main__':
    main()
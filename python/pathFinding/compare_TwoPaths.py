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

import numpy as np
import cv2 # Dont delete this!!!
import potential_field_planning as pfp
import A_star as Astar
import A_star_modified as Astar_modified
import plotPath as plotPath
import plotPath_mayavi as plotPath_mayavi
import computeCondNum as ccn

# ----------------------- Basic Infos ---------------------------------------------------
homography_iters = 1000 # TODO iterative for cam pose of each step
error_iters = 100       # TODO iterative for distance error

grid_reso = 0.1  # The length of each cell is 0.1m. Each cell of matrix is 0.1m x 0.1m.
robot_radius = 0.5 # [m]
grid_width = 6 # default 6[m]
grid_height = 3 # default 3[m]
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
    T_WM = ccn.getMarkerTransformationMatrix(width, height, grid_reso)

    # ------------------------ Initialization---------------------
    measured_path = np.zeros((2, 1), dtype=float)

    Rmat_error_list = []  # store the R error for current only one path
    tvec_error_list = []  # store the t error for current only one path

    # TODO 29
    # allPos_list = []  # store all the positions of each step ,each step is computed 1000 times

    for i in range(0, path_steps):
        cam_pos_measured_current_sum = np.zeros((2, 1), dtype=float)
        # The R errors and t errors
        Rmat_error_loop = []
        tvec_error_loop = []

        # TODO 29
        currentPos = np.zeros((2, 1), dtype=float)  # 2x1000 store current position, is computed 1000 times

        # For each step(each cam position need to compute iterative, obtain mean value)
        for j in range(homography_iters):
            fix_currrentStep = fix_path[:, i]  # current step point
            T_MC = ccn.getT_MC_and_Rt_errors(T_WM, fix_currrentStep, Rmat_error_loop, tvec_error_loop)
            T_WC = np.dot(T_MC, T_WM)
            cam_pos_measured_current = ccn.getCameraPosInWorld(T_WC)
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
    # paths_Astar_modified = Astar_modified.aStar(sx = sx_real, sy = sy_real, gx = gx_real, gy = gy_real, d_diagnoal = 14, d_straight = 10, grid_reso = grid_reso, grid_width = grid_width, grid_height = grid_height)


    fix_path_list = [] # first is pfp, second is A*
    fix_path_list.append(paths_pfp)
    fix_path_list.append(paths_Astar)
    # fix_path_list.append(paths_Astar_modified)
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
    plotPath.plotComparePaths(fix_path_list, disErrorMean_list, disErrorStd_list, Rmat_error_mean_list_AllPaths, tvec_error_mean_list_AllPaths, Rmat_error_std_list_AllPaths, tvec_error_std_list_AllPaths)
    plotPath.plotFixedMeasuredFillBetween(fix_path_list, disErrorMean_list)
    # ---------------------------- Plot with Mayavi ------------------------------------
    plotPath_mayavi.plotComparePaths_DisError_3DSurface(xyError_list_AllPaths, grid_reso = grid_reso)
    plotPath_mayavi.plotComparePaths_R_error_3DSurface(fix_path_list, Rmat_error_mean_list_AllPaths, grid_reso = grid_reso, width = grid_width, height = grid_height)
    plotPath_mayavi.plotComparePaths_t_error_3DSurface(fix_path_list, tvec_error_mean_list_AllPaths, grid_reso = grid_reso, width = grid_width, height = grid_height)

    # ===================================== End main() ===============================================

# =============================== Main Entry ==================================================================

if __name__ == '__main__':
    main()
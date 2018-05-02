#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 19.12.17 12:45
@File    : brute_force.py
@author: Yue Hu
"""
from __future__ import division # set / as float!!!!
import numpy as np
import pickle
import sys
sys.path.append("..")
from vision.camera import Camera
from vision.plane import Plane
import matplotlib.pyplot as plt
import scipy.io as sio
import error_functions as ef
from ippe import homo2d
from homographyHarker.homographyHarker import homographyHarker as hh
import hT_gradient as gd
from solve_ippe import pose_ippe_both
from solve_pnp import pose_pnp
import cv2
import math
import decimal


calc_metrics = False
number_of_points = 4
## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
cam.set_width_heigth(640, 480)
## CREATE A SET OF IMAGE POINTS FOR VALIDATION OF THE HOMOGRAPHY ESTIMATION
# This will create a grid of 16 points of size = (0.3,0.3) meters
validation_plane = Plane(origin=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), size=(0.3, 0.3), n=(4, 4))
validation_plane.uniform()
# ---------This plane should be same to the plane in cam distribution!!!!-----------------------
plane_size = (0.3, 0.3)
plane = Plane(origin=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), size=plane_size, n=(2, 2))
plane.set_origin(np.array([0, 0, 0]))
plane.uniform()
objectPoints = plane.get_points()
new_objectPoints = np.copy(objectPoints)
# print "new_objectPoints",new_objectPoints
# -------------------------------------------------------------------------------------------------
normalize = False
homography_iters = 1     # TODO homography_iters changed

def heightGetCondNum(cams, accuracy_mat, theta_params, r_params):

    accuracy_mat_new = np.copy(accuracy_mat)
    accuracy_mat_new_rowNum = accuracy_mat_new.shape[0]
    accuracy_mat_new_colNum = accuracy_mat_new.shape[1]
    num_mat = np.copy(accuracy_mat) # each cell stores how many condition numbers in corresponding cell of accuracy_mat_new

    # ------------------------For all cameras------------------------------
    mat_cond_list = []
    imagePoints_des = []
    cam_valid = []
    display_mat = np.array([[0.0], [0.0], [0.0], [0.0]])
    transfer_error_list = []
    ippe_tvec_error_list1 = []
    ippe_rmat_error_list1 = []
    ippe_tvec_error_list2 = []
    ippe_rmat_error_list2 = []
    pnp_tvec_error_list = []
    pnp_rmat_error_list = []
    # ------------------------------------------------------------------
    for cam in cams:
        objectPoints = np.copy(new_objectPoints)
        imagePoints = np.array(cam.project(objectPoints, False))
        if ((imagePoints[0, :] < cam.img_width) & (imagePoints[0, :] > 0) & (imagePoints[1, :] < cam.img_height) & (
            imagePoints[1, :] > 0)).all():
            # calculate:  cond num at this camera position belongs to which cell of accuracy matrix
            cam_r = np.copy(cam.radius)
            cam_angle = np.copy(cam.angle)

            accuracy_mat_row_list,accuracy_mat_col_list = camPositionInRealMatGrid(cam_r, cam_angle, accuracy_mat_new_rowNum, accuracy_mat_new_colNum)
            #----------------------------Calculate cond num-----------------------------------
            imagePoints_des.append(np.array(cam.project(objectPoints, False)))
            input_list = gd.extract_objectpoints_vars(objectPoints)
            input_list.append(np.array(cam.K))
            input_list.append(np.array(cam.R))
            input_list.append(cam.t[0, 3])
            input_list.append(cam.t[1, 3])
            input_list.append(cam.t[2, 3])
            input_list.append(cam.radius)
            # TODO normalize points!!!
            mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=True)
            mat_cond_list.append(mat_cond)

            updateAccuracySumMat(accuracy_mat_row_list, accuracy_mat_col_list, num_mat, accuracy_mat_new, mat_cond)
            # ------------------------Calculate Error------------------------
            computeError(cam,imagePoints, transfer_error_list, ippe_tvec_error_list1, ippe_rmat_error_list1,
                             ippe_tvec_error_list2, ippe_rmat_error_list2, pnp_tvec_error_list, pnp_rmat_error_list)
            # ----------------------------------------------------------------------------------------
            cam_valid.append(cam)
            cam_position = cam.get_world_position()
            display_array = np.eye(4, 1, dtype=float)
            display_array[0:3, 0] = np.copy(cam_position[0:3])
            display_array[3] = np.copy(mat_cond)
            display_mat = np.hstack((display_mat, display_array))
            # print "-- valid cam position --\n",cam.get_world_position()
            # print "-- cond num --\n", mat_cond
        #------------------------------- If End------------------------------------------------------
        # ------------- plot the image points dynamiclly-----------------
        #plotImagePointsDyn(imagePoints)
    #----------------------------For End-------------------------------------------------------------------

    #-----------------Transfer_condNumMatrix_to_accuracyDegreeMatrix-------------------------
    min_cond_num = min(mat_cond_list)
    max_cond_num = max(mat_cond_list)
    degree = 5 # Set the degree of accuracy as 5! we can change this

    accuracy_mat_final = np.divide(accuracy_mat_new,num_mat,out=np.zeros_like(accuracy_mat_new), where=num_mat != 0)# get the average cond num in each cell
    # 27.04.2017 DO not need to convert accuracy_mat_final to accuracyDegreeMatrix
    # accuracyDegreeMatrix = transfer_condNumMatrix_to_accuracyDegreeMatrix(accuracy_mat_final,min_cond_num,max_cond_num,degree)
    display_mat = display_mat[:,1:]
    # -----------------For Loop End--------------------------------------------------------
    print "=============================Info====================================================="
    print "-- Best image point --\n", imagePoints_des[mat_cond_list.index(min(mat_cond_list))]
    print "-- Min cond num --\n", min(mat_cond_list)
    print "-- Best cam position --\n", cam_valid[mat_cond_list.index(min(mat_cond_list))].get_world_position()
    print "-------------------------"
    print "-- Max cond num --\n", max(mat_cond_list)
    print "-- Worst cam position --\n", cam_valid[mat_cond_list.index(max(mat_cond_list))].get_world_position()

    print "-- Valid cams size--\n", len(cam_valid)
    print "-- Cond num size--\n", len(mat_cond_list)
    print "-- ImagePoints_des size --\n", len(imagePoints_des)
    print "-- Display_mat shappe --\n", display_mat.shape
    print "========================================================================================"
    #-------------------------Return parameters-------------------------------------------
    inputX = np.copy(display_mat[0,:])
    inputY = np.copy(display_mat[1,:])
    inputZ = np.copy(display_mat[2,:])

    input_ippe1_t = np.copy(ippe_tvec_error_list1)
    input_ippe1_R = np.copy(ippe_rmat_error_list1)
    input_ippe2_t = np.copy(ippe_tvec_error_list2)
    input_ippe2_R = np.copy(ippe_rmat_error_list2)
    input_pnp_t = np.copy(pnp_tvec_error_list)
    input_pnp_R = np.copy(pnp_rmat_error_list)
    input_transfer_error = np.copy(transfer_error_list)

    return inputX,inputY,inputZ,input_ippe1_t,input_ippe1_R,input_ippe2_t,input_ippe2_R,input_pnp_t,input_pnp_R,input_transfer_error,display_mat,accuracy_mat_final


def transfer_condNumMatrix_to_accuracyDegreeMatrix(condNumMatrix,min_cond_num,max_cond_num,degree):
    """
    Transfer condNum matrix to accuracy degree matrix,
    """
    # divide condition num distribution into 5 degree
    degree_step = (max_cond_num - min_cond_num) / degree
    accuracy_degree1 = max_cond_num - 4 * degree_step
    accuracy_degree2 = max_cond_num - 3 * degree_step
    accuracy_degree3 = max_cond_num - 2 * degree_step
    accuracy_degree4 = max_cond_num - degree_step

    row_area_matrix = condNumMatrix.shape[0]
    col_area_matrix = condNumMatrix.shape[1]
    accuracyDegreeMatrix = np.copy(condNumMatrix)
    for i in range(row_area_matrix):
        for j in range(col_area_matrix):
            cond_num = condNumMatrix[i,j]
            if cond_num >= accuracy_degree4:
                accuracyDegreeMatrix[i,j] = 5
            elif cond_num >= accuracy_degree3:
                accuracyDegreeMatrix[i, j] = 4
            elif cond_num >= accuracy_degree2:
                accuracyDegreeMatrix[i, j] = 3
            elif cond_num >= accuracy_degree1:
                accuracyDegreeMatrix[i, j] = 2
            elif cond_num >= min_cond_num:
                accuracyDegreeMatrix[i, j] = 1
            else:
                accuracyDegreeMatrix[i, j] = 0
    return accuracyDegreeMatrix


def camPositionInRealMatGrid(cam_r, cam_angle, accuracy_mat_new_rowNum, accuracy_mat_new_colNum):
    cam_z = cam_r * np.sin(np.deg2rad(cam_angle))  # row
    cam_y = cam_r * np.cos(np.deg2rad(cam_angle))  # column
    cam_z_str = str(round(cam_z, 4))
    cam_y_str = str(round(cam_y, 4))

    accuracy_mat_row_list = []
    accuracy_mat_col_list = []

    # mat row  Z
    if decimal.Decimal(cam_z_str) % decimal.Decimal('.1') == 0:
        tem = int(math.floor(round(cam_z, 4) * 10))
        if tem > 0:
            accuracy_mat_row = tem - 1
            accuracy_mat_row_list.append(accuracy_mat_row)
        if tem < accuracy_mat_new_rowNum:
            accuracy_mat_row = tem
            accuracy_mat_row_list.append(accuracy_mat_row)
    else:
        tem = int(math.floor(round(cam_z, 4) * 10))
        accuracy_mat_row = tem
        accuracy_mat_row_list.append(accuracy_mat_row)

    # mat column  Y
    if decimal.Decimal(cam_y_str) % decimal.Decimal('.1') == 0:
        tem = int(math.ceil(round(cam_y, 4) * 10))
        if (-tem + 30) > 0:
            accuracy_mat_col = -tem + 30 - 1
            accuracy_mat_col_list.append(accuracy_mat_col)
        if (-tem + 30) < accuracy_mat_new_colNum:
            accuracy_mat_col = -tem + 30
            accuracy_mat_col_list.append(accuracy_mat_col)
    else:
        tem = int(math.ceil(round(cam_y, 4) * 10))  # avoid 4 decimal point
        accuracy_mat_col = -tem + 30
        accuracy_mat_col_list.append(accuracy_mat_col)

    return accuracy_mat_row_list, accuracy_mat_col_list


def updateAccuracySumMat(accuracy_mat_row_list, accuracy_mat_col_list, num_mat, accuracy_mat_new, mat_cond):
    for i_row in accuracy_mat_row_list:
        for i_col in accuracy_mat_col_list:
            num_mat[i_row, i_col] += 1
            accuracy_mat_new[i_row, i_col] += mat_cond  # Store the condition num at corresponding position


def computeError(cam, imagePoints, transfer_error_list, ippe_tvec_error_list1, ippe_rmat_error_list1,
                 ippe_tvec_error_list2, ippe_rmat_error_list2, pnp_tvec_error_list, pnp_rmat_error_list):
    transfer_error_loop = []
    ippe_tvec_error_loop1 = []
    ippe_rmat_error_loop1 = []
    ippe_tvec_error_loop2 = []
    ippe_rmat_error_loop2 = []
    pnp_tvec_error_loop = []
    pnp_rmat_error_loop = []
    new_imagePoints = np.copy(imagePoints)
    for j in range(homography_iters):
        new_imagePoints_noisy = cam.addnoise_imagePoints(new_imagePoints, mean=0, sd=4)
        # Calculate the pose using IPPE (solution with least repro error)
        normalizedimagePoints = cam.get_normalized_pixel_coordinates(new_imagePoints_noisy)
        ippe_tvec1, ippe_rmat1, ippe_tvec2, ippe_rmat2 = pose_ippe_both(new_objectPoints, normalizedimagePoints,
                                                                        debug=False)
        # Calculate the pose using ippeCam1,ippeCam2
        ippeCam1 = cam.clone_withPose(ippe_tvec1, ippe_rmat1)
        ippeCam2 = cam.clone_withPose(ippe_tvec2, ippe_rmat2)
        # Calculate the pose using solvepnp
        debug = False
        # TODO  cv2.SOLVEPNP_DLS, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE
        pnp_tvec, pnp_rmat = pose_pnp(new_objectPoints, new_imagePoints_noisy, cam.K, debug, cv2.SOLVEPNP_EPNP, False)
        pnpCam = cam.clone_withPose(pnp_tvec, pnp_rmat)
        # Calculate errors
        pnp_tvec_error, pnp_rmat_error = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, pnpCam.get_tvec(),
                                                                      pnp_rmat)
        pnp_tvec_error_loop.append(pnp_tvec_error)
        pnp_rmat_error_loop.append(pnp_rmat_error)

        ippe_tvec_error1, ippe_rmat_error1 = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R,
                                                                          ippeCam1.get_tvec(), ippe_rmat1)
        ippe_tvec_error2, ippe_rmat_error2 = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R,
                                                                          ippeCam2.get_tvec(), ippe_rmat2)
        ippe_tvec_error_loop1.append(ippe_tvec_error1)
        ippe_rmat_error_loop1.append(ippe_rmat_error1)
        ippe_tvec_error_loop2.append(ippe_tvec_error2)
        ippe_rmat_error_loop2.append(ippe_rmat_error2)

        # Homography Estimation from noisy image points
        Xo = new_objectPoints[[0, 1, 3], :]
        Xi = new_imagePoints_noisy
        # Hnoisy,A_t_ref,H_t = homo2d.homography2d(Xo,Xi)
        # Hnoisy = Hnoisy/Hnoisy[2,2]
        # TODO Change H
        # HO Method
        # Hnoisy = hh(Xo, Xi)
        # OpenCV Method
        Hnoisy_OpenCV, _ = cv2.findHomography(Xo[:2].T.reshape(1, -1, 2), Xi[:2].T.reshape(1, -1, 2))
        ## ERRORS FOR THE NOISY HOMOGRAPHY
        ## VALIDATION OBJECT POINTS
        validation_objectPoints = validation_plane.get_points()
        validation_imagePoints = np.array(cam.project(validation_objectPoints, False))
        Xo = np.copy(validation_objectPoints)
        Xo = np.delete(Xo, 2, axis=0)
        Xi = np.copy(validation_imagePoints)
        transfer_error_loop.append(ef.validation_points_error(Xi, Xo, Hnoisy_OpenCV))

    transfer_error_list.append(np.mean(transfer_error_loop))
    ippe_tvec_error_list1.append(np.mean(ippe_tvec_error_loop1))
    ippe_rmat_error_list1.append(np.mean(ippe_rmat_error_loop1))
    ippe_tvec_error_list2.append(np.mean(ippe_tvec_error_loop2))
    ippe_rmat_error_list2.append(np.mean(ippe_rmat_error_loop2))
    pnp_tvec_error_list.append(np.mean(pnp_tvec_error_loop))
    pnp_rmat_error_list.append(np.mean(pnp_rmat_error_loop))


def plotImagePointsDyn(imagePoints):
    # ------------- plot the image points dynamiclly-----------------
    print "imagePoints\n", imagePoints
    fig1 = plt.figure('Image points')
    ax_image = fig1.add_subplot(211)
    ax_image.cla()
    plt.sca(ax_image)
    plt.ion()
    ax_image.plot(imagePoints[0], imagePoints[1], '.', color='blue', )
    ax_image.set_xlim(0, 1280)
    ax_image.set_ylim(0, 960)
    ax_image.invert_yaxis()
    ax_image.set_title('Image Points')
    plt.show()
    plt.pause(0.001)





# ===========================================Test============================================
# condNumMatrix = np.array([[100,200,300,400,500,],[600,700,800,99,20],[0,0,0,0,0]])
# print transfer_condNumMatrix_to_accuracyDegreeMatrix(condNumMatrix,100,500,5)
# ---------------------------Test normalise_points----------------------------------------
# TODO 12.03.2018 at 11:30
# cams = []
# cam.set_t(0., 0.40957603, 0.28678823,"world")
# cam.look_at([0, 0, 0])
# cams.append(cam)
# accuracy_mat = np.zeros([5,5])
# radius_step = 0.1
# angle_step = 5
# angle_begin = 0.0
# angle_end = 180.0
# angle_num = 3  # 37 TODO need to set
# angle_step = (angle_end - angle_begin) / (angle_num - 1)
# theta_params = (angle_begin, angle_end, angle_num)
#
# r_begin = 0.1
# r_end = 3.0
# r_num = 3  # 31 TODO need to set
# r_step = (r_end - r_begin) / (r_num - 1)
# r_params = (r_begin, r_end, r_num)
#
# heightGetCondNum(cams,accuracy_mat,theta_params,r_params)

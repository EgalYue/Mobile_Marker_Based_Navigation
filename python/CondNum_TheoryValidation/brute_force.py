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
from vision.camera_distribution import create_cam_distribution
# from vision.rt_matrix import R_matrix_from_euler_t
from vision.camera import Camera
from vision.plane import Plane
from vision.circular_plane import CircularPlane
import matplotlib.pyplot as plt
import scipy.io as sio
import error_functions as ef
from ippe import homo2d
from homographyHarker.homographyHarker import homographyHarker as hh
import hT_gradient as gd
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t
from solve_ippe import pose_ippe_both
from solve_pnp import pose_pnp
import cv2
import display_condNum as dc

calc_metrics = False
number_of_points = 4

## Define a Display plane with random initial points
# TODO Change the radius of circular plane
pl = CircularPlane(origin=np.array([0., 0., 0.]), normal=np.array([0, 0, 1]), radius=0.15, n=4)
pl.random(n=number_of_points, r=0.01, min_sep=0.01)
plane_size = (0.3, 0.3)

## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
cam.set_width_heigth(640, 480)

## CREATE A SET OF IMAGE POINTS FOR VALIDATION OF THE HOMOGRAPHY ESTIMATION
# This will create a grid of 16 points of size = (0.3,0.3) meters
validation_plane = Plane(origin=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), size=(0.3, 0.3), n=(4, 4))
validation_plane.uniform()

## we create the gradient for the point distribution
normalize = False

# 4 points: An ideal square
x1 = round(pl.radius * np.cos(np.deg2rad(45)), 3)
y1 = round(pl.radius * np.sin(np.deg2rad(45)), 3)
objectPoints_square = np.array(
    [[x1, -x1, -x1, x1],
     [y1, y1, -y1, -y1],
     [0., 0., 0., 0.],
     [1., 1., 1., 1.]])

new_objectPoints = np.copy(objectPoints_square)
# print "new_objectPoints", new_objectPoints
# ------------------------All cams point straight down: Test Height factor---------------------------------------
# TODO cam distribution position PARAMETER CHANGE!!!
cams = create_cam_distribution(cam, plane_size,
                               theta_params=(0, 360, 30), phi_params=(0, 70, 10),
                               r_params=(0.2, 2.0, 20), plot=False)
homography_iters = 1000     # TODO homography_iters changed

def heightGetCondNum(cams,accuracy_mat,radius_step,angle_step):
    # fig1 = plt.figure('Image points')
    # ax_image = fig1.add_subplot(211)
    accuracy_mat_new = np.copy(accuracy_mat)
    mat_cond_list = []
    imagePoints_des = []
    cam_valid = []
    display_mat = np.array([[0.0], [0.0], [0.0], [0.0]])
    # ------------------------for all cams------------------------------
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

            accuracy_mat_row = int(np.copy(cam.radius) / radius_step) # row index, accuracy_mat store cond num for each cam position
            # print "radius_step",radius_step
            # print "accuracy_mat_row",accuracy_mat_row
            # print "valid cam r \n",cam.radius
            accuracy_mat_col = int(np.copy(cam.angle) / angle_step) # col index, accuracy_mat store cond num for each cam position
            # print "angle_step",angle_step
            # print "accuracy_mat_col",accuracy_mat_col
            # print "accuracy_mat",accuracy_mat.shape
            # print "valid cam angle \n", cam.angle
            # ------------------------Calculate Error------------------------
            transfer_error_loop = []
            ippe_tvec_error_loop1 = []
            ippe_rmat_error_loop1 = []
            ippe_tvec_error_loop2 = []
            ippe_rmat_error_loop2 = []
            pnp_tvec_error_loop = []
            pnp_rmat_error_loop = []
            new_imagePoints = np.copy(imagePoints)
            for j in range(homography_iters):
                # TODO change sd
                new_imagePoints_noisy = cam.addnoise_imagePoints(new_imagePoints, mean=0, sd=4)
                # Calculate the pose using IPPE (solution with least repro error)
                normalizedimagePoints = cam.get_normalized_pixel_coordinates(new_imagePoints_noisy)
                ippe_tvec1, ippe_rmat1, ippe_tvec2, ippe_rmat2 = pose_ippe_both(new_objectPoints, normalizedimagePoints,
                                                                                debug=False)
                ippeCam1 = cam.clone_withPose(ippe_tvec1, ippe_rmat1)
                ippeCam2 = cam.clone_withPose(ippe_tvec2, ippe_rmat2)

                # Calculate the pose using solvepnp
                debug = False
                # TODO  cv2.SOLVEPNP_DLS, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE
                pnp_tvec, pnp_rmat = pose_pnp(new_objectPoints, new_imagePoints_noisy, cam.K, debug, cv2.SOLVEPNP_DLS,
                                              False)
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
                # TODO replace Xi = new_imagePoints_noisy with Xi = normalizedimagePoints
                Xi = new_imagePoints_noisy
                # Xi = normalizedimagePoints
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
            # ----------------------------------------------------------------------------------------
            imagePoints_des.append(np.array(cam.project(objectPoints, False)))
            input_list = gd.extract_objectpoints_vars(objectPoints)
            input_list.append(np.array(cam.K))
            input_list.append(np.array(cam.R))
            input_list.append(cam.t[0, 3])
            input_list.append(cam.t[1, 3])
            input_list.append(cam.t[2, 3])
            mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)

            accuracy_mat_new[accuracy_mat_row,accuracy_mat_col] = mat_cond # Store the condition num at corresponding position
            mat_cond_list.append(mat_cond)
            print "valid cam position:",cam.get_world_position()
            cam_valid.append(cam)
            print "mat_cond=", mat_cond
            # write the data(cam position + cond num) to  file, its a 4*1 array
            cam_position = cam.get_world_position()
            display_array = np.eye(4, 1, dtype=float)
            display_array[0:3, 0] = np.copy(cam_position[0:3])
            display_array[3] = np.copy(mat_cond)
            display_mat = np.hstack((display_mat, display_array))

        # ax_image.cla()
        # plt.sca(ax_image)
        # plt.ion()
        # ax_image.plot(imagePoints[0], imagePoints[1], '.', color='blue', )
        # ax_image.set_xlim(0, 1280)
        # ax_image.set_ylim(0, 960)
        # ax_image.invert_yaxis()
        # ax_image.set_title('Image Points')
        # plt.show()
        # plt.pause(0.001)
    #-----------------transfer_condNumMatrix_to_accuracyDegreeMatrix-------------------------
    min_cond_num = min(mat_cond_list)
    max_cond_num = max(mat_cond_list)
    degree = 5
    accuracyDegreeMatrix = transfer_condNumMatrix_to_accuracyDegreeMatrix(accuracy_mat_new,min_cond_num,max_cond_num,degree)
    display_mat = display_mat[:,1:]
    # -----------------For Loop End--------------------------------------------------------
    print "--best image points--", imagePoints_des[mat_cond_list.index(min(mat_cond_list))]
    print "--cond num min--", min(mat_cond_list)
    print "--cam position Best--", cam_valid[mat_cond_list.index(min(mat_cond_list))].get_world_position()
    print "-------------------------"
    print "--cond num max--", max(mat_cond_list)
    print "--cam position Worst--", cam_valid[mat_cond_list.index(max(mat_cond_list))].get_world_position()

    print "--cam valid size--", len(cam_valid)
    print "--cond num size--", len(mat_cond_list)
    print "--imagePoints_des size--", len(imagePoints_des)
    print "--display_mat--", display_mat.shape


    # # -----------------------Draw the image points-----------------------------------------------------------------------
    # fig1 = plt.figure('Image points')
    # ax_image_best = fig1.add_subplot(212)
    # plt.sca(ax_image_best)
    # ax_image_best.plot(imagePoints_des[mat_cond_list.index(min(mat_cond_list))][0],
    #                    imagePoints_des[mat_cond_list.index(min(mat_cond_list))][1], '.', color='blue', )
    # ax_image_best.set_xlim(0, 1280)
    # ax_image_best.set_ylim(0, 960)
    # ax_image_best.invert_yaxis()
    # ax_image_best.set_title('Image Points')
    # plt.show()

    ## plt.pause(100)

    ##------------Display cond num distribution-------------------
    # print "start to show "
    display_mat = display_mat[:,1:]
    # dc.displayCondNumDistribution(display_mat)
    # dc.displayCondNumDistriMayavi(display_mat)
    # print "finish!!!"
    #-------------Display error---------------------------------
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
    # ------------------Error-----------------------------
    print transfer_error_list
    print ippe_tvec_error_list1
    print ippe_rmat_error_list1
    print ippe_tvec_error_list2
    print ippe_rmat_error_list2
    print pnp_rmat_error_list
    return inputX,inputY,inputZ,input_ippe1_t,input_ippe1_R,input_ippe2_t,input_ippe2_R,input_pnp_t,input_pnp_R,input_transfer_error,display_mat,accuracyDegreeMatrix

def transfer_condNumMatrix_to_accuracyDegreeMatrix(condNumMatrix,min_cond_num,max_cond_num,degree):
    """
    Transfer condNum matrix to accuracy degree matrix,
    """
    # divide condition num distribution into 5 degree
    degree_step = (max_cond_num - min_cond_num) / (degree - 1)
    accuracy_degree2 = max_cond_num - 3 * degree_step
    accuracy_degree3 = max_cond_num - 2 * degree_step
    accuracy_degree4 = max_cond_num - degree_step

    row_area_matrix = condNumMatrix.shape[0]
    col_area_matrix = condNumMatrix.shape[1]
    accuracyDegreeMatrix = np.copy(condNumMatrix)
    for i in range(row_area_matrix):
        for j in range(col_area_matrix):
            cond_num = condNumMatrix[i,j]
            if cond_num >= max_cond_num:
                accuracyDegreeMatrix[i,j] = 5
            elif cond_num >= accuracy_degree4:
                accuracyDegreeMatrix[i, j] = 4
            elif cond_num >= accuracy_degree3:
                accuracyDegreeMatrix[i, j] = 3
            elif cond_num >= accuracy_degree2:
                accuracyDegreeMatrix[i, j] = 2
            elif cond_num >= min_cond_num:
                accuracyDegreeMatrix[i, j] = 1
            else:
                accuracyDegreeMatrix[i, j] = 0
    return accuracyDegreeMatrix


# ===========================================Test============================================
# condNumMatrix = np.array([[100,200,300,400,500,],[600,700,800,99,20],[0,0,0,0,0]])
# print transfer_condNumMatrix_to_accuracyDegreeMatrix(condNumMatrix,100,500,5)
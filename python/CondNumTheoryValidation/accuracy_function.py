#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Yue Hu on 03.03.2018
"""
Creat a accuracy function used to determine the condition number of valid area of marker.
The accuracy degree of marker based on the accuracy function is divided into 5 types: High accuracy -> ... -> Low -> no detection 
"""
from __future__ import division # set / as float!!!!
import sys
sys.path.append("..")
import numpy as np
import math
from vision.camera import Camera
from vision.camera_distribution import create_cam_distribution,create_cam_distribution_in_YZ,create_cam_distribution_in_XZ
import brute_force as bf

def accuracy_func(plane_size=(0.3, 0.3),camera_intrinsic_params=(800.,800.,320.,240.), theta_params=(0.0, 180.0, 5), r_params=(0.1, 2.0, 5), z=0, y=0):
    """
    Calculate the accuracy degree of given position y,z in world

    y:
    z:
    :return 0 means the cam is out of range of marker
    """

    # --------------------------------------accuracy_degree_distribution---------------------
    cam = Camera()
    fx = camera_intrinsic_params[0]
    fy = camera_intrinsic_params[1]
    cx = camera_intrinsic_params[2]
    cy = camera_intrinsic_params[3]
    cam.set_K(fx=fx, fy=fy, cx=cx, cy=cy)
    width = cx * 2
    height = cy * 2
    cam.set_width_heigth(width, height)

    cams,accuracy_mat = create_cam_distribution_in_YZ(cam=None, plane_size=plane_size, theta_params=theta_params, r_params=r_params,
                                  plot=False)

    accuracy_mat = np.zeros([30, 60])  # define the new acc_mat as 30 x 30 ,which means 3m x 3m area in real world and each cell is 0.1m x 0.1m
    inputX, inputY, inputZ, input_ippe1_t, input_ippe1_R, input_ippe2_t, input_ippe2_R, input_pnp_t, input_pnp_R, input_transfer_error, display_mat,accuracy_mat_new = bf.heightGetCondNum(cams,accuracy_mat,theta_params, r_params)

    # print accuracy_mat_new
    # --------------------------------------------------------------------------------------------
    degree = 5  # divide accuracy into 5 degree
    r = np.sqrt(z * z + y * y)
    # out of range
    if r >= r_params[1] or r == 0:
        return 0
    else:
        # according to radius and angle of cam, Calculate the index of accuracy matrix
        r_begin = r_params[0]
        r_end = r_params[1]
        r_num = r_params[2]
        r_step = (r_end - r_begin) / (r_num - 1)
        # print "r_step",r_step
        half_r_step = r_step / 2.0
        r_num1 = math.floor(r / r_step)
        r_num2 = math.floor((r - r_step * r_num1) / half_r_step)
        row_index = int(r_num1 + r_num2)  # row index

        angle_begin = theta_params[0]
        angle_end = theta_params[1]
        angle_num = theta_params[2]
        angle_step = (angle_end - angle_begin) / (angle_num - 1)
        # print "angle_step",angle_step
        half_angle_step = angle_step / 2.0
        angle = np.rad2deg(np.arccos(y / r))  # deg
        angle_num1 = math.floor(angle / angle_step)
        angle_num2 = math.floor((angle - angle_step * angle_num1) / half_angle_step)
        col_index = int(angle_num1 + angle_num2)
        # print "row_index,col_index",row_index,col_index
        acc_degree = accuracy_mat_new[row_index, col_index]
        # print "acc_degree",acc_degree
        return acc_degree


# =========================================Test=========================================================

# print accuracy_func(plane_size=(0.3, 0.3),camera_intrinsic_params=(800.,800.,320.,240.), theta_params=(0.0, 180.0, 5), r_params=(0.1, 2.0, 5), z=1, y=0)









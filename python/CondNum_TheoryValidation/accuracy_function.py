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

def accuracy_func(accuracy_matrix, theta_params = (0.0,180.0,5), r_params = (0.0, 2.0, 5) , z = 0, y = 0):
    """
    Calculate the accuracy degree of given position y,z in world 
    
    y: 
    z: 
    :return 0 means the cam is out of range of marker
    """
    degree = 5 # divide accuracy into 5 degree
    r = np.sqrt(z*z + y*y)
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
        row_index = int(r_num1 + r_num2) # row index

        angle_begin = theta_params[0]
        angle_end = theta_params[1]
        angle_num = theta_params[2]
        angle_step = (angle_end - angle_begin) / (angle_num - 1)
        # print "angle_step",angle_step
        half_angle_step = angle_step / 2.0
        angle = np.rad2deg(np.arccos(y / r)) # deg
        angle_num1 = math.floor(angle / angle_step)
        angle_num2 = math.floor((angle - angle_step * angle_num1) / half_angle_step)
        col_index = int(angle_num1 + angle_num2)
        # print "row_index,col_index",row_index,col_index
        acc_degree = accuracy_matrix[row_index, col_index]
        # print "acc_degree",acc_degree
        return acc_degree

def accuracy_degree_distribution():
    """
    Using a matrix to describe accuracy degree
    :return: an accuracy matrix for detection area of marker  
    """
    # TODO Change the radius of circular plane
    cam = Camera()
    cam.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    cam.set_width_heigth(640, 480)
    angle_begin = 0.0
    angle_end = 180.0
    angle_num = 4
    angle_step = (angle_end - angle_begin) / (angle_num - 1)
    theta_params = (angle_begin,angle_end,angle_num)

    r_begin = 0.0
    r_end = 2.0
    r_num = 5
    r_step = (r_end - r_begin) / (r_num - 1)
    r_params = (r_begin,r_end,r_num)
    # TODO cam distribution position PARAMETER CHANGE!!!
    cams,accuracy_mat = create_cam_distribution_in_YZ(cam=None, plane_size=(0.3, 0.3), theta_params=theta_params, r_params=r_params,
                                  plot=False)
    inputX, inputY, inputZ, input_ippe1_t, input_ippe1_R, input_ippe2_t, input_ippe2_R, input_pnp_t, input_pnp_R, input_transfer_error, display_mat,accuracy_mat_new = bf.heightGetCondNum(cams,accuracy_mat,r_step,angle_step)
    # print "accuracy_mat distribution:\n",accuracy_mat_new
    return accuracy_mat_new


# =========================================Test=========================================================
accuracy_mat_new = accuracy_degree_distribution()
print "accuracy_mat_new",accuracy_mat_new
print "result: ",accuracy_func(accuracy_mat_new, theta_params = (0.0,180.0,5), r_params = (0.0, 2.0, 5) , z = 1, y = 0)

#=========================================Test==========================================================
# mat = np.eye(20,36)
# accuracy_func(mat,0.1,0.1)








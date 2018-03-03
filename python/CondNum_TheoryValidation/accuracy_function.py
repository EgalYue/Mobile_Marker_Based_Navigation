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

def accuracy_func(area_matrix, z, y):
    """
    Calculate the accuracy degree of given position y,z in world 
    
    y: 
    z: 
    :return -1 means the cam is out of range of marker
    """
    # Now only assume that the distance of marker is [0,2] m.
    distance = 2.0
    r_step = 0.1
    r_grid_num = int(distance/r_step) # matrix row num
    full_angle = 180.0
    angle_step = 5.0
    angle_grid_num = int(full_angle/angle_step) # matrix col num

    r = np.sqrt(z*z + y*y)
    row_index = int(math.floor(r * 10)) # [0,20]
    angle = np.rad2deg(np.arccos(y / r)) # deg
    col_index = int(math.floor(angle / angle_step)) # [0,36]
    # print "row_index",row_index
    # print "col_index", col_index
    if r >= distance:
        return -1
    else:
        acc_degree = area_matrix[row_index, col_index]
        return acc_degree





# ===================================Test=========================================
mat = np.eye(20,36)
accuracy_func(mat,0.1,0.1)







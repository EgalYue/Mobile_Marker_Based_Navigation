#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 16.04.18 12:50
@File    : pathFinding.py
@author: Yue Hu
"""
import sys
sys.path.append("..")

import Rt_matrix_from_euler_zyx as R_matrix_from_euler_zyx
import numpy as np
import decimal
from scipy.linalg import expm, rq, det, inv
import os  # Read matrix form file



def cellCenterPosition(path,grid_step):
    """
    Get the exact position from the center of each cell
    Based on Real World Coordinate System [30,60]
    :param pos:
    :param grid_step:
    :return:
    """
    real_path = []
    for pos in path:
        x_str = str(pos[0] * grid_step+grid_step / 2)
        y_str = str(pos[1] * grid_step+grid_step / 2)
        x = float(decimal.Decimal(x_str))
        y = float(decimal.Decimal(y_str))
        real_path.append(np.array([x,y]))
    return real_path

def marker_set_t(x,y,z,R, frame = 'world'):
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
    if frame=='world':
      marker_world = np.array([x,y,z,1]).T
      marker_t = np.dot(R,-marker_world)
      t[:3,3] = marker_t[:3]
      return t
    else:
      t[:3,3] = np.array([x,y,z])
      return t

def getMarkerTransformationMatrix(width,height,cell_length):
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
    R = R_matrix_from_euler_zyx.R_matrix_from_euler_zyx(alpha,beta,gamma)
    x = 0.0
    y = width / 2.0 * cell_length
    z = 0.0
    t = marker_set_t(x, y, z, R, frame='world')
    T = np.dot(t,R) # Transformation matrix between World and Marker coordinate systems
    return T

def moveTo(current_pos,next_pos):
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
    return np.array([x_step,y_step])

def getCameraPosInWorld(T_WC):
    """
    Get the camear position in the real world cordinate system
    :param T:
    :return:
    """
    t = np.dot(inv(T_WC), np.array([0, 0, 0, 1]))
    cam_x = t[0]
    cam_y = t[1]
    return np.array([cam_x,cam_y])



cell_length = 0.1 # The length of each cell is 0.1m. Each cell of matrix is 0.1m x 0.1m.
width = 30
height = 60
cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../CondNum_TheoryValidation_newAccMat/accuracyMatrix.txt', cur_path)
f = open(new_path, 'r')
l = [map(int, line.split(' ')) for line in f]
accuracy_mat = np.asarray(l)  # convert to matrix
#--------------------Test for a simple path----------------------------------------
#                    A[26,18] -> B[26,22]
#                    A - - - B
path = [np.array([26,18]),np.array([26,19]),np.array([26,20]),np.array([26,21]),np.array([26,22])]
fix_path = cellCenterPosition(path,cell_length)
print "-- fix_path --:\n",fix_path

T_WM = getMarkerTransformationMatrix(width,height,cell_length)
#------------------------ Initialization---------------------
cam_pos_real_current = np.array([2.65,1.85])
cam_pos_measured_current = cam_pos_real_current
move_dis = moveTo(cam_pos_measured_current, fix_path[1])

real_path =[]
real_path.append(cam_pos_real_current)
measured_path = []
measured_path.append(cam_pos_measured_current)
#------------------------------------------------------------
for i in range(1,len(fix_path)):

    T_MC =  # TODO
    T_WC = np.dot(T_MC, T_WM)
    cam_pos_measured_current = getCameraPosInWorld(T_WC)
    # cam_pos_measured_current = fix_path[i] #TEST
    measured_path.append(cam_pos_measured_current)
    # Update camera current real position
    cam_pos_real_current = cam_pos_real_current + move_dis # this move_dis is the previous value
    real_path.append(cam_pos_real_current)

    # Update move_dis
    if i == len(fix_path) - 1:
        move_dis = [0.0,0.0]
    else:
        move_dis = moveTo(cam_pos_measured_current, fix_path[i + 1])



#========================================Test===============================================
print "-- measured_path --:\n",measured_path
print "-- real_path --:\n",real_path







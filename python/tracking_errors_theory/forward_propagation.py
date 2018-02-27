#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This class is used to verify forward propagation from object points to image points


@Time    : 27.02.18 09:02
@File    : forward_propagation.py
@author: Yue Hu
"""
from __future__ import division # set / as float!!!!
import sys
sys.path.append("..")

import numpy as np
from vision.circular_plane import CircularPlane
import numdifftools as nd
from vision.camera import Camera
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t



def linear_covariance_f(var_cov,A):
    """
    Forward propagation for linear function
    :param var_mean:
    :param var_cov:
    :return:
    """
    cov_f = np.dot(A,np.dot(var_cov,A.T))
    return cov_f

def nonlinear_covariance_f(var_cov, jf):
    """
    Forward propagation for nonlinear function
    :param jf:
    :param A:
    :return:
    """
    cov_f = np.dot(jf,np.dot(var_cov,jf.T))
    return cov_f

def nonlinear_covariance_f_obj_to_image(cam,covariance_alpha_belt_r):
    """

    """
    pl = CircularPlane(origin=np.array([0., 0., 0.]), normal = np.array([0, 0, 1]), radius=0.15, n = 4)
    x1  = round(pl.radius*np.cos(np.deg2rad(45)),3)
    y1  = round(pl.radius*np.sin(np.deg2rad(45)),3)
    objectPoints_square= np.array(
    [[ x1, -x1, -x1, x1],
     [ y1, y1, -y1, -y1],
     [ 0., 0., 0., 0.],
     [ 1., 1., 1., 1.]])

    new_objectPoints = np.copy(objectPoints_square)

    P = cam.P
    # derivation value
    d_alpha = 0.0
    d_belt = 0.0
    d_r = 0.0
    # Jacobian function at (0,0,0)
    j_f = jacobian_function([d_alpha, d_belt, d_r], P, new_objectPoints)
    cov_f = nonlinear_covariance_f(covariance_alpha_belt_r, j_f)
    return cov_f



def jacobian_function(input, P, obj_point):
    """
    Use your Jacobian function at any point you want.
    """
    f_jacob = nd.Jacobian(f)

    # 2*4(points) equations
    matrix_point1 = f_jacob(input, P, obj_point[:, 0])
    matrix_point2 = f_jacob(input, P, obj_point[:, 1])
    matrix_point3 = f_jacob(input, P, obj_point[:, 2])
    matrix_point4 = f_jacob(input, P, obj_point[:, 3])
    jacobian_funs = np.vstack((matrix_point1, matrix_point2, matrix_point3, matrix_point4))
    return jacobian_funs


def f(input, P, obj_point):
    """
    Define your function
    Can be R^n -> R^n as long as you use numpy arrays as output
    """

    alpha = input[0]
    belt = input[1]
    r = input[2]

    x = r * np.cos(belt) * np.sin(alpha)
    y = r * np.sin(belt) * np.sin(alpha)
    z = r * np.cos(alpha)

    image_point = np.dot(P, obj_point)
    # print "image_point\n", image_point

    # u = image_point[0,0]/image_point[0,2]
    # v = image_point[0,1]/image_point[0,2]
    # TODO u v ??
    u = image_point[0, 0]
    v = image_point[0, 1]
    return np.array([u, v])

#=============================Code End=========================================

#=============================Test=============================================

#-----------------------------covariance_f-------------------------------------
# var_cov = np.array([[4,0],[0,5]])
# A = np.array([[1,2],[3,4]])
# print covariance_f(var_cov,A)
# theta,vec= np.linalg.eig(covariance_f(var_cov,A)) # eigenvalues of covariance matrix
# print theta

#-----------------------------nonlinear_covariance_f_obj_to_image------------------------------------------------
cam = Camera()
cam.set_K(fx = 800,fy = 800,cx = 640/2.,cy = 480/2.)
cam.set_width_heigth(640,480)
cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(180), 0))
cam.set_t(0., 0., 0.5, 'world')
covariance_alpha_belt_r = np.array([[1,0,0],[0,2,0],[0,0,3]])
print nonlinear_covariance_f_obj_to_image(cam,covariance_alpha_belt_r)
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 07.02.18 18:59
@File    : covariance_mat_sphericalCoordinate.py
@author: Yue Hu
"""
import numpy as np
import numdifftools as nd


def covariance_alpha_belt_r(alpha, belt, r, cam):
    """
    Covariance matrix of the spherical angles α, β and the distance r

    """

    # x = r * np.cos(belt) * np.sin(alpha)
    # y = r * np.sin(belt) * np.sin(alpha)
    # z = r * np.cos(alpha)

    cam = cam.clone()
    K = cam.K
    T = cam.Rt[:3,:4]

    # j_f should be 8*3 for 4 ponits
    j_f = jacobian_function([alpha,belt,r],K,T)
    # TODO
    cov_mat_p1 =
    cov_mat_p2 =
    cov_mat_p3 =
    cov_mat_p4 =
    block_mat_image4points = np.eye(8,8)
    block_mat_image4points[0:1, 0:1] = cov_mat_p1
    block_mat_image4points[2:3, 2:3] = cov_mat_p2
    block_mat_image4points[4:5, 4:5] = cov_mat_p3
    block_mat_image4points[6:7, 6:7] = cov_mat_p4

    cov_mat = np.inv((j_f.T) * (np.inv(block_mat_image4points)) * (j_f))
    return cov_mat



def f(input, K, T):
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

    obj_point = [x,y,z]
    P = np.dot(K, T)
    image_point = np.dot(P, obj_point)
    # u = (P[0, 0] * x + P[0, 1] * y + P[0, 2] * z) / (P[2, 0] * x + P[2, 1] * y + P[2, 2] * z)
    # v = (P[1, 0] * x + P[1, 1] * y + P[1, 2] * z) / (P[2, 0] * x + P[2, 1] * y + P[2, 2] * z)
    u = image_point[0]/image_point[2]
    v = image_point[1]/image_point[2]
    return np.array([u, v])


def jacobian_function(input, K, T):
    """
    Use your Jacobian function at any point you want.
    """
    f_jacob = nd.Jacobian(f)
    matrix = f_jacob(input, K, T)
    return matrix


# ===========Test======================================
K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print jacobian_function([1, 2, 3],K,T)

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 07.02.18 18:59
@File    : covariance_mat_sphericalCoordinate.py
@author: Yue Hu
"""
import sys
sys.path.append("..")
import numpy as np
import numdifftools as nd
from scipy.linalg import expm, rq, det, inv,pinv
from vision.camera import Camera
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t
from vision.circular_plane import CircularPlane
import vision.rt_matrix as rt
from math import pi

def covariance_alpha_belt_r(cam,new_objectPoints):
    """
    Covariance matrix of the spherical angles α, β and the distance r

    """

    cam = cam.clone()
    K = cam.K
    objectPoints = np.copy(new_objectPoints)
    imagePoint = np.array(cam.project(objectPoints, False))
    print "imagePoint",imagePoint

    # j_f should be 8*3 for 4 ponits
    # objectPoints = np.array(
    #     [[1, 1, 1, 1],
    #      [1, 1, 1, 1],
    #      [0., 0., 0., 0.],
    #      [1., 1., 1., 1.]]) # TEST
    j_f = jacobian_function([0,0,0],K,objectPoints)

    gaussian_noise_px1 = np.random.normal(0, 4, 10000) + imagePoint[0,0]
    gaussian_noise_py1 = np.random.normal(0, 4, 10000) + imagePoint[1,0]


    gaussian_noise_px2 = np.random.normal(0, 4, 10000) + imagePoint[0,1]
    gaussian_noise_py2 = np.random.normal(0, 4, 10000) + imagePoint[1,1]

    gaussian_noise_px3 = np.random.normal(0, 4, 10000) + imagePoint[0,2]
    gaussian_noise_py3 = np.random.normal(0, 4, 10000) + imagePoint[1,2]

    gaussian_noise_px4 = np.random.normal(0, 4, 10000) + imagePoint[0,3]
    gaussian_noise_py4 = np.random.normal(0, 4, 10000) + imagePoint[1,3]

    cov_mat_p1 = np.cov(gaussian_noise_px1,gaussian_noise_py1)
    cov_mat_p2 = np.cov(gaussian_noise_px2,gaussian_noise_py2)
    cov_mat_p3 = np.cov(gaussian_noise_px3,gaussian_noise_py3)
    cov_mat_p4 = np.cov(gaussian_noise_px4,gaussian_noise_py4)


    block_mat_image4points = np.block([[cov_mat_p1, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
                    [np.zeros((2, 2)), cov_mat_p2, np.zeros((2, 2)), np.zeros((2, 2))],
                    [np.zeros((2, 2)), np.zeros((2, 2)), cov_mat_p3, np.zeros((2, 2))],
                    [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), cov_mat_p4]])
    # print "block_mat_image4points",block_mat_image4points
    # TODO Pinv inv
    # print "j_f:\n",j_f
    cov_mat = inv(np.dot(np.dot(j_f.T, inv(block_mat_image4points)), j_f))
    return cov_mat


def jacobian_function(input, K, obj_point):
    """
    Use your Jacobian function at any point you want.
    """
    f_jacob = nd.Jacobian(f)

    # 2*4(points) equations
    matrix_point1 = f_jacob(input, K, obj_point[:,0])
    matrix_point2 = f_jacob(input, K, obj_point[:,1])
    matrix_point3 = f_jacob(input, K, obj_point[:,2])
    matrix_point4 = f_jacob(input, K, obj_point[:,3])
    jacobian_funs = np.vstack((matrix_point1,matrix_point2,matrix_point3,matrix_point4))
    return jacobian_funs



def f(input, K,obj_point):
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

    # ---------------------------------------
    # world_position = [x,y,z]
    # eye = world_position
    # target = np.array([0, 0, 0])
    # up = np.array([0, 1, 0])
    # zaxis = (target - eye) / np.linalg.norm(target - eye)
    # xaxis = (np.cross(up, zaxis)) / np.linalg.norm(np.cross(up, zaxis))
    # yaxis = np.cross(zaxis, xaxis)

    # TODO need to change R maybe?
    # R = np.array([[xaxis[0], yaxis[0], zaxis[0]],
    #               [xaxis[1], yaxis[1], zaxis[1]],
    #               [xaxis[2], yaxis[2], zaxis[1]],])

    R = np.array([[np.cos(belt)*np.cos(alpha), -np.sin(belt), np.cos(belt)*np.sin(alpha)],
                  [np.sin(belt)*np.cos(alpha), np.cos(belt), np.sin(belt)*np.sin(alpha)],
                  [-np.sin(alpha), 0, np.cos(alpha)],])
    Rx = rt.rotation_matrix([1,0,0],pi)
    R = np.dot(Rx[:3,:3],R) # Z-axis points to origin
    # ---------------------------------------
    # R = np.array([[input[0], input[0], input[0]],
    #               [input[1], input[1], input[1]],
    #               [input[2], input[2], input[1]],]) # Test

    T_R = R
    cam_world = np.array([[x, y, z]]).T
    T_t = np.dot(R, -cam_world)
    # T_t = np.array([[alpha],[belt],[r]]) # Test

    T = np.hstack((T_R,T_t)) # 3*4
    P = np.dot(K, T) # 3*4
    image_point = np.dot(P, obj_point)


    # u = (P[0, 0] * x + P[0, 1] * y + P[0, 2] * z) / (P[2, 0] * x + P[2, 1] * y + P[2, 2] * z)
    # v = (P[1, 0] * x + P[1, 1] * y + P[1, 2] * z) / (P[2, 0] * x + P[2, 1] * y + P[2, 2] * z)
    # u = image_point[0,0]/image_point[0,2]
    # v = image_point[0,1]/image_point[0,2]
    # TODO u v ??
    u = image_point[0,0]
    v = image_point[0,1]
    return np.array([u, v])




# ============================================Test=================================================
# K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# print jacobian_function([1, 2, 3],K,T)

cam = Camera()
cam.set_K(fx = 800,fy = 800,cx = 640/2.,cy = 480/2.)
cam.set_width_heigth(640,480)
cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(180), 0))
cam.set_t(0, 0, 5, 'world')

calc_metrics = False
number_of_points = 4

pl = CircularPlane(origin=np.array([0., 0., 0.]), normal = np.array([0, 0, 1]), radius=0.15, n = 4)
x1  = round(pl.radius*np.cos(np.deg2rad(45)),3)
y1  = round(pl.radius*np.sin(np.deg2rad(45)),3)
objectPoints_square= np.array(
[[ x1, -x1, -x1, x1],
 [ y1, y1, -y1, -y1],
 [ 0., 0., 0., 0.],
 [ 1., 1., 1., 1.]])

new_objectPoints = np.copy(objectPoints_square)
print "new_objectPoints",new_objectPoints
result = covariance_alpha_belt_r(cam,new_objectPoints)
print "covariance_alpha_belt_r : \n",result

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 19.12.17 12:45
@File    : test_brute_force.py
@author: Yue Hu

This class is used to test brute_force.py
"""
import numpy as np
import sys
sys.path.append("..")
from vision.camera import Camera
from vision.plane import Plane
from vision.circular_plane import CircularPlane
from vision.camera_distribution import create_cam_distribution,create_cam_distribution_in_YZ,create_cam_distribution_in_XZ
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t
import brute_force as bf
import display_condNum as dc
import saveMatrixToFile as smtf

def Z_fixed_study_square():
    cams_HeightFixed = []
    cam_1 = Camera()
    cam_1.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    cam_1.set_width_heigth(640, 480)
    cam_1.set_R_axisAngle(1.0, 0.0, 0.0, np.deg2rad(180.0))
    cam_1.set_t(0.1, 0.1, 1, frame='world')
    cams_HeightFixed.append(cam_1)

    cam_2 = Camera()
    cam_2.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    cam_2.set_width_heigth(640, 480)
    cam_2.set_R_axisAngle(1.0, 0.0, 0.0, np.deg2rad(180.0))
    cam_2.set_t(-0.1, 0.1, 1, frame='world')
    # 0.28075725, -0.23558331, 1.31660688
    cams_HeightFixed.append(cam_2)

    cam_3 = Camera()
    cam_3.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    cam_3.set_width_heigth(640, 480)
    cam_3.set_R_axisAngle(1.0, 0.0, 0.0, np.deg2rad(180.0))
    cam_3.set_t(-0.1, -0.1, 1, frame='world')
    # 0.28075725, -0.23558331, 1.31660688
    cams_HeightFixed.append(cam_3)

    cam_4 = Camera()
    cam_4.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    cam_4.set_width_heigth(640, 480)
    cam_4.set_R_axisAngle(1.0, 0.0, 0.0, np.deg2rad(180.0))
    cam_4.set_t(0.1, -0.1, 1, frame='world')
    # 0.28075725, -0.23558331, 1.31660688
    cams_HeightFixed.append(cam_4)
    inputX, inputY, inputZ, input_ippe1_t, input_ippe1_R, input_ippe2_t, input_ippe2_R, input_pnp_t, input_pnp_R, input_transfer_error, display_mat = bf.heightGetCondNum(cams_HeightFixed)

def Z_fixed_study_XY():
    """
    Z fixed, study X Y
    """
    cams_Zfixed = []
    for x in np.linspace(-0.5,0.5,50):
        for y in np.linspace(-0.5,0.5,50):
            cam1 = Camera()
            cam1.set_K(fx = 800,fy = 800,cx = 640/2.,cy = 480/2.)
            cam1.set_width_heigth(640,480)
            # cam1.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
            # cam1.set_t(x, y, 1.3, frame='world')
            cam1.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0.0, 0, 0))
            cam1.set_t(x, y, 1.3, frame='world')
            cam1.look_at([0, 0, 0])
            # 0.28075725, -0.23558331, 1.31660688
            cams_Zfixed.append(cam1)
    #------------For 1 End-------------------------------------------------
    inputX, inputY, inputZ, input_ippe1_t, input_ippe1_R, input_ippe2_t, input_ippe2_R, input_pnp_t, input_pnp_R, input_transfer_error, display_mat,accuracy_mat_new = bf.heightGetCondNum(cams_Zfixed)
    dc.displayError_Zfixed3D(inputX, inputY, input_ippe1_t, input_ippe1_R, input_ippe2_t, input_ippe2_R,
                                 input_pnp_t, input_pnp_R, input_transfer_error)


def XY_fixed_study_Z():
    """
    X Y fixed, study Z
    """
    # cams_XYfixed = []
    # for i in np.linspace(0.5, 2, 100):
    #     cam1 = Camera()
    #     cam1.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    #     cam1.set_width_heigth(640, 480)
    #     cam1.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0.0, 0, 0))
    #     cam1.set_t(0.1, -0.1, i, frame='world')
    #     cam1.look_at([0, 0, 0])
    #     cams_XYfixed.append(cam1)
    angle_begin = 90.0
    angle_end = 90.0
    angle_num = 1
    if angle_num ==  1:
        angle_step = 0
    else:
        angle_step = (angle_end - angle_begin) / (angle_num - 1)
    theta_params = (angle_begin,angle_end,angle_num)

    r_begin = 0.0
    r_end = 2.0
    r_num = 21
    r_step = (r_end - r_begin) / (r_num - 1)
    r_params = (r_begin,r_end,r_num)
    cams,accuracy_mat = create_cam_distribution_in_YZ(cam=None, plane_size=(0.3, 0.3), theta_params=theta_params, r_params=r_params,
                                  plot=False)
    inputX, inputY, inputZ, input_ippe1_t, input_ippe1_R, input_ippe2_t, input_ippe2_R, input_pnp_t, input_pnp_R, input_transfer_error, display_mat,accuracy_mat_new = bf.heightGetCondNum(cams,accuracy_mat,r_step,angle_step)
    dc.displayError_XYfixed3D(inputZ,input_ippe1_t,input_ippe1_R,input_ippe2_t,input_ippe2_R,input_pnp_t,input_pnp_R,input_transfer_error)
    # dc.displayCondNumDistribution(display_mat)

def cam_distribution_study():
    """
    cam Distribution in 3D, show the cond num distribution for each cam position
    """
    number_of_points = 4
    # TODO Change the radius of circular plane
    pl = CircularPlane(origin=np.array([0., 0., 0.]), normal=np.array([0, 0, 1]), radius=0.15, n=4)
    pl.random(n=number_of_points, r=0.01, min_sep=0.01)
    plane_size = (0.3, 0.3)
    cam = Camera()
    cam.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    cam.set_width_heigth(640, 480)
    # TODO cam distribution position PARAMETER CHANGE!!!
    # cams = create_cam_distribution(cam, plane_size,
    #                                theta_params=(0, 360, 30), phi_params=(0, 70, 10),
    #                                r_params=(0.2, 2.0, 20), plot=False)
    angle_begin = 10.0#0.0
    angle_end = 170.0#180.0
    angle_num = 171 #37 TODO need to set
    # angle_step = (angle_end - angle_begin) / (angle_num - 1)
    theta_params = (angle_begin,angle_end,angle_num)

    r_begin = 0.1 # we cant set this as 0.0 !!! avoid float error!!!
    r_end = 3.0
    r_num = 30 #30 TODO need to set
    # r_step = (r_end - r_begin) / (r_num - 1)
    r_params = (r_begin,r_end,r_num)

    cams,accuracy_mat = create_cam_distribution_in_YZ(cam=None, plane_size=(0.3, 0.3), theta_params=theta_params, r_params=r_params,
                                  plot=False)
    # TODO
    accuracy_mat = np.zeros([30, 60])# define the new acc_mat as 30 x 30 ,which means 3m x 3m area in real world and each cel
    inputX, inputY, inputZ, input_ippe1_t, input_ippe1_R, input_ippe2_t, input_ippe2_R, input_pnp_t, input_pnp_R, input_transfer_error, display_mat, accuracy_mat_new = bf.heightGetCondNum(
        cams, accuracy_mat, theta_params, r_params)
    # print "accuracy_mat distribution:\n",accuracy_mat_new
    print "----------------Start to show:-------------------"
    # print "accuracy_mat_new\n",accuracy_mat_new
    # save accuracy matrix in file
    print "-----accuracy_mat_new---------\n",accuracy_mat_new
    smtf.saveMatToFile(accuracy_mat_new)
    # plot condition number distribution
    dc.displayCondNumDistribution(display_mat)
    dc.displayError_YZ_plane_2D(inputY, inputZ, input_ippe1_t, input_ippe1_R, input_ippe2_t, input_ippe2_R, input_pnp_t, input_pnp_R)
    dc.displayError_YZ_plane_3D(inputY, inputZ, input_ippe1_t, input_ippe1_R, input_ippe2_t, input_ippe2_R, input_pnp_t,
                             input_pnp_R, input_transfer_error)
    print "----------------End-----------------------------"

# ======================================Main ========================================================
if __name__ == '__main__':
# ---------------------------Test Method-------------------------
    # XY_fixed_study_Z()
    # Z_fixed_study_XY()
    cam_distribution_study()
# =================================Main End=================================================



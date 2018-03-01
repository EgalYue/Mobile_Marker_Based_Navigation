#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 05.01.18 12:45
@File    : best_angle_test.py
@author: Yue Hu

Not used yet!!!
"""
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
# import error_functions as ef
# from ippe import homo2d
# from homographyHarker.homographyHarker import homographyHarker as hh
# from solve_ippe import pose_ippe_both, pose_ippe_best
# from solve_pnp import pose_pnp
# import cv2
# import matplotlib.pyplot as plt
import hT_gradient as gd
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t

calc_metrics = False
number_of_points = 4

## Define a Display plane with random initial points
# TODO Change the radius of circular plane
pl = CircularPlane(origin=np.array([0., 0., 0.]), normal = np.array([0, 0, 1]), radius=0.15, n = 4)
pl.random(n =number_of_points, r = 0.01, min_sep = 0.01)


## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx = 800,fy = 800,cx = 640/2.,cy = 480/2.)
cam.set_width_heigth(640,480)

#Now we define a distribution of cameras on the space based on this plane
#An optional paremeter is de possible deviation from uniform points
plane_size = (0.3,0.3)
new_objectPoints = pl.get_points()

## CREATE A SET OF IMAGE POINTS FOR VALIDATION OF THE HOMOGRAPHY ESTIMATION
# This will create a grid of 16 points of size = (0.3,0.3) meters
validation_plane =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.3,0.3), n = (4,4))
validation_plane.uniform()

## we create the gradient for the point distribution
normalize= False


#4 points: An ideal square
x1  = round(pl.radius*np.cos(np.deg2rad(45)),3)
y1  = round(pl.radius*np.sin(np.deg2rad(45)),3)
objectPoints_square= np.array(
[[ x1, -x1, -x1, x1],
 [ y1, y1, -y1, -y1],
 [ 0., 0., 0., 0.],
 [ 1., 1., 1., 1.]])

new_objectPoints = np.copy(objectPoints_square)

# TODO CHANGE tx,ty,tz,
fig1 = plt.figure('Image points')
ax_image = fig1.add_subplot(211)


def angleSim(cam, t):
    # ----------------------------Factor: Angle----------------------------------------------------
    # set camera again
    # TODO
    cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(180), 0))
    # cam.set_t( 0.28075725, -0.23558331, 1.31660688, frame='world')
    cam.set_t(t[0], t[1], t[2], frame='world')
    # cam.look_at([0,0,0])

    imagePoints_des = []
    mat_cond_list = []
    angles_x = []
    angles_y = []
    fovx, fovy = cam.fov()
    display_mat = np.array([[0],[0],[0]])
    #  No rotation

    objectPoints = np.copy(new_objectPoints)
    imagePoint = np.array(cam.project(objectPoints, False))
    # Determine whether image is valid
    if ((imagePoint[0, :] < cam.img_width) & (imagePoint[0, :] > 0) & (imagePoint[1, :] < cam.img_height) & (
        imagePoint[1, :] > 0)).all():

        angles_x.append(0)
        angles_y.append(0)

        imagePoints_des.append(np.copy(imagePoint))
        input_list = gd.extract_objectpoints_vars(objectPoints)
        input_list.append(np.array(cam.K))
        input_list.append(np.array(cam.R))
        input_list.append(cam.t[0, 3])
        input_list.append(cam.t[1, 3])
        input_list.append(cam.t[2, 3])
        mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
        mat_cond_list.append(mat_cond)


    # rotation around x axis
    for anglex in  np.linspace(-fovy,fovy,100):
       #  rotation around y axis

        for angley in np.linspace(-fovx,fovx,100):
            # Rotation of cam

            cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(180), 0))

            anglex = 0
            # angley = 0
            cam.rotate_x(np.deg2rad(anglex))
            cam.rotate_y(np.deg2rad(angley))
            imagePoint = np.array(cam.project(objectPoints, False))
            # print "cam.project!!!!",imagePoint
            # Determine whether image is valid
            if ((imagePoint[0,:]< cam.img_width) & (imagePoint[0,:]>0) & (imagePoint[1,:]< cam.img_height) & (imagePoint[1,:]>0)).all():

                angles_x.append(anglex)
                angles_y.append(angley)
                print "anglex=", anglex
                print "angley=", angley
                imagePoints_des.append(np.copy(imagePoint))
                input_list = gd.extract_objectpoints_vars(objectPoints)
                input_list.append(np.array(cam.K))
                input_list.append(np.array(cam.R))
                input_list.append(cam.t[0,3])
                input_list.append(cam.t[1,3])
                input_list.append(cam.t[2,3])
                mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
                mat_cond_list.append(mat_cond)
                print "--cond num--", mat_cond
                # write the data(anglex + angley + cond num) to  file, its a 3*1 array
                display_array = np.eye(3,1,dtype=float)
                display_array[0, 0] = np.copy(anglex)
                display_array[1, 0] = np.copy(angley)
                display_array[2, 0] = np.copy(mat_cond)
                display_mat = np.hstack((display_mat,display_array))

                # ax_image.cla()
                # plt.sca(ax_image)
                # plt.ion()
                # ax_image.plot(imagePoint[0], imagePoint[1], '.', color='blue', )
                # ax_image.set_xlim(0, 1280)
                # ax_image.set_ylim(0, 960)
                # ax_image.invert_yaxis()
                # ax_image.set_title('Image Points')
                # plt.show()
                # plt.pause(0.01)

    print "--cond num min--", min(mat_cond_list)
    print "--cam angle x Best--", angles_x[mat_cond_list.index(min(mat_cond_list))]
    print "--cam angle y Best--", angles_y[mat_cond_list.index(min(mat_cond_list))]
    print "--cond num size--", len(mat_cond_list)
    print "--imagePoints_des size--", len(imagePoints_des)
    print "--angles_x size--", len(angles_x)
    print "--angles_y size--", len(angles_y)
    print "--display_mat size--",display_mat.shape

    plt.sca(ax_image)

    ax_image.plot(imagePoints_des[mat_cond_list.index(min(mat_cond_list))][0], imagePoints_des[mat_cond_list.index(min(mat_cond_list))][1], '.', color='blue', )
    # ax_image.plot(imagePoints_des[0][0], imagePoints_des[0][1], '.', color='blue', )

    ax_image.set_xlim(0, 1280)
    ax_image.set_ylim(0, 960)
    ax_image.invert_yaxis()
    ax_image.set_title('Image Points')
    plt.show()
    # plt.pause(100)
    #-----------------------Write data into file-----------------------------------
    # sio.savemat('angle_data3.mat', {'data': display_mat})
    print "start to show "
    import display_condNum as dc
    dc.anglePlot(display_mat)
    print "finish!!!"
# ------------------------------------------------------------------------------------------------


def angleXSim(cam,t):
    # ----------------------------Factor: Angle----------------------------------------------------
    # set camera again
    # TODO
    cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(180), 0))
    # cam.set_t( 0.28075725, -0.23558331, 1.31660688, frame='world')
    cam.set_t( t[0], t[1], t[2], frame='world')

    # cam.look_at([0,0,0])

    imagePoints_des = []
    mat_cond_list = []
    angles_x = []
    angles_y = []
    fovx, fovy = cam.fov()
    display_mat = np.array([[0],[0],[0]])
    #  No rotation

    objectPoints = np.copy(new_objectPoints)
    imagePoint = np.array(cam.project(objectPoints, False))
    # Determine whether image is valid
    if ((imagePoint[0, :] < cam.img_width) & (imagePoint[0, :] > 0) & (imagePoint[1, :] < cam.img_height) & (
        imagePoint[1, :] > 0)).all():

        angles_x.append(0)
        angles_y.append(0)

        imagePoints_des.append(np.copy(imagePoint))
        input_list = gd.extract_objectpoints_vars(objectPoints)
        input_list.append(np.array(cam.K))
        input_list.append(np.array(cam.R))
        input_list.append(cam.t[0, 3])
        input_list.append(cam.t[1, 3])
        input_list.append(cam.t[2, 3])
        mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
        mat_cond_list.append(mat_cond)


    # rotation around x axis
    for anglex in  np.linspace(-fovy,fovy,100):
            # Rotation of cam
            cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(180), 0))
            angley = 0
            cam.rotate_x(np.deg2rad(anglex))
            cam.rotate_y(np.deg2rad(angley))
            imagePoint = np.array(cam.project(objectPoints, False))
            # print "cam.project!!!!",imagePoint
            # Determine whether image is valid
            if ((imagePoint[0,:]< cam.img_width) & (imagePoint[0,:]>0) & (imagePoint[1,:]< cam.img_height) & (imagePoint[1,:]>0)).all():

                angles_x.append(anglex)
                angles_y.append(angley)
                print "anglex=", anglex
                print "angley=", angley
                imagePoints_des.append(np.copy(imagePoint))
                input_list = gd.extract_objectpoints_vars(objectPoints)
                input_list.append(np.array(cam.K))
                input_list.append(np.array(cam.R))
                input_list.append(cam.t[0,3])
                input_list.append(cam.t[1,3])
                input_list.append(cam.t[2,3])
                mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
                mat_cond_list.append(mat_cond)
                print "--cond num--", mat_cond
                # write the data(anglex + angley + cond num) to  file, its a 3*1 array
                display_array = np.eye(3,1,dtype=float)
                display_array[0, 0] = np.copy(anglex)
                display_array[1, 0] = np.copy(angley)
                display_array[2, 0] = np.copy(mat_cond)
                display_mat = np.hstack((display_mat,display_array))

                # ax_image.cla()
                # plt.sca(ax_image)
                # plt.ion()
                # ax_image.plot(imagePoint[0], imagePoint[1], '.', color='blue', )
                # ax_image.set_xlim(0, 1280)
                # ax_image.set_ylim(0, 960)
                # ax_image.invert_yaxis()
                # ax_image.set_title('Image Points')
                # plt.show()
                # plt.pause(0.01)

    print "--cond num min--", min(mat_cond_list)
    print "--cam angle x Best--", angles_x[mat_cond_list.index(min(mat_cond_list))]
    print "--cam angle y Best--", angles_y[mat_cond_list.index(min(mat_cond_list))]
    print "--cond num size--", len(mat_cond_list)
    print "--imagePoints_des size--", len(imagePoints_des)
    print "--angles_x size--", len(angles_x)
    print "--angles_y size--", len(angles_y)
    print "--display_mat size--",display_mat.shape

    plt.sca(ax_image)

    ax_image.plot(imagePoints_des[mat_cond_list.index(min(mat_cond_list))][0], imagePoints_des[mat_cond_list.index(min(mat_cond_list))][1], '.', color='blue', )
    # ax_image.plot(imagePoints_des[0][0], imagePoints_des[0][1], '.', color='blue', )

    ax_image.set_xlim(0, 1280)
    ax_image.set_ylim(0, 960)
    ax_image.invert_yaxis()
    ax_image.set_title('Image Points')
    plt.show()
    # plt.pause(100)
    #-----------------------Write data into file-----------------------------------
    # sio.savemat('angle_data3.mat', {'data': display_mat})
    print "start to show "
    import display_condNum as dc
    dc.anglePlot(display_mat)
    print "finish!!!"
# ------------------------------------------------------------------------------------------------
def angleYSim(cam,t):
    # ----------------------------Factor: Angle----------------------------------------------------
    # set camera again
    # TODO
    cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(180), 0))
    # cam.set_t( 0.28075725, -0.23558331, 1.31660688, frame='world')
    cam.set_t( t[0], t[1], t[2], frame='world')

    # cam.look_at([0,0,0])

    imagePoints_des = []
    mat_cond_list = []
    angles_x = []
    angles_y = []
    fovx, fovy = cam.fov()
    display_mat = np.array([[0],[0],[0]])
    #  No rotation

    objectPoints = np.copy(new_objectPoints)
    imagePoint = np.array(cam.project(objectPoints, False))
    # Determine whether image is valid
    if ((imagePoint[0, :] < cam.img_width) & (imagePoint[0, :] > 0) & (imagePoint[1, :] < cam.img_height) & (
        imagePoint[1, :] > 0)).all():

        angles_x.append(0)
        angles_y.append(0)

        imagePoints_des.append(np.copy(imagePoint))
        input_list = gd.extract_objectpoints_vars(objectPoints)
        input_list.append(np.array(cam.K))
        input_list.append(np.array(cam.R))
        input_list.append(cam.t[0, 3])
        input_list.append(cam.t[1, 3])
        input_list.append(cam.t[2, 3])
        mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
        mat_cond_list.append(mat_cond)



       #  rotation around y axis
    for angley in np.linspace(-fovx,fovx,100):
            # Rotation of cam
            cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(180), 0))
            anglex = 0
            cam.rotate_x(np.deg2rad(anglex))
            cam.rotate_y(np.deg2rad(angley))
            imagePoint = np.array(cam.project(objectPoints, False))
            # print "cam.project!!!!",imagePoint
            # Determine whether image is valid
            if ((imagePoint[0,:]< cam.img_width) & (imagePoint[0,:]>0) & (imagePoint[1,:]< cam.img_height) & (imagePoint[1,:]>0)).all():

                angles_x.append(anglex)
                angles_y.append(angley)
                print "anglex=", anglex
                print "angley=", angley
                imagePoints_des.append(np.copy(imagePoint))
                input_list = gd.extract_objectpoints_vars(objectPoints)
                input_list.append(np.array(cam.K))
                input_list.append(np.array(cam.R))
                input_list.append(cam.t[0,3])
                input_list.append(cam.t[1,3])
                input_list.append(cam.t[2,3])
                mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
                mat_cond_list.append(mat_cond)
                print "--cond num--", mat_cond
                # write the data(anglex + angley + cond num) to  file, its a 3*1 array
                display_array = np.eye(3,1,dtype=float)
                display_array[0, 0] = np.copy(anglex)
                display_array[1, 0] = np.copy(angley)
                display_array[2, 0] = np.copy(mat_cond)
                display_mat = np.hstack((display_mat,display_array))

                ax_image.cla()
                plt.sca(ax_image)
                plt.ion()
                ax_image.plot(imagePoint[0], imagePoint[1], '.', color='blue', )
                ax_image.set_xlim(0, 1280)
                ax_image.set_ylim(0, 960)
                ax_image.invert_yaxis()
                ax_image.set_title('Image Points')
                plt.show()
                plt.pause(0.01)

    print "--cond num min--", min(mat_cond_list)
    print "--cam angle x Best--", angles_x[mat_cond_list.index(min(mat_cond_list))]
    print "--cam angle y Best--", angles_y[mat_cond_list.index(min(mat_cond_list))]
    print "--cond num size--", len(mat_cond_list)
    print "--imagePoints_des size--", len(imagePoints_des)
    print "--angles_x size--", len(angles_x)
    print "--angles_y size--", len(angles_y)
    print "--display_mat size--",display_mat.shape

    plt.sca(ax_image)

    ax_image.plot(imagePoints_des[mat_cond_list.index(min(mat_cond_list))][0], imagePoints_des[mat_cond_list.index(min(mat_cond_list))][1], '.', color='blue', )
    # ax_image.plot(imagePoints_des[0][0], imagePoints_des[0][1], '.', color='blue', )

    ax_image.set_xlim(0, 1280)
    ax_image.set_ylim(0, 960)
    ax_image.invert_yaxis()
    ax_image.set_title('Image Points')
    plt.show()
    # plt.pause(100)
    #-----------------------Write data into file-----------------------------------
    # sio.savemat('angle_data3.mat', {'data': display_mat})
    print "start to show "
    import display_condNum as dc
    dc.anglePlot(display_mat)
    print "finish!!!"
# ------------------------------------------------------------------------------------------------

def angleGetCondNum(cam,t,new_objectPoints,anglex,angley):

    # set camera again
    cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(180), 0))
    cam.set_t( t[0], t[1], t[2], frame='world')
    fovx, fovy = cam.fov()
    objectPoints = np.copy(new_objectPoints)
    # Determine whether angle is valid
    if((anglex <= fovy) & (anglex >= -fovy) & (angley <= fovx) & (angley >= -fovx)):

        # rotation around x axis and rotation around y axis
        cam.rotate_x(np.deg2rad(anglex))
        cam.rotate_y(np.deg2rad(angley))
        imagePoint = np.array(cam.project(objectPoints, False))
        # check the image points valid or not!
        if ((imagePoint[0, :] < cam.img_width) & (imagePoint[0, :] > 0) & (imagePoint[1, :] < cam.img_height) & (
            imagePoint[1, :] > 0)).all():
            print "anglex=", anglex
            print "angley=", angley
            input_list = gd.extract_objectpoints_vars(objectPoints)
            input_list.append(np.array(cam.K))
            input_list.append(np.array(cam.R))
            input_list.append(cam.t[0, 3])
            input_list.append(cam.t[1, 3])
            input_list.append(cam.t[2, 3])
            mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
            return mat_cond
            print "--cond num--", mat_cond

        else:
            return -1
    else:
        return -1
# ------------------------------------------------------------------------------------------------

camPosition = np.array([0.26517749, -0.20158269, 1.19660807])
# angleXSim(cam,camPosition)
angleYSim(cam,camPosition)
# angleSim(cam, camPosition)
# print angleGetCondNum(cam,camPosition,new_objectPoints,0,0)
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 05.01.18 12:45
@File    : best_position_test.py
@author: Yue Hu
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
#
cams = []
for i in np.linspace(0,2,100):

    cam1 = Camera()
    cam1.set_K(fx = 800,fy = 800,cx = 640/2.,cy = 480/2.)
    cam1.set_width_heigth(640,480)

    cam1.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
    cam1.set_t(-i,-i,1, frame='world')
    cams.append(cam1)

print len(cams)

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


# ------------------------All cams point straight down: Test Height factor---------------------------------------
# TODO cam distribution position PARAMETER CHANGE!!!
# cams = create_cam_distribution(cam, plane_size,
#                                theta_params = (0,360,10), phi_params =  (0,70,10),
#                                r_params = (0.1,2.0,10), plot=False)
#
mat_cond_list = []
imagePoints_des = []
best_imgaePoint = np.copy(objectPoints_square)
display_mat = np.array([[0],[0],[0],[0]])

for cam in cams:
    objectPoints = np.copy(new_objectPoints)
    imagePoints = np.array(cam.project(objectPoints, False))
    if ((imagePoints[0,:]< cam.img_width) & (imagePoints[0,:]>0) & (imagePoints[1,:]< cam.img_height) & (imagePoints[1,:]>0)).all():
            imagePoints_des.append(imagePoints)
            print '--cam position==',cam.get_world_position()
            print "imagePoints==",imagePoints
            input_list = gd.extract_objectpoints_vars(objectPoints)
            input_list.append(np.array(cam.K))
            input_list.append(np.array(cam.R))
            input_list.append(cam.t[0,3])
            input_list.append(cam.t[1,3])
            input_list.append(cam.t[2,3])
            mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
            mat_cond_list.append(mat_cond)
            print "mat_cond==",mat_cond

            # write the data(cam position + cond num) to  file, its a 4*1 array
            cam_position = cam.get_world_position()
            display_array = np.eye(4,1,dtype=float)
            display_array[0:3, 0] = np.copy(cam_position[0:3])
            display_array[3] = np.copy(mat_cond)
            display_mat = np.hstack((display_mat,display_array))


    ax_image.cla()
    plt.sca(ax_image)
    plt.ion()
    ax_image.plot(imagePoints[0], imagePoints[1], '.', color='blue', )
    ax_image.set_xlim(0, 1280)
    ax_image.set_ylim(0, 960)
    ax_image.invert_yaxis()
    ax_image.set_title('Image Points')
    plt.show()
    plt.pause(0.01)

#-----------------------Write data into file-----------------------------------
# sio.savemat('testpython.mat', {'data': display_mat})

print "--cond num min--", min(mat_cond_list)
print "--cam angle Best--", cams[mat_cond_list.index(min(mat_cond_list))].R
print "--cam position Best--", cams[mat_cond_list.index(min(mat_cond_list))].get_world_position()
print "--best image points--", imagePoints_des[mat_cond_list.index(min(mat_cond_list))]
print "--cond num size--", len(mat_cond_list)
print "--imagePoints_des size--", len(imagePoints_des)

# # -----------------------Draw the image points-----------------------------------------------------------------------
fig1 = plt.figure('Image points')
ax_image_best = fig1.add_subplot(212)
plt.sca(ax_image_best)
ax_image_best.plot(imagePoints_des[mat_cond_list.index(min(mat_cond_list))][0], imagePoints_des[mat_cond_list.index(min(mat_cond_list))][1], '.', color='blue', )
ax_image_best.set_xlim(0, 1280)
ax_image_best.set_ylim(0, 960)
ax_image_best.invert_yaxis()
ax_image_best.set_title('Image Points')
plt.show()
plt.pause(100)


# ----------------------------Factor: Angle----------------------------------------------------
# set camera again
# TODO
# cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(180), 0))
# cam.set_t( -0.3, 0.05, 1.5, frame='world')
# # cam.look_at([0,0,0])
#
# imagePoints_des = []
# mat_cond_list = []
# angles_x = []
# angles_y = []
# fovx, fovy = cam.fov()
# #  No rotation
#
# objectPoints = np.copy(new_objectPoints)
# imagePoint = np.array(cam.project(objectPoints, False))
# print imagePoint
# # Determine whether image is valid
# if ((imagePoint[0, :] < cam.img_width) & (imagePoint[0, :] > 0) & (imagePoint[1, :] < cam.img_height) & (
#     imagePoint[1, :] > 0)).all():
#
#     angles_x.append(0)
#     angles_y.append(0)
#
#     imagePoints_des.append(np.copy(imagePoint))
#     input_list = gd.extract_objectpoints_vars(objectPoints)
#     input_list.append(np.array(cam.K))
#     input_list.append(np.array(cam.R))
#     input_list.append(cam.t[0, 3])
#     input_list.append(cam.t[1, 3])
#     input_list.append(cam.t[2, 3])
#     mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
#     mat_cond_list.append(mat_cond)
#
# rotation around x axis
# for anglex in  np.linspace(-fovy,fovy,100):
#    #  rotation around y axis
#    for angley in np.linspace(-fovx,fovx,100 ):
#         # Rotation of cam
#         cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0, np.deg2rad(180), 0))
#         cam.rotate_x(np.deg2rad(anglex))
#         cam.rotate_y(np.deg2rad(angley))
#
#         objectPoints = np.copy(new_objectPoints)
#         imagePoint = np.array(cam.project(objectPoints, False))
#         # Determine whether image is valid
#         if ((imagePoint[0,:]< cam.img_width) & (imagePoint[0,:]>0) & (imagePoint[1,:]< cam.img_height) & (imagePoint[1,:]>0)).all():
#
#             angles_x.append(anglex)
#             angles_y.append(angley)
#
#             imagePoints_des.append(np.copy(imagePoint))
#             input_list = gd.extract_objectpoints_vars(objectPoints)
#             input_list.append(np.array(cam.K))
#             input_list.append(np.array(cam.R))
#             input_list.append(cam.t[0,3])
#             input_list.append(cam.t[1,3])
#             input_list.append(cam.t[2,3])
#             mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
#             mat_cond_list.append(mat_cond)
#             print "--cond num--", mat_cond
#             print "--cam R after rotation--", cam.R

# print "--cond num min--", min(mat_cond_list)
# print "--cam angle x Best--", angles_x[mat_cond_list.index(min(mat_cond_list))]
# print "--cam angle y Best--", angles_y[mat_cond_list.index(min(mat_cond_list))]
# print "--cond num size--", len(mat_cond_list)
# print "--imagePoints_des size--", len(imagePoints_des)
# print "--angles_x size--", len(angles_x)
# print "--angles_y size--", len(angles_y)
#
# plt.sca(ax_image)
# ax_image.plot(imagePoints_des[mat_cond_list.index(min(mat_cond_list))][0], imagePoints_des[mat_cond_list.index(min(mat_cond_list))][1], '.', color='blue', )
# ax_image.set_xlim(0, 1280)
# ax_image.set_ylim(0, 960)
# ax_image.invert_yaxis()
# ax_image.set_title('Image Points')
# plt.show()
# plt.pause(100)
# ------------------------------------------------------------------------------------------------

# ------------------------------Factor: Look at origin, Zc not changed-----------------------------
# tx_max = 10
# x_list = []
# mat_cond_list = []
# for tx in np.linspace(-10,10,10000):
#     x_list.append(tx)
#     # set camera
#     cam.set_t(tx,0.0,0.303, frame='world')
#     cam.look_at([0, 0, 0])
#     objectPoints = np.copy(new_objectPoints)
#     input_list = gd.extract_objectpoints_vars(objectPoints)
#     input_list.append(np.array(cam.K))
#     # new R
#     print "cam R"
#     print cam.R
#     input_list.append(np.array(cam.R))
#     input_list.append(cam.t[0,3])
#     input_list.append(cam.t[1,3])
#     input_list.append(cam.t[2,3])
#     mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
#     mat_cond_list.append(mat_cond)
#     print "--cam tx--", tx
#     print "--cond num--", mat_cond
#
# print "--cond num min--", min(mat_cond_list)
# print "--cam tx Best--", x_list[mat_cond_list.index(min(mat_cond_list))]
# ------------------------------------------------------------------------------------------------

# ------------------------------Factor: Look at origin, Zc not changed-----------------------------
# ty_max = 10
# y_list = []
# mat_cond_list = []
# for ty in np.linspace(-10,10,10000):
#     y_list.append(ty)
#     # set camera
#     cam.set_t(0.0,ty,0.303, frame='world')
#     cam.look_at([0, 0, 0])
#     objectPoints = np.copy(new_objectPoints)
#     input_list = gd.extract_objectpoints_vars(objectPoints)
#     input_list.append(np.array(cam.K))
#     # new R
#     print "cam R"
#     print cam.R
#     input_list.append(np.array(cam.R))
#     input_list.append(cam.t[0,3])
#     input_list.append(cam.t[1,3])
#     input_list.append(cam.t[2,3])
#     mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize=False)
#     mat_cond_list.append(mat_cond)
#     print "--cam ty--", ty
#     print "--cond num--", mat_cond
#
# print "--cond num min--", min(mat_cond_list)
# print "--cam tx Best--", y_list[mat_cond_list.index(min(mat_cond_list))]
# ------------------------------------------------------------------------------------------------
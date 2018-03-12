# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:38:35 2017

@author: racuna
"""
#======================================================================
import sys
sys.path.append("..")
#======================================================================
import matplotlib.pyplot as plt
import autograd.numpy as np
# from mayavi import mlab #TODO comment because of from mayavi import mlab
from numpy import random, cos, sin, sqrt, pi, linspace, deg2rad, meshgrid
from mpl_toolkits.mplot3d import Axes3D
from camera import Camera
from plane import Plane
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t


def uniform_sphere(theta_params = (0,360,10), phi_params = (0,90,10), r = 1., plot = False):
  """n points distributed evenly on the surface of a unit sphere
  theta_params: tuple (min = 0,max = 360, N divisions = 10)
  phi_params: tuple (min =0,max =90, N divisions = 10)
  r: radius of the sphere
  n_theta: number of points in theta
  n_phi: number of points in phi

  """
  space_theta = linspace(deg2rad(theta_params[0]), deg2rad(theta_params[1]), theta_params[2])
  space_phi = linspace(deg2rad(phi_params[0]), deg2rad(phi_params[1]), phi_params[2])
  theta, phi = meshgrid(space_theta,space_phi )

  x = r*cos(theta)*sin(phi)
  y = r*sin(theta)*sin(phi)
  z = r*cos(phi)
  if plot:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()

  return x, y, z


def uniform_halfCircle_in_XZ(theta_params = (0,180,10), r = 1., plot = False):
  """
  n points distributed evenly on the half circle in XZ plane
  theta_params: tuple (min = 0,max = 360, N divisions = 10)
  r: radius of the half circle
  """
  theta = linspace(deg2rad(theta_params[0]), deg2rad(theta_params[1]), theta_params[2])

  x = r*cos(theta)
  y = 0 * theta # 0
  z = r*sin(theta)

  if plot:
    fig, ax = plt.subplots()
    ax.scatter(x,z)
    plt.show()

  return x,y,z

def uniform_halfCircle_in_YZ(theta_params = (0,180,10), r = 1., plot = False):
  """
  n points distributed evenly on the half circle in XZ plane
  theta_params: tuple (min = 0,max = 360, N divisions = 10)
  r: radius of the half circle
  """
  theta = linspace(deg2rad(theta_params[0]), deg2rad(theta_params[1]), theta_params[2])
  x = 0 * theta # 0
  y = r*cos(theta)
  z = r*sin(theta)

  if plot:
    fig, ax = plt.subplots()
    ax.scatter(y,z)
    plt.show()

  return x,y,z

  
def plot3D_cam(cam, axis_scale = 0.2):
    
    #Coordinate Frame of real camera
    #Camera axis
    cam_axis_x = np.array([1,0,0,1]).T
    cam_axis_y = np.array([0,1,0,1]).T
    cam_axis_z = np.array([0,0,1,1]).T
    # TODO This place should use cam.R not cam.R.T
    cam_axis_x = np.dot(cam.R, cam_axis_x)
    cam_axis_y = np.dot(cam.R, cam_axis_y)
    cam_axis_z = np.dot(cam.R, cam_axis_z)

    cam_world = cam.get_world_position()

    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_x[0], cam_axis_x[1], cam_axis_x[2], line_width=3, scale_factor=axis_scale, color=(1-axis_scale,0,0))
    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_y[0], cam_axis_y[1], cam_axis_y[2], line_width=3, scale_factor=axis_scale, color=(0,1-axis_scale,0))
    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_z[0], cam_axis_z[1], cam_axis_z[2], line_width=3, scale_factor=axis_scale, color=(0,0,1-axis_scale))


def plot3D(cams, planes):
    #mlab.figure(figure=None, bgcolor=(0.1,0.5,0.5), fgcolor=None, engine=None, size=(400, 350))
    axis_scale = 0.05
    for cam in cams:
        plot3D_cam(cam, axis_scale)
        #axis_scale = axis_scale - 0.1

    for plane in planes:
        #Plot plane points in 3D
        plane_points = plane.get_points()
        mlab.points3d(plane_points[0], plane_points[1], plane_points[2], scale_factor=0.05, color = plane.get_color())
        mlab.points3d(plane_points[0,0], plane_points[1,0], plane_points[2,0], scale_factor=0.05, color = (0.,0.,1.))

    mlab.show()

def create_cam_distribution(cam = None, plane_size = (0.3,0.3), theta_params = (0,360,10), phi_params =  (0,70,5), r_params = (0.25,1.0,4), plot=False):
    if cam == None:
        # Create an initial camera on the center of the world
        cam = Camera()
        f = 800
        cam.set_K(fx=f, fy=f, cx=320, cy=240)  # Camera Matrix
        cam.img_width = 320 * 2
        cam.img_height = 240 * 2

    # we create a default plane with 4 points with a side lenght of w (meters)
    plane = Plane(origin=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), size=plane_size, n=(2, 2))
    # We extend the size of this plane to account for the deviation from a uniform pattern
    # plane.size = (plane.size[0] + deviation, plane.size[1] + deviation)

    d_space = np.linspace(r_params[0], r_params[1], r_params[2])
    t_list = []
    for d in d_space:
        xx, yy, zz = uniform_sphere(theta_params, phi_params, d, False)
        sphere_points = np.array([xx.ravel(), yy.ravel(), zz.ravel()], dtype=np.float32)
        t_list.append(sphere_points)
    t_space = np.hstack(t_list)

    cams = []
    for t in t_space.T:
        cam = cam.clone()
        cam.set_t(-t[0], -t[1], -t[2])
        cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0.0, 0, 0))
        cam.look_at([0, 0, 0])

        # cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0,deg2rad(180),0))
        # cam.set_t(t[0], t[1],t[2],'world')

        plane.set_origin(np.array([0, 0, 0]))
        plane.uniform()
        objectPoints = plane.get_points()
        imagePoints = cam.project(objectPoints)

        # if plot:
        #  cam.plot_image(imagePoints)
        if ((imagePoints[0, :] < cam.img_width) & (imagePoints[0, :] > 0)).all():
            if ((imagePoints[1, :] < cam.img_height) & (imagePoints[1, :] > 0)).all():
                cams.append(cam)

    if plot:
        planes = []
        plane.uniform()
        planes.append(plane)
        plot3D(cams, planes)

    return cams


def create_cam_distribution_in_XZ(cam=None, plane_size=(0.3, 0.3), theta_params=(0, 180, 10),r_params=(0.25, 1.0, 4), plot=False):
    """
    cam distritubution in XZ plane
    :param cam:
    :param plane_size:
    :param theta_params:
    :param phi_params:
    :param r_params:
    :param plot:
    :return:
    """
    if cam == None:
        # Create an initial camera on the center of the world
        cam = Camera()
        f = 800
        cam.set_K(fx=f, fy=f, cx=320, cy=240)  # Camera Matrix
        cam.img_width = 320 * 2
        cam.img_height = 240 * 2

    # we create a default plane with 4 points with a side lenght of w (meters)
    plane = Plane(origin=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), size=plane_size, n=(2, 2))
    # We extend the size of this plane to account for the deviation from a uniform pattern
    # plane.size = (plane.size[0] + deviation, plane.size[1] + deviation)

    d_space = np.linspace(r_params[0], r_params[1], r_params[2])
    t_list = []
    for d in d_space:
        xx,yy,zz = uniform_halfCircle_in_XZ(theta_params, d, False) # XZ plane
        sphere_points = np.array([xx.ravel(), yy.ravel(), zz.ravel()], dtype=np.float32)
        t_list.append(sphere_points)
    t_space = np.hstack(t_list)

    cams = []
    for t in t_space.T:
        cam = cam.clone()
        cam.set_t(-t[0], -t[1], -t[2])
        cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0.0, 0, 0))
        cam.look_at([0, 0, 0])

        plane.set_origin(np.array([0, 0, 0]))
        plane.uniform()
        objectPoints = plane.get_points()
        imagePoints = cam.project(objectPoints)

        # if plot:
        #  cam.plot_image(imagePoints)
        if ((imagePoints[0, :] < cam.img_width) & (imagePoints[0, :] > 0)).all():
            if ((imagePoints[1, :] < cam.img_height) & (imagePoints[1, :] > 0)).all():
                cams.append(cam)

    if plot:
        planes = []
        plane.uniform()
        planes.append(plane)
        plot3D(cams, planes)

    return cams

def create_cam_distribution_in_YZ(cam=None, plane_size=(0.3, 0.3), theta_params=(0, 180, 10),r_params=(0.25, 1.0, 4), plot=False):
    """
    cam distritubution in YZ plane
    :param cam:
    :param plane_size:
    :param theta_params:
    :param phi_params:
    :param r_params:
    :param plot:
    :return:
    """
    if cam == None:
        # Create an initial camera on the center of the world
        cam = Camera()
        f = 800
        cam.set_K(fx=f, fy=f, cx=320, cy=240)  # Camera Matrix
        cam.img_width = 320 * 2
        cam.img_height = 240 * 2

    # we create a default plane with 4 points with a side lenght of w (meters)
    plane = Plane(origin=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), size=plane_size, n=(2, 2))
    # We extend the size of this plane to account for the deviation from a uniform pattern
    # plane.size = (plane.size[0] + deviation, plane.size[1] + deviation)

    d_space = np.linspace(r_params[0], r_params[1], r_params[2])
    t_list = []
    for d in d_space:
        xx, yy, zz = uniform_halfCircle_in_YZ(theta_params, d, False)  # YZ plane
        sphere_points = np.array([xx.ravel(), yy.ravel(), zz.ravel()], dtype=np.float32)
        t_list.append(sphere_points)
    t_space = np.hstack(t_list)
    # print "t_space:",t_space.shape
    acc_row = r_params[2]
    acc_col = theta_params[2]
    accuracy_mat = np.zeros([acc_row,acc_col]) # accuracy_mat is used to describe accuracy degree for marker area
    # print accuracy_mat
    cams = []
    for t in t_space.T:
        cam = cam.clone()
        cam.set_t(-t[0], -t[1], -t[2])
        cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0.0, 0, 0))
        cam.look_at([0, 0, 0])

        radius = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
        # print "radius",radius
        angle = np.rad2deg(np.arccos(t[1]/radius))
        # print "angle",angle
        cam.set_radius(radius)
        cam.set_angle(angle)

        plane.set_origin(np.array([0, 0, 0]))
        plane.uniform()
        objectPoints = plane.get_points()
        # print "objectPoints",objectPoints
        imagePoints = cam.project(objectPoints)

        # if plot:
        #  cam.plot_image(imagePoints)
        if ((imagePoints[0, :] < cam.img_width) & (imagePoints[0, :] > 0)).all():
            if ((imagePoints[1, :] < cam.img_height) & (imagePoints[1, :] > 0)).all():
                cams.append(cam)

    if plot:
        planes = []
        plane.uniform()
        planes.append(plane)
        # plot3D(cams, planes) #TODO comment because of from mayavi import mlab

    return cams,accuracy_mat






# ==============================Test=================================================
# cams = create_cam_distribution(cam = None, plane_size = (0.5,0.5), theta_params = (0,360,20), phi_params =  (0,70,10), r_params = (0.2,2.0,10), plot=True)
# create_cam_distribution_in_YZ(cam = None, plane_size = (0.3,0.3), theta_params = (0,180,3), r_params = (0.3,0.9,3), plot=False)
# print "cams size: ",len(cams)
# -----------------------------Test for cam look at method------------------------------
# cam = Camera()
# f = 800
# cam.set_K(fx = f, fy = f, cx = 320, cy = 240)  #Camera Matrix
# cam.img_width = 320*2
# cam.img_height = 240*2
# cam.set_t(1,1,1,"world")
# cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0,np.deg2rad(0),0))
# cam.look_at([0,0,0])
# plane_size = (0.3,0.3)
# plane =  Plane(origin=np.array([0, 0, 0] ), normal = np.array([0, 0, 1]), size=plane_size, n = (2,2))
# plane.set_origin(np.array([0, 0, 0]))
# plane.uniform()
# planes = []
# planes.append(plane)
# cams = []
# cams.append(cam)
# plot3D(cams,planes)
#
# print "cam.R",cam.R
# print "cam.Rt",cam.Rt
# print "cam.P",cam.P
# ------------------Code End-----------Test for cam look at method------------------------------
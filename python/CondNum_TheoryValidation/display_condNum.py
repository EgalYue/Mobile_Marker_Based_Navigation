#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 19.12.17 12:45
@File    : display_condNum.py
@author: Yue Hu

This class is used to draw figure for condition number (served for brute_force.py)
"""
import sys
sys.path.append("..")
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
# from mayavi import mlab
from vision.camera import Camera
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t

# sio.savemat('testpython.mat', {'data': [[1, 2, 3], [1, 2, 3], [9, 19, 29],[80000,60000,3000]]})
# data = sio.loadmat('testpython.mat')
# sio.whosmat('testpython.mat')

def plot3D_cam(cam, axis_scale=0.2):
    # Coordinate Frame of real camera
    # Camera axis
    axis_scale = 0.05
    cam_axis_x = np.array([1, 0, 0, 1]).T
    cam_axis_y = np.array([0, 1, 0, 1]).T
    cam_axis_z = np.array([0, 0, 1, 1]).T

    cam_axis_x = np.dot(cam.R.T, cam_axis_x)
    cam_axis_y = np.dot(cam.R.T, cam_axis_y)
    cam_axis_z = np.dot(cam.R.T, cam_axis_z)

    cam_world = cam.get_world_position()

    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_x[0], cam_axis_x[1], cam_axis_x[2], line_width=3,
                  scale_factor=axis_scale, color=(1 - axis_scale, 0, 0))
    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_y[0], cam_axis_y[1], cam_axis_y[2], line_width=3,
                  scale_factor=axis_scale, color=(0, 1 - axis_scale, 0))
    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_z[0], cam_axis_z[1], cam_axis_z[2], line_width=3,
                  scale_factor=axis_scale, color=(0, 0, 1 - axis_scale))


# -------------------------------------------------------------------------------
def getConNumColor(condNum):
        color = "white"
        if condNum > 70000.0:
            color = 'linen'
        elif condNum > 60000.0:
            color = 'antiquewhite'
        elif condNum > 36100.0:
            color = 'papayawhip'
        elif condNum > 30300.0:
            color = 'oldlace'
        elif condNum > 24600.0:
            color = 'cornsilk'
        elif condNum > 18800.0:
            color = 'palegoldenrod'
        elif condNum > 13000.0:
            color = 'yellow'
        # elif condNum > 8000:
        #     color = 'lightblue'
        elif condNum > 7000.0:
            color = 'deepskyblue'
        elif condNum > 4000.0:
            color = 'red'
        elif condNum > 2000.0:
            color = 'green'
        elif condNum > 1000.0:
            color = 'maroon'
        else:
            color = 'black'

        return color

def displayCondNumDistribution(m):
    """
    Display distribution of cond num for cam distribution in 3D
    :param m: 4 * n,  each column vector is position x y z + cond num
    :return: 
    """
    cam = Camera()
    cam.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    cam.set_width_heigth(640, 480)
    cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0,np.deg2rad(180),0))
    cam.set_t(0, 0,5,'world')

    # plot3D_cam(cam)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = m[0,:]
    y = m[1,:]
    z = m[2,:]
    condNum = m[3,:]
    ax.scatter(x, y, z, s=100, c=condNum, marker="o", cmap="magma") # matplotlib colormap

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    # plt.pause(1000)

def displayError_XYfixed3D(z,input_ippe1_t,input_ippe1_R,input_ippe2_t,input_ippe2_R,input_pnp_t,input_pnp_R,input_transfer_error):
    """
    Fix X Y, only study Z 
    Display R_error and t_error for ippe and pnp method 
    :param z: 
    :param input_ippe1_t: 
    :param input_ippe1_R: 
    :param input_ippe2_t: 
    :param input_ippe2_R: 
    :param input_pnp_t: 
    :param input_pnp_R: 
    :param input_transfer_error: 
    :return: 
    """
    fig1 = plt.figure("Error")
    # ax = fig.gca(projection='3d')
    ax1 = fig1.add_subplot(231)
    ax1.plot(z, input_ippe1_t, label='ippe_tvec_error1')
    ax1.legend()
    ax1.set_xlabel('Z Label')
    ax1.set_ylabel('Error')

    ax2 = fig1.add_subplot(234)
    ax2.plot(z, input_ippe1_R, label='ippe_rmat_error1')
    ax2.legend()
    ax2.set_xlabel('Z Label')
    ax2.set_ylabel('Error')

    ax3 = fig1.add_subplot(232)
    ax3.plot(z, input_ippe2_t, label='ippe_tvec_error2')
    ax3.legend()
    ax3.set_xlabel('Z Label')
    ax3.set_ylabel('Error')

    ax2 = fig1.add_subplot(235)
    ax2.plot(z, input_ippe2_R, label='ippe_rmat_error2')
    ax2.legend()
    ax2.set_xlabel('Z Label')
    ax2.set_ylabel('Error')

    ax2 = fig1.add_subplot(233)
    ax2.plot(z, input_pnp_t, label='pnp_tmat_error')
    ax2.legend()
    ax2.set_xlabel('Z Label')
    ax2.set_ylabel('Error')

    ax2 = fig1.add_subplot(236)
    ax2.plot(z, input_pnp_R, label='pnp_rmat_error')
    ax2.legend()
    ax2.set_xlabel('Z Label')
    ax2.set_ylabel('Error')

    # ----------------- Transfer Error ----------------------------------
    fig2 = plt.figure("Transfer Error ")
    # ax = fig.gca(projection='3d')
    ax_transfer_error = fig2.add_subplot(111)
    ax_transfer_error.plot(z, input_transfer_error, label='transfer_error')
    ax_transfer_error.legend()
    ax_transfer_error.set_xlabel('Z Label')
    ax_transfer_error.set_ylabel('Transfer Error')


    plt.show()
    plt.pause(1000)

def displayError_Zfixed3D(x,y,input_ippe1_t,input_ippe1_R,input_ippe2_t,input_ippe2_R,input_pnp_t,input_pnp_R,input_transfer_error):
    """
     Fix Z, study X Y
     Display R_error and t_error for ippe and pnp method 
    """
    fig1 = plt.figure("Error")
    # ax = fig.gca(projection='3d')
    ax1 = fig1.add_subplot(231, projection='3d')
    ax1.scatter(x, y, input_ippe1_t, marker = ".")
    ax1.legend()
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('ippe_tvec_error1')

    ax2 = fig1.add_subplot(234, projection='3d')
    ax2.scatter(x, y, input_ippe1_R, marker = ".")
    ax2.legend()
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_zlabel('ippe_rmat_error1')

    ax3 = fig1.add_subplot(232, projection='3d')
    ax3.scatter(x, y, input_ippe2_t, marker = ".")
    ax3.legend()
    ax3.set_xlabel('X Label')
    ax3.set_ylabel('Y Label')
    ax3.set_zlabel('ippe_tvec_error2')

    ax4 = fig1.add_subplot(235, projection='3d')
    ax4.scatter(x, y, input_ippe2_R, marker = ".")
    ax4.legend()
    ax4.set_xlabel('X Label')
    ax4.set_ylabel('Y Label')
    ax4.set_zlabel('ippe_rmat_error2')

    ax5 = fig1.add_subplot(233, projection='3d')
    ax5.scatter(x, y, input_pnp_t, marker = ".")
    ax5.legend()
    ax5.set_xlabel('X Label')
    ax5.set_ylabel('Y Label')
    ax5.set_zlabel('pnp_tmat_error')

    ax6 = fig1.add_subplot(236, projection='3d')
    ax6.scatter(x, y, input_pnp_R, marker = ".")
    ax6.legend()
    ax6.set_xlabel('X Label')
    ax6.set_ylabel('Y Label')
    ax6.set_zlabel('pnp_rmat_error')

    plt.savefig("Error.png")
    plt.show()


def displayError_YZ_plane_3D(y, z, input_ippe1_t, input_ippe1_R, input_ippe2_t, input_ippe2_R, input_pnp_t, input_pnp_R, input_transfer_error):
    """
     Display R_error and t_error for ippe and pnp method
     cam dstribution only on YZ plane
    """
    fig1 = plt.figure("R_Error_and_t_Error")
    ax1 = fig1.add_subplot(234, projection='3d')
    ax1.scatter(y, z, input_ippe1_t, marker=".")
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('ippe_tvec_error1')

    ax2 = fig1.add_subplot(231, projection='3d')
    ax2.scatter(y, z, input_ippe1_R, marker=".")
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('ippe_rmat_error1')

    ax3 = fig1.add_subplot(235, projection='3d')
    ax3.scatter(y, z, input_ippe2_t, marker=".")
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_zlabel('ippe_tvec_error2')

    ax4 = fig1.add_subplot(232, projection='3d')
    ax4.scatter(y, z, input_ippe2_R, marker=".")
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_zlabel('ippe_rmat_error2')

    ax5 = fig1.add_subplot(236, projection='3d')
    ax5.scatter(y, z, input_pnp_t, marker=".")
    ax5.set_xlabel('Y')
    ax5.set_ylabel('Z')
    ax5.set_zlabel('pnp_tmat_error')

    ax6 = fig1.add_subplot(233, projection='3d')
    ax6.scatter(y, z, input_pnp_R, marker=".")
    ax6.set_xlabel('Y')
    ax6.set_ylabel('Z')
    ax6.set_zlabel('pnp_rmat_error')

    plt.savefig("R_Error_and_t_Error_3D.png")
    plt.show()
    # ----------------- Transfer Error ----------------------------------
    # fig2 = plt.figure("Transfer Error ")
    # ax_transfer_error = fig2.add_subplot(111, projection='3d')
    # ax_transfer_error.scatter(y,z, input_transfer_error,marker=".", label='transfer_error')
    # ax_transfer_error.legend()
    # ax_transfer_error.set_xlabel('Y')
    # ax_transfer_error.set_ylabel('Z')
    # ax_transfer_error.set_zlabel('Transfer Error')
    # plt.show()

def displayError_YZ_plane_2D(y, z, input_ippe1_t, input_ippe1_R, input_ippe2_t, input_ippe2_R, input_pnp_t,
                          input_pnp_R):
    """
     Display R_error and t_error for ippe and pnp method
     Using colorbar on YZ plane
    """
    fig1 = plt.figure("R_Error_and_t_Error")
    ax1 = fig1.add_subplot(231)
    plt.scatter(y, z, s=100, c=input_ippe1_R, marker="o", cmap="magma")
    ax1.set_xlabel('Y ippe1_R_Error')
    ax1.set_ylabel('Z ippe1_R_Error')

    ax2 = fig1.add_subplot(234)
    plt.scatter(y, z, s=100, c=input_ippe1_t, marker="o", cmap="magma")
    ax2.set_xlabel('Y ippe1_t_Error')
    ax2.set_ylabel('Z ippe1_t_Error')

    ax3 = fig1.add_subplot(232)
    plt.scatter(y, z, s=100, c=input_ippe2_R, marker="o", cmap="magma")
    ax3.set_xlabel('Y ippe2_R_Error')
    ax3.set_ylabel('Z ippe2_R_Error')

    ax4 = fig1.add_subplot(235)
    plt.scatter(y, z, s=100, c=input_ippe2_t, marker="o", cmap="magma")
    ax4.set_xlabel('Y ippe2_t_Error')
    ax4.set_ylabel('Z ippe2_t_Error')

    ax5 = fig1.add_subplot(233)
    plt.scatter(y, z, s=100, c=input_pnp_R, marker="o", cmap="magma")
    ax5.set_xlabel('Y pnp_R_Error')
    ax5.set_ylabel('Z pnp_R_Error')

    ax6 = fig1.add_subplot(236)
    plt.scatter(y, z, s=100, c=input_pnp_t, marker="o", cmap="magma")
    ax6.legend()
    ax6.set_xlabel('Y pnp_t_Error')
    ax6.set_ylabel('Z pnp_t_Error')
    plt.colorbar()
    plt.savefig("R_Error_and_t_Error.png")
    plt.show()

#--------------------------------------------- Test--------------------------------------------------
# Detectionplot()
# anglePlot()
# matrix = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1000,20000,5000,50000]])
# displayCondNumDistribution(matrix)
# displayError_Zfixed3D(matrix)


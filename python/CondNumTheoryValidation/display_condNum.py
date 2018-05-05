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

# -------------------------------------------------------------------------------
def displayCondNumDistribution(m):
    """
    Display distribution of condition number for cam distribution in 3D
    :param m: 4 * n,  each column vector is position x y z + cond num
    :return: 
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = m[0,:]
    y = m[1,:]
    z = m[2,:]
    condNum = m[3,:]
    surf = ax.scatter(x, y, z, s=100, c=condNum, marker="o", cmap="magma") # matplotlib colormap
    ax.set_title("Condition number distribution")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.8, aspect=10)

    plt.show()

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
    ax_transfer_error = fig2.add_subplot(111)
    ax_transfer_error.plot(z, input_transfer_error, label='transfer_error')
    ax_transfer_error.legend()
    ax_transfer_error.set_xlabel('Z Label')
    ax_transfer_error.set_ylabel('Transfer Error')

    plt.show()

def displayError_Zfixed3D(x,y,input_ippe1_t,input_ippe1_R,input_ippe2_t,input_ippe2_R,input_pnp_t,input_pnp_R,input_transfer_error):
    """
     Fix Z, study X Y
     Display R_error and t_error for ippe and pnp method 
    """
    fig1 = plt.figure("Error")
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
    ax1.set_zlabel('ippe1_tvec_error')

    ax2 = fig1.add_subplot(231, projection='3d')
    ax2.scatter(y, z, input_ippe1_R, marker=".")
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('ippe1_rmat_error')

    ax3 = fig1.add_subplot(235, projection='3d')
    ax3.scatter(y, z, input_ippe2_t, marker=".")
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_zlabel('ippe2_tvec_error')

    ax4 = fig1.add_subplot(232, projection='3d')
    ax4.scatter(y, z, input_ippe2_R, marker=".")
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_zlabel('ippe2_rmat_error')

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
    ax1.set_title("ippe1_R_Error")
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')

    ax2 = fig1.add_subplot(234)
    plt.scatter(y, z, s=100, c=input_ippe1_t, marker="o", cmap="magma")
    ax2.set_title("ippe1_t_Error")
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')

    ax3 = fig1.add_subplot(232)
    plt.scatter(y, z, s=100, c=input_ippe2_R, marker="o", cmap="magma")
    ax3.set_title("ippe2_R_Error")
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')

    ax4 = fig1.add_subplot(235)
    plt.scatter(y, z, s=100, c=input_ippe2_t, marker="o", cmap="magma")
    ax4.set_title("ippe2_t_Error")
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')

    ax5 = fig1.add_subplot(233)
    plt.scatter(y, z, s=100, c=input_pnp_R, marker="o", cmap="magma")
    ax5.set_title("pnp_R_Error")
    ax5.set_xlabel('Y')
    ax5.set_ylabel('Z')

    ax6 = fig1.add_subplot(236)
    plt.scatter(y, z, s=100, c=input_pnp_t, marker="o", cmap="magma")
    ax6.set_title("pnp_t_Error")
    ax6.set_xlabel('Y')
    ax6.set_ylabel('Z')

    plt.colorbar()
    plt.savefig("R_Error_and_t_Error.png")
    plt.show()

#--------------------------------------------- Test--------------------------------------------------
# Detectionplot()
# anglePlot()
# matrix = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1000,20000,5000,50000]])
# displayCondNumDistribution(matrix)
# displayError_Zfixed3D(matrix)


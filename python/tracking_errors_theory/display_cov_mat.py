#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 14.02.18 22:41
@File    : display_cov_mat.py
@author: Yue Hu
"""
import matplotlib.pyplot as plt

def displayCovVolume_Zfixed3D(xInputs,yInputs,volumes):
    """
    Display covariance matrix volume
    :return:
    """
    fig = plt.figure("Transfer Error ")
    # ax = fig.gca(projection='3d')
    ax_transfer_error = fig.add_subplot(111, projection='3d')
    ax_transfer_error.scatter(xInputs, yInputs, volumes, marker = ".")
    ax_transfer_error.legend()
    ax_transfer_error.set_xlabel('X Label')
    ax_transfer_error.set_ylabel('Y Label')
    ax_transfer_error.set_zlabel('Volume')

    plt.savefig("covariance matrix volume.png")
    plt.show()
    # plt.pause(1000)


def displayCovVolume_XYfixed3D(zInputs,volumes):
    """
    Display covariance matrix volume
    :return:
    """
    fig = plt.figure("Transfer Error ")
    # ax = fig.gca(projection='3d')
    ax_transfer_error = fig.add_subplot(111)
    ax_transfer_error.scatter(zInputs, volumes, marker = ".")
    ax_transfer_error.legend()
    ax_transfer_error.set_xlabel('Z Label')
    ax_transfer_error.set_ylabel('Volume')

    plt.savefig("covariance_matrix_volume_XY_fixed.png")
    plt.show()
    # plt.pause(1000)
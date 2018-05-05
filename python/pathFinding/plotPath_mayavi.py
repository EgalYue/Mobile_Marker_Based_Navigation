#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 05.05.18 12:36
@File    : plotPath_mayavi.py
@author: Yue Hu
"""

# --------------!!! important!!!----------
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'wx'
#-----------------------------------------
import numpy as np
from mayavi import mlab

def plotComparePaths_DisError_3DSurface(xyError_list_AllPaths, grid_reso = 0.1, width = 3, height = 6):
    """
    Using Mayavi
    Plot the measured paths with 3D surface(like mountain, high mountain means big error, low mountain means small error)
    1.Figure:
    :param xyError_list_AllPaths:
    :param width: 3 [m]
    :param height: 6 [m]
    :param grid_reso: default 0.1 [m]
    :return:
    """
    path_num = len(xyError_list_AllPaths)
    gridX = int(width / grid_reso)
    gridY = int(height / grid_reso)

    X, Y = np.mgrid[0:width:grid_reso, 0:height:grid_reso]
    Z = np.zeros((gridX, gridY))

    for i in range(path_num):
        xyError_list = xyError_list_AllPaths[i]
        iter_num = xyError_list.shape[0] / 3
        for j in range(iter_num):
            x_real = xyError_list[0+j*3,:]
            y_real = xyError_list[1+j*3,:]

            ix = (x_real - grid_reso / 2) / grid_reso
            ix = ix.astype(int)

            iy = (y_real - grid_reso / 2) / grid_reso
            iy = iy.astype(int)
            Z[ix,iy] = xyError_list[2+j*3,:]

    s = mlab.surf(X, Y, Z,warp_scale="auto")

    cat1_extent = (0, 30, 0, 60, -10, 10)
    mlab.outline(s, color=(.7, .7, .7), extent=cat1_extent)
    mlab.text(0, 30, 'pf', z=0, width=0.14)
    mlab.text(30, 30, 'A*', z=0, width=0.14)
    mlab.title('Compare measured error(Euclidean distance) between Potential field method and A*')

    mlab.show()

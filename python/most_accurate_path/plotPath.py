#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 17.04.18 11:45
@File    : plotPath.py
@author: Yue Hu
"""
import matplotlib.pyplot as plt
import numpy as np


def plotPath(fix_path,real_path,measured_path):
    plt.figure()
    # plt.axis([0, 6, 0, 3])
    plt.grid(True)                         # set the grid

    # !!! In our case in the real world coordinate system we set x-axis as the plot-y, y-axis as the plot-x
    #---------------------Fix path----------------------------------------
    x_fix = fix_path[0,:]
    y_fix = fix_path[1,:]
    plt.plot(y_fix, x_fix, color='black', label='Fix path') #  x,y  exchange position
    plt.scatter(y_fix, x_fix, c="black", marker='o')
    #---------------------Real path----------------------------------------
    x_real = real_path[0,:]
    y_real = real_path[1,:]
    plt.plot(y_real, x_real, color='red', label='Real path') #  x,y  exchange position
    plt.scatter(y_real, x_real, c="red", marker='x')
    #---------------------Measured path----------------------------------------
    x_measured = measured_path[0,:]
    y_measured = measured_path[1,:]
    plt.plot(y_measured, x_measured, color='blue', label='measured path') #  x,y  exchange position
    plt.scatter(y_measured, x_measured,c="blue", marker='+')
    plt.legend() # show label of line

    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax.xaxis.tick_top()                     # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax.yaxis.tick_left()                    # remove right y-Ticks
    ax.set_title("Path")
    ax.set_xlabel('Y_W')
    ax.set_ylabel('X_W')
    plt.show()





#=====================================Test=====================================================
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 17.04.18 11:45
@File    : plotPath.py
@author: Yue Hu
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

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

    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop=fontP,loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3) # Move legend outside of figure in matplotlib

    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax.xaxis.tick_top()                     # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax.yaxis.tick_left()                    # remove right y-Ticks
    ax.set_title("Path")
    ax.set_xlabel('Y_W')
    ax.set_ylabel('X_W')
    # plt.show() # Just use one plt.show in plotAll() method

def plotPositionError(fix_path,real_path,measured_path):
    tem_real = np.square(fix_path - real_path)
    real_path_error = np.sqrt(tem_real[0,:] + tem_real[1,:])

    tem_measured = np.square(fix_path - measured_path)
    measured_path_error = np.sqrt(tem_measured[0,:] + tem_measured[1,:])

    plt.figure()
    #---------------------Real path----------------------------------------
    x_real = range(1,len(real_path_error)+1)
    y_real = real_path_error
    plt.plot(x_real, y_real, color='red', label='Real path')
    plt.scatter(x_real, y_real, c="red", marker='x')
    #---------------------Measured path----------------------------------------
    x_measured = range(1,len(measured_path_error)+1)
    y_measured = measured_path_error
    plt.plot(x_measured, y_measured, color='blue', label='measured path') #  x,y  exchange position
    plt.scatter(x_measured, y_measured,c="blue", marker='+')

    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop=fontP,loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3) # Move legend outside of figure in matplotlib

    ax=plt.gca()
    ax.set_title("Position error (Euclidean distance)")
    ax.set_xlabel('Step')
    ax.set_ylabel('Error(m)')
    # plt.show() # Just use one plt.show in plotAll() method

def plot_Rt_error(Rmat_error_list, tvec_error_list):
    fig = plt.figure('Pose estimation errors')
    ax_R_error = fig.add_subplot(121)
    ax_t_error = fig.add_subplot(122)
    #------------------------ R error ---------------------------------
    x_R_error = np.arange(1, len(Rmat_error_list)+1, 1)
    y_R_error = Rmat_error_list
    ax_R_error.xaxis.set_ticks(np.arange(0, len(Rmat_error_list)+1, 1))
    ax_R_error.plot(x_R_error, y_R_error, color='blue', label='measured path')
    ax_R_error.scatter(x_R_error, y_R_error, c="red", marker='o')

    plt.sca(ax_R_error)
    ax_R_error.set_title("R error(" + u"\u00b0" + ")")
    ax_R_error.set_xlabel('Step')
    ax_R_error.set_ylabel('Angle(degree)')

    #-------------------------r error ---------------------------------
    x_t_error = np.arange(1, len(tvec_error_list)+1, 1)
    y_t_error = tvec_error_list
    ax_t_error.xaxis.set_ticks(np.arange(0, len(tvec_error_list)+1, 1))
    ax_t_error.plot(x_t_error, y_t_error, color='blue', label='measured path')
    ax_t_error.scatter(x_t_error, y_t_error, c="red", marker='o')

    plt.sca(ax_t_error)
    ax_t_error.set_title("t error(%)")
    ax_t_error.set_xlabel('Step')
    ax_t_error.set_ylabel('Percent')
    # plt.show()

def plotAll(fix_path, real_path,measured_path, Rmat_error_list, tvec_error_list):
    plotPath(fix_path, real_path, measured_path)
    plotPositionError(fix_path, real_path, measured_path)
    plot_Rt_error(Rmat_error_list, tvec_error_list)
    plt.show()


#=====================================Test=====================================================
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 28.04.18 22:19
@File    : plotPath.py
@author: Yue Hu
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plotPath(fix_path, measured_path):
    plt.figure()
    # plt.axis([0, 6, 0, 3])
    plt.grid(True)  # set the grid

    # !!! In our case in the real world coordinate system we set x-axis as the plot-y, y-axis as the plot-x
    # ---------------------Fix path----------------------------------------
    x_fix = fix_path[0, :]
    y_fix = fix_path[1, :]
    plt.plot(y_fix, x_fix, color='black', label='Fix path')  # x,y  exchange position
    plt.scatter(y_fix, x_fix, c="black", marker='o')
    # ---------------------Measured path----------------------------------------
    x_measured = measured_path[0, :]
    y_measured = measured_path[1, :]
    plt.plot(y_measured, x_measured, color='blue', label='measured path')  # x,y  exchange position
    plt.scatter(y_measured, x_measured, c="blue", marker='+')

    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax = plt.gca()  # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax.yaxis.tick_left()  # remove right y-Ticks
    ax.set_title("Path")
    ax.set_xlabel('Y_W')
    ax.set_ylabel('X_W')
    plt.show() # Just use one plt.show in plotAll() method


def plotPositionError(fix_path, measured_path):
    tem_measured = np.square(fix_path - measured_path)
    measured_path_error = np.sqrt(tem_measured[0, :] + tem_measured[1, :])

    plt.figure()

    # ---------------------Measured path----------------------------------------
    x_measured = range(1, len(measured_path_error) + 1)
    y_measured = measured_path_error
    plt.plot(x_measured, y_measured, color='blue', label='measured path')  # x,y  exchange position
    plt.scatter(x_measured, y_measured, c="blue", marker='+')

    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax = plt.gca()
    ax.set_title("Position error (Euclidean distance)")
    ax.set_xlabel('Step')
    ax.set_ylabel('Error(m)')
    plt.show() # Just use one plt.show in plotAll() method

def plotAllPaths(fix_path_list, measured_path_list, Rmat_error_list, tvec_error_list):
    """
    Plot all paths in one figure
    :param fix_path_list:
    :param real_path_list:
    :param measured_path_list:
    :param Rmat_error_list:
    :param tvec_error_list:
    :return:
    """
    fig = plt.figure("Compare different paths")
    ax_fix = fig.add_subplot(121)
    ax_measured = fig.add_subplot(122)

    path_num = len(fix_path_list)
    # colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1.0, path_num))))

    #====================== Plot position error ============================
    fig2 = plt.figure("Position error (Euclidean distance)")
    ax_measured_distance = fig2.add_subplot(111)

    # ====================== Plot R t error ============================
    fig3 = plt.figure('Pose estimation errors')
    ax_R_error = fig3.add_subplot(121)
    ax_t_error = fig3.add_subplot(122)

    for i in range(0, path_num):
        x_fix = fix_path_list[i][0, :]
        y_fix = fix_path_list[i][1, :]
        l = ax_fix.plot(y_fix, x_fix, label=str(i))
        ax_fix.scatter(y_fix, x_fix, c="black", marker='o')

        colour = l[0].get_color()

        x_measured = measured_path_list[i][0, :]
        y_measured = measured_path_list[i][1, :]
        ax_measured.plot(y_measured, x_measured, color= colour, label=str(i))
        ax_measured.scatter(y_measured, x_measured, c="black", marker='o')
        # labels.append(r'$$')

        # ====================== Plot position error ============================
        # ---------------------Measured path----------------------------------------
        tem_measured = np.square(fix_path_list[i] - measured_path_list[i])
        measured_path_error = np.sqrt(tem_measured[0, :] + tem_measured[1, :])
        x_measured = np.arange(1, len(measured_path_error) + 1, 1)
        y_measured = measured_path_error
        ax_measured_distance.plot(x_measured, y_measured, color=colour, label=str(i))  # x,y  exchange position
        ax_measured_distance.scatter(x_measured, y_measured, c="black", marker='o')

        # ====================== Plot R t error ============================
        # ------------------------ R error ---------------------------------
        Rmat_error = Rmat_error_list[i]
        x_R_error = np.arange(1, len(Rmat_error) + 1, 1)

        y_R_error = Rmat_error
        ax_R_error.xaxis.set_ticks(np.arange(0, len(Rmat_error) + 1, 1))
        ax_R_error.plot(x_R_error, y_R_error, color=colour, label='Fixed path R error')
        ax_R_error.scatter(x_R_error, y_R_error, c="black", marker='o')
        # -------------------------r error ---------------------------------
        tvec_error = tvec_error_list[i]
        x_t_error = np.arange(1, len(tvec_error) + 1, 1)
        y_t_error = tvec_error
        ax_t_error.xaxis.set_ticks(np.arange(0, len(tvec_error) + 1, 1))
        ax_t_error.plot(x_t_error, y_t_error, color=colour, label='Fixed path t error')
        ax_t_error.scatter(x_t_error, y_t_error, c="black", marker='o')

    #---------------- Fix path ------------------------------
    fontP = FontProperties()
    fontP.set_size('small')
    ax_fix.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax_fix.set_ylim(ax_fix.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax_fix.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax_fix.yaxis.tick_left()  # remove right y-Ticks
    # ax_fix.set_title("Compare different fix paths")
    ax_fix.set_xlabel('Fixed path: Y_W')
    ax_fix.set_ylabel('Fixed path: X_W')

    #---------------- Measured path ------------------------------
    fontP = FontProperties()
    fontP.set_size('small')
    ax_measured.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax_measured.set_ylim(ax_measured.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax_measured.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax_measured.yaxis.tick_left()  # remove right y-Ticks
    # ax_real.set_title("Compare different real paths")
    ax_measured.set_xlabel('Measured path: Y_W')
    ax_measured.set_ylabel('Measured path: X_W')

    # ====================== Plot position error =========================================
    fontP = FontProperties()
    fontP.set_size('small')
    ax_measured_distance.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax_measured_distance.set_title("Measured paths: Position error (Euclidean distance)")
    ax_measured_distance.set_xlabel('Step')
    ax_measured_distance.set_ylabel('Error(m)')

    # ====================== Plot R t error ============================
    plt.sca(ax_R_error)
    ax_R_error.set_title("R error(" + u"\u00b0" + ")")
    ax_R_error.set_xlabel('Step')
    ax_R_error.set_ylabel('Angle(degree)')

    plt.sca(ax_t_error)
    ax_t_error.set_title("t error(%)")
    ax_t_error.set_xlabel('Step')
    ax_t_error.set_ylabel('Percent')

    plt.show() # Just use one plt.show in plotAll() method

def plotPositionErrorFillBetween(mean_error, std_error):
    """
    ONLY plot one path error
    :param fix_path:
    :param mean_error:
    :param std_error:
    :return:
    """
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax = plt.gca()
    x_steps = range(1, len(mean_error) + 1)
    y_error = np.asarray(mean_error) # convert to array
    yerr = np.asarray(std_error) # convert to array

    color = "r"
    alpha_fill = 0.3
    if np.isscalar(yerr) or len(yerr) == len(y_error):
        ymin = y_error - yerr
        ymax = y_error + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr

    ax.plot(x_steps, y_error, color='blue', label='measured path')
    plt.scatter(x_steps, y_error, c="blue", marker='+')
   # ax.fill_between(x_steps, ymax, ymin, color=color, alpha=alpha_fill)

    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax = plt.gca()
    ax.set_title("Position error (Euclidean distance)")
    ax.set_xlabel('Step')
    ax.set_ylabel('Error(m)')

    plt.show()  # Just use one plt.show in plotAll() method

def plotScatterEachStep(allPaths_pos_list):
    colors = ['r','b','g','y','r','b','g','y','r','b','g','y']
    i = -1
    fig = plt.figure("plotScatterEachStep")
    ax_fix = fig.add_subplot(111)
    for allPos_list in allPaths_pos_list:
        i +=1
        for step in allPos_list:
            x = step[0,:]
            y = step[1,:]
            area = 1  # 0 to 15 point radii
            # plt.scatter(x, y, s=area, c= colors,alpha=0.5,cmap="magma")
            ax_fix.plot(y, x, '.',color = colors[i])

    plt.xlabel('Y measured')
    plt.ylabel('X measured')
    plt.title('')
    fontP = FontProperties()
    fontP.set_size('small')
    ax_fix.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax_fix.set_ylim(ax_fix.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax_fix.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax_fix.yaxis.tick_left()  # remove right y-Ticks

    plt.show()

def plotFixedMeasuredFillBetween(fix_path_list, disErrorMean_list):
    """
    Plot the fixed paths and fill with mean error.
    1.Figure: Fixed paths and fill with error(eulidean diantance)
    """
    fig = plt.figure("Compare different paths")
    ax_fix = fig.add_subplot(111)
    path_num = len(fix_path_list)
    # colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1.0, path_num))))

    colours = ['r','b']
    for i in range(0, path_num):
        #-------------------------------------------------------------
        if i ==0:
            label = "Potential field pathfinding"
        if i ==1:
            label = "A star pathfinding"
        color = colours[i] # only for two paths

        x_fix = fix_path_list[i][0, :]
        y_fix = fix_path_list[i][1, :]
        xerr = disErrorMean_list[i]

        # ax_fix.plot(y_fix, x_fix,color =color, label=label)
        # ax_fix.scatter(y_fix, x_fix, c="black", marker='o')

        alpha_fill = 0.3
        if np.isscalar(xerr) or len(xerr) == len(x_fix):
            xmin = x_fix - xerr
            xmax = x_fix + xerr
        elif len(xerr) == 2:
            xmin, xmax = xerr

        ax_fix.plot(y_fix, x_fix, color=color, label = label)
        plt.scatter(y_fix, x_fix, c="blue", marker='o')
        ax_fix.fill_between(y_fix, xmax, xmin, color=color, alpha=alpha_fill)

    #---------------- Fix path ------------------------------
    fontP = FontProperties()
    # fontP.set_size('small')
    ax_fix.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib
    ax_fix.set_ylim(ax_fix.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax_fix.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax_fix.yaxis.tick_left()  # remove right y-Ticks
    # ax_fix.set_title("Compare different fix paths")
    ax_fix.set_xlabel('Fixed path: Y_W')
    ax_fix.set_ylabel('Fixed path: X_W')

    plt.show() # Just use one plt.show in plotAll() method

def plotComparePaths(fix_path_list, disErrorMean_list, disErrorStd_list, Rmat_error_list_AllPaths, tvec_error_list_AllPaths, Rmat_error_std_list_AllPaths, tvec_error_std_list_AllPaths):
    """
    Plot two paths in order to compare
    1.Figure compare fixed paths (in wolrd coordinate)
    2.Figure compare position error(Euclidean distance)
    3.Figure compare R_error and t_error (errorbar)

    :param fix_path_list:
    :param real_path_list:
    :param measured_path_list:
    :return:
    """
    fig = plt.figure("Compare different paths(fixed paths)")
    ax_fix = fig.add_subplot(111)

    path_num = len(fix_path_list)
    # colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1.0, path_num))))

    #====================== Plot position error ============================
    fig2 = plt.figure("Position error (Euclidean distance)")
    ax_measured_distance = fig2.add_subplot(111)
    #=======================R t error ======================================
    fig3 = plt.figure('Compare R error and t error of different paths')
    ax_R_error = fig3.add_subplot(121)
    ax_t_error = fig3.add_subplot(122)

    colours = ['r','b'] # Now we only compare 2 paths, just use 2 colors
    for i in range(0, path_num):
        #-------------------------------------------------------------
        color = colours[i] # only for two paths
        x_fix = fix_path_list[i][0, :]
        if i ==0:
            label = "Potential field pathfinding"
        if i ==1:
            label = "A star pathfinding"
        #-------------------------------------------------------------
        y_fix = fix_path_list[i][1, :]
        l = ax_fix.plot(y_fix, x_fix,color =color, label=label)
        ax_fix.scatter(y_fix, x_fix, c="black", marker='o')

        # ====================== Plot position error ============================
        # ---------------------Measured path----------------------------------------
        x_measured = np.arange(1, len(disErrorMean_list[i]) + 1, 1)
        y_measured = disErrorMean_list[i]
        ax_measured_distance.plot(x_measured, y_measured, color=color, label=label)  # x,y  exchange position
        ax_measured_distance.scatter(x_measured, y_measured, c="black", marker='o')
        # # fill between
        # alpha_fill = 0.3
        # yerr = disErrorStd_list[i]  # convert to array
        # if np.isscalar(yerr) or len(yerr) == len(y_measured):
        #     ymin = y_measured - yerr
        #     ymax = y_measured + yerr
        # elif len(yerr) == 2:
        #     ymin, ymax = yerr
        #
        # ax_measured_distance.fill_between(x_measured, ymax, ymin, color=colour, alpha=alpha_fill)
        #-----------errorbar------------------------------------------------------------
        yerr = disErrorStd_list[i]
        ax_measured_distance.errorbar(x_measured, y_measured, yerr=yerr,color =color, ecolor=color, ms=20*i+20, mew=4*i+4)

        # ====================== Plot R t error ============================
        # ------------------------ R error ---------------------------------
        Rmat_error = Rmat_error_list_AllPaths[i]
        x_R_error = np.arange(1, len(Rmat_error) + 1, 1)
        y_R_error = Rmat_error
        ax_R_error.xaxis.set_ticks(np.arange(0, len(Rmat_error) + 1, 1))
        ax_R_error.plot(x_R_error, y_R_error, color=color, label= label)
        ax_R_error.scatter(x_R_error, y_R_error, c="black", marker='o')
        yerr_Rerror = Rmat_error_std_list_AllPaths[i]

        ax_R_error.errorbar(x_R_error, y_R_error, yerr=yerr_Rerror,color =color, ecolor=color, ms=20*i+20, mew=4*i+4)
        # -------------------------r error ---------------------------------
        tvec_error = tvec_error_list_AllPaths[i]
        x_t_error = np.arange(1, len(tvec_error) + 1, 1)
        y_t_error = tvec_error
        ax_t_error.xaxis.set_ticks(np.arange(0, len(tvec_error) + 1, 1))
        ax_t_error.plot(x_t_error, y_t_error, color=color, label= label)
        ax_t_error.scatter(x_t_error, y_t_error, c="black", marker='o')
        yerr_terror = tvec_error_std_list_AllPaths[i]
        ax_t_error.errorbar(x_t_error, y_t_error, yerr=yerr_terror, color=color, ecolor=color, ms=20 * i + 20, mew=4 * i + 4)

    #---------------- Fix path ------------------------------
    fontP = FontProperties()
    # fontP.set_size('small')
    ax_fix.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax_fix.set_ylim(ax_fix.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax_fix.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax_fix.yaxis.tick_left()  # remove right y-Ticks
    # ax_fix.set_title("Compare different fix paths")
    ax_fix.set_xlabel('Fixed path: Y_W')
    ax_fix.set_ylabel('Fixed path: X_W')

    # ====================== Plot position error =========================================
    fontP = FontProperties()
    # fontP.set_size('small')
    ax_measured_distance.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib
    ax_measured_distance.set_title("Position error between fixed path and measured path(Euclidean distance)")
    ax_measured_distance.set_xlabel('Step')
    ax_measured_distance.set_ylabel('Error(m)')

    # ====================== Plot R t error ============================
    plt.sca(ax_R_error)
    # fontP = FontProperties()
    # fontP.set_size('small')
    # ax_R_error.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib
    ax_R_error.set_title("R error(" + u"\u00b0" + ")")
    ax_R_error.set_xlabel('Step')
    ax_R_error.set_ylabel('Angle(degree)')

    plt.sca(ax_t_error)
    fontP = FontProperties()
    fontP.set_size('small')
    ax_R_error.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib
    ax_t_error.set_title("t error(%)")
    ax_t_error.set_xlabel('Step')
    ax_t_error.set_ylabel('Percent')
    plt.show() # Just use one plt.show in plotAll() method

def plotComparePaths_DisError_3DSurface(xyError_list_AllPaths, grid_reso = 0.1, width = 3, height = 6):
    """
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

    fig = plt.figure("Compare different paths")
    ax = Axes3D(fig)

    Y = np.arange(0, width, grid_reso)
    X = np.arange(0, height, grid_reso)
    X, Y = np.meshgrid(X, Y)
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

            # print "X.shape",X.shape
            # print "Y.shape",Y.shape
            # print "Z.shape",Z.shape
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma)

    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    # ax.yaxis.tick_left()  # remove right y-Ticks
    ax.set_xlabel('Y: world coordinate')
    ax.set_ylabel('X: world coordinate')
    ax.set_zlabel('Error between fixed path and measured path (Euclidean distance)')
    plt.show()

def plotComparePaths_R_error_3DSurface(fix_path_list, Rmat_error_mean_list_AllPaths, grid_reso = 0.1, width = 6, height = 3):
    """
    Plot the measured paths with 3D surface(like mountain, high mountain means big error, low mountain means small error)
    1.Figure:
    :param xyError_list_AllPaths:
    :param width: 3 [m]
    :param height: 6 [m]
    :param resolution: default 0.1 [m]
    :return:
    """
    #TODO
    path_num = len(fix_path_list)

    gridX = int(height / grid_reso)
    gridY = int(width / grid_reso)

    fig = plt.figure("Compare R error of different paths")
    ax = Axes3D(fig)

    Y = np.arange(0, height, grid_reso)
    X = np.arange(0, width, grid_reso)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((gridX, gridY))

    for i in range(path_num):
        Rmat_error_mean_list = Rmat_error_mean_list_AllPaths[i]
        fixed_path = fix_path_list[i]
        x_real = fixed_path[0,:]
        y_real = fixed_path[1,:]

        ix = (x_real - grid_reso / 2) / grid_reso
        ix = ix.astype(int)
        iy = (y_real - grid_reso / 2) / grid_reso
        iy = iy.astype(int)

        Z[ix,iy] = Rmat_error_mean_list

            # print "X.shape",X.shape
            # print "Y.shape",Y.shape
            # print "Z.shape",Z.shape
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma)

    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    # ax.yaxis.tick_left()  # remove right y-Ticks
    ax.set_xlabel('Y: world coordinate')
    ax.set_ylabel('X: world coordinate')
    ax.set_zlabel("R error(" + u"\u00b0" + ")")
    plt.show()

def plotComparePaths_t_error_3DSurface(fix_path_list, tvec_error_mean_list_AllPaths, grid_reso = 0.1, width = 6, height = 3):
    """
    Plot the measured paths with 3D surface(like mountain, high mountain means big error, low mountain means small error)
    1.Figure:
    :param xyError_list_AllPaths:
    :param width: 3 [m]
    :param height: 6 [m]
    :param grid_reso: default 0.1 [m]
    :return:
    """
    #TODO
    path_num = len(fix_path_list)

    gridX = int(height / grid_reso)
    gridY = int(width / grid_reso)

    fig = plt.figure("Compare t error of different paths")
    ax = Axes3D(fig)

    Y = np.arange(0, height, grid_reso)
    X = np.arange(0, width, grid_reso)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((gridX, gridY))

    for i in range(path_num):
        Rmat_error_mean_list = tvec_error_mean_list_AllPaths[i]
        fixed_path = fix_path_list[i]
        x_real = fixed_path[0,:]
        y_real = fixed_path[1,:]

        ix = (x_real - grid_reso / 2) / grid_reso
        ix = ix.astype(int)
        iy = (y_real - grid_reso / 2) / grid_reso
        iy = iy.astype(int)

        Z[ix,iy] = Rmat_error_mean_list

            # print "X.shape",X.shape
            # print "Y.shape",Y.shape
            # print "Z.shape",Z.shape
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma)

    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    # ax.yaxis.tick_left()  # remove right y-Ticks
    ax.set_xlabel('Y: world coordinate')
    ax.set_ylabel('X: world coordinate')
    ax.set_zlabel('t error(%)')
    plt.show()
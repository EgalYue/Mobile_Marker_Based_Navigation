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


def plotPath(fix_path, real_path, measured_path):
    plt.figure()
    # plt.axis([0, 6, 0, 3])
    plt.grid(True)  # set the grid

    # !!! In our case in the real world coordinate system we set x-axis as the plot-y, y-axis as the plot-x
    # ---------------------Fix path----------------------------------------
    x_fix = fix_path[0, :]
    y_fix = fix_path[1, :]
    plt.plot(y_fix, x_fix, color='black', label='Fix path')  # x,y  exchange position
    plt.scatter(y_fix, x_fix, c="black", marker='o')
    # ---------------------Real path----------------------------------------
    x_real = real_path[0, :]
    y_real = real_path[1, :]
    plt.plot(y_real, x_real, color='red', label='Real path')  # x,y  exchange position
    plt.scatter(y_real, x_real, c="red", marker='x')
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
    # plt.show() # Just use one plt.show in plotAll() method


def plotPositionError(fix_path, real_path, measured_path):
    tem_real = np.square(fix_path - real_path)
    real_path_error = np.sqrt(tem_real[0, :] + tem_real[1, :])

    tem_measured = np.square(fix_path - measured_path)
    measured_path_error = np.sqrt(tem_measured[0, :] + tem_measured[1, :])

    plt.figure()
    # ---------------------Real path----------------------------------------
    x_real = range(1, len(real_path_error) + 1)
    y_real = real_path_error
    plt.plot(x_real, y_real, color='red', label='Real path')
    plt.scatter(x_real, y_real, c="red", marker='x')
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
    # plt.show() # Just use one plt.show in plotAll() method


def plot_Rt_error(Rmat_error_list, tvec_error_list):
    fig = plt.figure('Pose estimation errors')
    ax_R_error = fig.add_subplot(121)
    ax_t_error = fig.add_subplot(122)
    # ------------------------ R error ---------------------------------
    x_R_error = np.arange(1, len(Rmat_error_list) + 1, 1)
    y_R_error = Rmat_error_list
    ax_R_error.xaxis.set_ticks(np.arange(0, len(Rmat_error_list) + 1, 1))
    ax_R_error.plot(x_R_error, y_R_error, color='blue', label='measured path')
    ax_R_error.scatter(x_R_error, y_R_error, c="red", marker='o')

    plt.sca(ax_R_error)
    ax_R_error.set_title("R error(" + u"\u00b0" + ")")
    ax_R_error.set_xlabel('Step')
    ax_R_error.set_ylabel('Angle(degree)')

    # -------------------------r error ---------------------------------
    x_t_error = np.arange(1, len(tvec_error_list) + 1, 1)
    y_t_error = tvec_error_list
    ax_t_error.xaxis.set_ticks(np.arange(0, len(tvec_error_list) + 1, 1))
    ax_t_error.plot(x_t_error, y_t_error, color='blue', label='measured path')
    ax_t_error.scatter(x_t_error, y_t_error, c="red", marker='o')

    plt.sca(ax_t_error)
    ax_t_error.set_title("t error(%)")
    ax_t_error.set_xlabel('Step')
    ax_t_error.set_ylabel('Percent')
    # plt.show()


def plotAll(fix_path, real_path, measured_path, Rmat_error_list, tvec_error_list):
    plotPath(fix_path, real_path, measured_path)
    plotPositionError(fix_path, real_path, measured_path)
    plot_Rt_error(Rmat_error_list, tvec_error_list)
    plt.show()


# ------------------------------------ Compare differnet paths at the same time -------------------------
def plotAllPaths_OneList(real_path_list):
    """
    Plot only one list
    :param real_path_list:
    :return:
    """
    plt.figure()
    path_num = len(real_path_list)

    # colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 0.9, path_num))))

    labels = []
    for i in range(0, path_num):
        x = real_path_list[i][0, :]
        y = real_path_list[i][1, :]
        plt.plot(y, x)
        plt.scatter(y, x, c="black", marker='o')
        labels.append(r'$$')

    plt.legend(labels, ncol=4, loc='upper center',
               bbox_to_anchor=[0.5, 1.1],
               columnspacing=1.0, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=True, shadow=True)


    ax = plt.gca()  # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax.yaxis.tick_left()  # remove right y-Ticks
    ax.set_title("Compare different paths")
    ax.set_xlabel('Y_W')
    ax.set_ylabel('X_W')
    plt.show() # Just use one plt.show in plotAll() method

def plotAllPaths_separate(fix_path_list, real_path_list, measured_path_list, Rmat_error_list, tvec_error_list):
    """
    Separately plot all paths
    :param fix_path_list:
    :param real_path_list:
    :param measured_path_list:
    :param Rmat_error_list:
    :param tvec_error_list:
    :return:
    """

    path_num = len(real_path_list)
    for i in range(0, path_num):
        newfig = plt.figure("Path " + str(i))

        plt.grid(True)  # set the grid

        # !!! In our case in the real world coordinate system we set x-axis as the plot-y, y-axis as the plot-x
        # ---------------------Fix path----------------------------------------
        fix_path = fix_path_list[i]
        x_fix = fix_path[0, :]
        y_fix = fix_path[1, :]
        plt.plot(y_fix, x_fix, color='black', label='Fix path')  # x,y  exchange position
        plt.scatter(y_fix, x_fix, c="black", marker='o')
        # ---------------------Real path----------------------------------------
        real_path = real_path_list[i]
        x_real = real_path[0, :]
        y_real = real_path[1, :]
        plt.plot(y_real, x_real, color='red', label='Real path')  # x,y  exchange position
        plt.scatter(y_real, x_real, c="red", marker='x')

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






    fig = plt.figure("Compare diffient paths")
    ax_fix = fig.add_subplot(131)
    ax_real = fig.add_subplot(132)
    ax_measured = fig.add_subplot(133)

    # colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 0.9, path_num))))

    #====================== Plot position error ============================
    fig2 = plt.figure("Position error (Euclidean distance)")
    ax_real_distance = fig2.add_subplot(121)
    ax_measured_distance = fig2.add_subplot(122)

    # ====================== Plot R t error ============================
    fig3 = plt.figure('Pose estimation errors')
    ax_R_error = fig3.add_subplot(121)
    ax_t_error = fig3.add_subplot(122)

    labels = []
    for i in range(0, path_num):
        x_fix = fix_path_list[i][0, :]
        y_fix = fix_path_list[i][1, :]
        l = ax_fix.plot(y_fix, x_fix)
        ax_fix.scatter(y_fix, x_fix, c="black", marker='o')

        colour = l[0].get_color()

        x_real = real_path_list[i][0, :]
        y_real = real_path_list[i][1, :]
        ax_real.plot(y_real, x_real, color= colour)
        ax_real.scatter(y_real, x_real, c="black", marker='o')

        x_measured = measured_path_list[i][0, :]
        y_measured = measured_path_list[i][1, :]
        ax_measured.plot(y_measured, x_measured, color= colour)
        ax_measured.scatter(y_measured, x_measured, c="black", marker='o')
        # labels.append(r'$$')

        # ====================== Plot position error ============================
        # ---------------------Real path----------------------------------------
        tem_real = np.square(fix_path_list[i] - real_path_list[i])
        real_path_error = np.sqrt(tem_real[0, :] + tem_real[1, :])
        x_real = np.arange(1, len(real_path_error) + 1, 1)
        y_real = real_path_error
        ax_real_distance.plot(x_real, y_real, color=colour, label='Real path')
        ax_real_distance.scatter(x_real, y_real, c="black", marker='o')
        # ---------------------Measured path----------------------------------------
        tem_measured = np.square(fix_path_list[i] - measured_path_list[i])
        measured_path_error = np.sqrt(tem_measured[0, :] + tem_measured[1, :])
        x_measured = np.arange(1, len(measured_path_error) + 1, 1)
        y_measured = measured_path_error
        ax_measured_distance.plot(x_measured, y_measured, color=colour, label='measured path')  # x,y  exchange position
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

    # plt.legend(labels, ncol=4, loc='upper center',
    #            bbox_to_anchor=[0.5, 1.1],
    #            columnspacing=1.0, labelspacing=0.0,
    #            handletextpad=0.0, handlelength=1.5,
    #            fancybox=True, shadow=True)


    #---------------- Fix path ------------------------------
    ax_fix.set_ylim(ax_fix.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax_fix.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax_fix.yaxis.tick_left()  # remove right y-Ticks
    # ax_fix.set_title("Compare different fix paths")
    ax_fix.set_xlabel('Fixed path: Y_W')
    ax_fix.set_ylabel('Fixed path: X_W')

    #---------------- Real path ------------------------------
    ax_real.set_ylim(ax_real.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax_real.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax_real.yaxis.tick_left()  # remove right y-Ticks
    # ax_real.set_title("Compare different real paths")
    ax_real.set_xlabel('Real path: Y_W')
    ax_real.set_ylabel('Real path: X_W')

    #---------------- Measured path ------------------------------
    ax_measured.set_ylim(ax_measured.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax_measured.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax_measured.yaxis.tick_left()  # remove right y-Ticks
    # ax_real.set_title("Compare different real paths")
    ax_measured.set_xlabel('Measured path: Y_W')
    ax_measured.set_ylabel('Measured path: X_W')

    # ====================== Plot position error =========================================
    # fontP = FontProperties()
    # fontP.set_size('small')
    # plt.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax_real_distance.set_title("Real paths: Position error (Euclidean distance)")
    ax_real_distance.set_xlabel('Step')
    ax_real_distance.set_ylabel('Error(m)')

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


def plotAllPaths(fix_path_list, real_path_list, measured_path_list, Rmat_error_list, tvec_error_list):
    """
    Plot all paths in one figure
    :param fix_path_list:
    :param real_path_list:
    :param measured_path_list:
    :param Rmat_error_list:
    :param tvec_error_list:
    :return:
    """
    fig = plt.figure("Compare diffient paths")
    ax_fix = fig.add_subplot(131)
    ax_real = fig.add_subplot(132)
    ax_measured = fig.add_subplot(133)

    path_num = len(real_path_list)
    # colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1.0, path_num))))

    #====================== Plot position error ============================
    fig2 = plt.figure("Position error (Euclidean distance)")
    ax_real_distance = fig2.add_subplot(121)
    ax_measured_distance = fig2.add_subplot(122)

    # ====================== Plot R t error ============================
    fig3 = plt.figure('Pose estimation errors')
    ax_R_error = fig3.add_subplot(121)
    ax_t_error = fig3.add_subplot(122)

    labels = []
    for i in range(0, path_num):
        x_fix = fix_path_list[i][0, :]
        y_fix = fix_path_list[i][1, :]
        l = ax_fix.plot(y_fix, x_fix, label=str(i))
        ax_fix.scatter(y_fix, x_fix, c="black", marker='o')

        colour = l[0].get_color()

        x_real = real_path_list[i][0, :]
        y_real = real_path_list[i][1, :]
        ax_real.plot(y_real, x_real, color= colour, label=str(i))
        ax_real.scatter(y_real, x_real, c="black", marker='o')

        x_measured = measured_path_list[i][0, :]
        y_measured = measured_path_list[i][1, :]
        ax_measured.plot(y_measured, x_measured, color= colour, label=str(i))
        ax_measured.scatter(y_measured, x_measured, c="black", marker='o')
        # labels.append(r'$$')

        # ====================== Plot position error ============================
        # ---------------------Real path----------------------------------------
        tem_real = np.square(fix_path_list[i] - real_path_list[i])
        real_path_error = np.sqrt(tem_real[0, :] + tem_real[1, :])
        x_real = np.arange(1, len(real_path_error) + 1, 1)
        y_real = real_path_error
        ax_real_distance.plot(x_real, y_real, color=colour, label=str(i))
        ax_real_distance.scatter(x_real, y_real, c="black", marker='o')
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

    # plt.legend(labels, ncol=4, loc='upper center',
    #            bbox_to_anchor=[0.5, 1.1],
    #            columnspacing=1.0, labelspacing=0.0,
    #            handletextpad=0.0, handlelength=1.5,
    #            fancybox=True, shadow=True)


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

    #---------------- Real path ------------------------------
    fontP = FontProperties()
    fontP.set_size('small')
    ax_real.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax_real.set_ylim(ax_real.get_ylim()[::-1])  # invert the axis
    # ax.xaxis.set_ticks(np.arange(0, 6, 0.1)) # set x-ticks
    ax_real.xaxis.tick_top()  # and move the X-Axis
    # ax.yaxis.set_ticks(np.arange(0, 3, 0.1)) # set y-ticks
    ax_real.yaxis.tick_left()  # remove right y-Ticks
    # ax_real.set_title("Compare different real paths")
    ax_real.set_xlabel('Real path: Y_W')
    ax_real.set_ylabel('Real path: X_W')

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
    ax_real_distance.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend outside of figure in matplotlib

    ax_real_distance.set_title("Real paths: Position error (Euclidean distance)")
    ax_real_distance.set_xlabel('Step')
    ax_real_distance.set_ylabel('Error(m)')

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



# =====================================Test=====================================================
# fix_path = np.array([[0.45, 0.45, 0.45, 0.45, 0.45],
#                      [1.65, 1.75, 1.85, 1.95, 2.05]])
# fix_path1 = np.array([[0.45, 0.46, 0.47, 0.48, 0.49],
#                      [1.65, 1.75, 1.85, 1.95, 2.05]])
# fix_path2 = np.array([[0.50, 0.51, 0.52, 0.53, 0.54],
#                      [1.65, 1.75, 1.85, 1.95, 2.05]])
# real_path_list = [fix_path1,fix_path2]
# measured_path_list = [fix_path1,fix_path2]
#
# plotAllPaths(real_path_list, measured_path_list)
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 11.05.18 16:14
@File    : plotA*.py
@author: Yue Hu
"""

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import changeAccMatSize as changeAccMatSize
import A_star_modified as astarm

def realPosTogridPos(x_real, y_real, grid_reso = 0.1):
    ix = int(round((x_real - grid_reso/2) /grid_reso))
    iy = int(round((y_real - grid_reso/2) /grid_reso))
    return ix,iy

#-----------------Initialization-------------------------------
grid_reso = 0.1
d_diagnoal = 1.4
d_straight = 1.0
grid_width = 6
grid_height = 3

startX_real = 1.25
startY_real = 2.05
startX_grid, startY_grid = realPosTogridPos(startX_real, startY_real, grid_reso=0.1)

goalX_real = 1.25
goalY_real = 4.05
goalX_grid, goalY_grid = realPosTogridPos(goalX_real, goalY_real, grid_reso=0.1)

plt.grid(True)
plt.axis("equal")

data = np.zeros((30,60)).T
plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)
plt.plot(startX_grid, startY_grid, "*k")
plt.plot(goalX_grid, goalY_grid, "*m")

path = astarm.aStar(sx = startX_real, sy = startY_real, gx = goalX_real, gy = goalY_real, d_diagnoal = d_diagnoal, d_straight = d_straight, grid_reso = grid_reso, grid_width = grid_width, grid_height = grid_height)
steps = path.shape[1]

for i in range(steps):
    x_real = path[0,i]
    y_real = path[1,i]

    ix, iy = realPosTogridPos(x_real, y_real, grid_reso=0.1)
    # print ix
    # print iy
    plt.plot(ix, iy, ".r")
    plt.pause(0.1)

plt.show()
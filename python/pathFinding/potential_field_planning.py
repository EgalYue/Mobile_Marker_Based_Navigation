"""

Potential Field based path planner

author: Yue Hu

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

and

https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/PotentialFieldPlanning


"""
import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import changeAccMatSize as changeAccMatSize
import os  # Read matrix form file
# Parameters
KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain

show_animation = False # TODO need to set

# grid_width = 60
# grid_height = 30
# real_width = 6 # [m]
# real_height = 3 # [m]

# Read accuracy matrix
cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../pathFinding/accuracyMatrix.txt', cur_path)
f = open(new_path, 'r')
l = [map(float, line.split(' ')) for line in f]
accuracy_mat = np.asarray(l)  # convert to matrix : 30 x 60
# TODO NEED TO SET
# accuracy_mat = changeAccMatSize.changeAccMatSize(accuracy_mat,6,12)


def calc_potential_field(gx, gy, ox, oy, grid_reso, rr, grid_width, grid_height):
    xw = grid_height
    yw = grid_width

    # calc each potential
    pmap = [[0.0 for i in range(grid_width)] for i in range(grid_height)]

    for ix in range(xw):
        x = ix * grid_reso + grid_reso / 2

        for iy in range(yw):
            y = iy * grid_reso + grid_reso / 2
            ug = calc_attractive_potential(x, y, gx, gy)
            # print "ug ",ug
            uo = calc_repulsive_potential(x, y, ox, oy, rr)
            # uf = ug + uo
            # pmap[ix][iy] = uf
            u_condNum = calc_repulsive_potential_condNum(ix, iy)
            # print "u_condNum ",u_condNum
            uf = ug + uo + u_condNum
            pmap[ix][iy] = uf

    return pmap


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i in range(len(ox)):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    if (len(ox) != 0):
        dq = np.hypot(x - ox[minid], y - oy[minid])
    else:
        dq = float("inf")

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1
        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0

def calc_repulsive_potential_condNum(x, y):
    return accuracy_mat[x,y]


def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def potential_field_planning(sx, sy, gx, gy, ox, oy, grid_reso, rr, grid_width, grid_height):
    # calc potential field
    pmap= calc_potential_field(gx, gy, ox, oy, grid_reso, rr, grid_width, grid_height)

    # search path
    d = np.hypot(sx - gx, sy - gy)
    d = round(d,4) # avoid float problem
    ix = round((sx - grid_reso/2) / grid_reso)
    iy = round((sy - grid_reso/2) / grid_reso)
    gix = round((gx - grid_reso/2) / grid_reso)
    giy = round((gy - grid_reso/2) / grid_reso)

    if show_animation:
        draw_heatmap(pmap)
        plt.plot(ix, iy, "*k")
        plt.plot(gix, giy, "*m")

    rx, ry = [sx], [sy]
    motion = get_motion_model()
    while d >= grid_reso:
        minp = float("inf")
        minix, miniy = -1, -1
        for i in range(len(motion)):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(pmap) or iny >= len(pmap[0]):
                p = float("inf")  # outside area
            else:
                p = pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
        ix = minix
        iy = miniy
        # print "ix iy",ix,iy
        xp = ix * grid_reso + grid_reso/2
        xp = round(xp, 4)  # avoid float problem
        yp = iy * grid_reso + grid_reso/2
        d = np.hypot(gx - xp, gy - yp)
        yp = round(yp, 4)  # avoid float problem
        # print "xp yp",xp,yp
        d = round(d, 4)  # avoid float problem
        # print "d",d
        rx.append(xp)
        ry.append(yp)

        if show_animation:
            plt.plot(ix, iy, ".r")
            plt.pause(0.01)

    if (gix * grid_reso + grid_reso/2) != gx or (giy * grid_reso + grid_reso/2) != gy:
        rx.append(gx)
        ry.append(gy)

    # print("Goal!!")

    return rx, ry


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def potentialField(sx = 2.15, sy = 2.05, gx = 2.15, gy = 3.05, ox = [], oy = [], grid_reso = 0.1, robot_radius = 0.5, grid_width = 6, grid_height = 3):
    """
    Entry, would be called in other class
    :param sx: start x position [m], sx= ix * reso + reso/2
    :param sy: start y positon [m], sy = iy * reso + reso/2
    :param gx: goal x position [m]
    :param gy: goal y position [m]
    :param ox: obstacle x position list [m]
    :param oy: obstacle y position list [m]
    :param grid_reso: default 0.1m
    :param robot_radius: default 0.5m
    :param grid_width: default 6m
    :param grid_height: default 3m
    :return:
    """

    # sx = 2.15  # start x position [m]
    # sy = 2.05  # start y positon [m]
    # gx = 2.15  # goal x position [m]
    # gy = 3.05  # goal y position [m]
    # grid_reso = 0.1  # potential grid size [m]
    # robot_radius = 0.5  # robot radius [m]
    # TODO set the obstacle
    # ox = []  # obstacle x position list [m]
    # oy = []  # obstacle y position list [m]

    width = int(grid_width / grid_reso)
    height = int(grid_height / grid_reso)
    # print "width,height",width,height

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    # path generation
    if sx < 0 or sy < 0 or gx < 0 or gy < 0:
        print "Error!!! The position can not be negative!"
    elif sx > grid_height or sy > grid_width or gx > grid_height or gy > grid_width:
        print "Error!!! The position is out of range!"
    else:
        rx, ry = potential_field_planning(
            sx, sy, gx, gy, ox, oy, grid_reso, robot_radius, width, height)
        # print "X position:\n", rx
        # print "Y position:\n", ry

        if show_animation:
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Potential field path planning')
            plt.show()
        paths_result = np.vstack((rx,ry))
        return paths_result

#==========================================================================================================
if __name__ == '__main__':
    print(__file__ + "------------ Start!!-----------")
    #called in other class
    potentialField(sx=1.25, sy=2.05, gx=1.25, gy=4.05, ox=[], oy=[], grid_reso=0.1,
                   robot_radius=0.5, grid_width=6, grid_height=3)
    print(__file__ + "------------ Done!!------------")
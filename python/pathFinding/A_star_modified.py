#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 11.05.18 14:32
@File    : A_star_modified.py
@author: Yue Hu

A Star algorithm with modified heuristic function based on condition number

"""
# from __future__ import division # set / as float!!!!
import sys
sys.path.append("..")
import numpy as np
import math
import os  # Read matrix form file
import saveMatrixToFile as smtf
import computeCondNum as ccn

# Read accuracy matrix
# cur_path = os.path.dirname(__file__)
# new_path = os.path.relpath('../pathFinding/accuracyMatrix.txt', cur_path)
# f = open(new_path, 'r')
# l = [map(float, line.split(' ')) for line in f]
# accuracy_mat = np.asarray(l)  # convert to matrix : 30 x 60
# accuracyMax = np.amax(accuracy_mat)
# # type: list
# # TODO Need to set
# # enlarge or  the condition number distribution
# accuracy_mat_scale = map(lambda x: np.interp(x,[0.0,accuracyMax],[0.0,accuracyMax*50]), accuracy_mat)
# # type: np.array
# accuracy_mat_scale= np.asarray(accuracy_mat_scale)
# # TODO Need to set
# # convert the undetected region as np.inf
# accuracy_mat_scale = np.where(accuracy_mat_scale == 0.0, np.inf, accuracy_mat_scale)

class Node:
    """
    Each position in matrix-map is a node
    """
    def __init__(self,x,y,father,g_value,h_value,f_value):
        self.x = x
        self.y = y
        self.father = father
        self.g_value = g_value
        self.h_value = h_value # Maybe we do not need this h_value
        self.f_value = f_value

    def equal(self,other):
        """
        If the x,y equal, we assume that the both Nodes are same
        :param other:
        :return:
        """
        if(self.x == other.x) and (self.y == other.y):
            return True
        else:
            return False

def create_gScore(width,height):
    """
    For each node, the cost of getting from the start node to that node.
    :param width: 60
    :param height: 30
    :return:
    """
    gScore = [[np.inf for i in range(width)] for i in range(height)]
    return gScore

def create_fScore(width,height):
    """
    For each node, the cost of getting from the start node to that node.
    :param width: 60
    :param height: 30
    :return:
    """
    fScore = [[np.inf for i in range(width)] for i in range(height)]
    return fScore

def heuristic_cost_estimate(start, goal,d_diagnoal,d_straight):
    """
    Diagonal distance
    h_diagonal(n) = min(abs(n.x - goal.x), abs(n.y - goal.y))
    h_straight(n) = (abs(n.x - goal.x) + abs(n.y - goal.y))
    h(n) = D_diagnoal * h_diagonal(n) + D_straight * (h_straight(n) - 2*h_diagonal(n)))
    :param start:
    :param goal:
    :return:
    """
    start_x = start.x
    start_y = start.y
    goal_x = goal.x
    goal_y = goal.y

    h_diagonal = min(np.abs(start_x - goal_x),np.abs(start_y - goal_y))
    h_straight = np.abs(start_x - goal_x) + np.abs(start_y - goal_y)
    h = d_diagnoal * h_diagonal + d_straight * (h_straight - 2 * h_diagonal)
    return h

# def heuristic_cost_estimate_modified(start, goal,d_diagnoal,d_straight):
#     """
#     Modified heuristic function
#     Adding condition number!
#
#     Diagonal distance
#     h_diagonal(n) = min(abs(n.x - goal.x), abs(n.y - goal.y))
#     h_straight(n) = (abs(n.x - goal.x) + abs(n.y - goal.y))
#     h(n) = D_diagnoal * h_diagonal(n) + D_straight * (h_straight(n) - 2*h_diagonal(n)))
#     :param start:
#     :param goal:
#     :return:
#     """
#     # TODO Adding condition number
#     start_x = start.x
#     start_y = start.y
#     goal_x = goal.x
#     goal_y = goal.y
#
#     h_diagonal = min(np.abs(start_x - goal_x),np.abs(start_y - goal_y))
#     h_straight = np.abs(start_x - goal_x) + np.abs(start_y - goal_y)
#     h_normal = d_diagnoal * h_diagonal + d_straight * (h_straight - 2 * h_diagonal)
#     h = h_normal + accuracy_mat_scale[start_x,start_y]
#     return h

def heuristic_cost_estimate_modified(start, goal,d_diagnoal,d_straight, grid_reso, width, height):
    """
    Modified heuristic function
    Adding condition number!

    Diagonal distance
    h_diagonal(n) = min(abs(n.x - goal.x), abs(n.y - goal.y))
    h_straight(n) = (abs(n.x - goal.x) + abs(n.y - goal.y))
    h(n) = D_diagnoal * h_diagonal(n) + D_straight * (h_straight(n) - 2*h_diagonal(n)))
    :param start:
    :param goal:
    :param d_diagnoal:
    :param d_straight:
    :param grid_reso:
    :param width: 60
    :param height: 30
    :return:
    """
    # TODO Adding condition number
    # In grid coordinate
    start_x = start.x
    start_y = start.y
    goal_x = goal.x
    goal_y = goal.y
    # In real coordinate
    x_w, y_w = gridPosToRealPos(start_x, start_y, grid_reso = grid_reso)

    h_diagonal = min(np.abs(start_x - goal_x),np.abs(start_y - goal_y))
    h_straight = np.abs(start_x - goal_x) + np.abs(start_y - goal_y)
    h_normal = d_diagnoal * h_diagonal + d_straight * (h_straight - 2 * h_diagonal)
    # h = h_normal + accuracy_mat_scale[start_x,start_y]
    # TODO enlarge
    condNum = ccn.getCondNum_camPoseInRealWord(x_w, y_w, grid_reso, width, height) * 50
    h = h_normal + condNum
    return h

def dist_between(current, neighbor,d_diagnoal,d_straight):
    """
    Diagonal distance
    h_diagonal(n) = min(abs(n.x - goal.x), abs(n.y - goal.y))
    h_straight(n) = (abs(n.x - goal.x) + abs(n.y - goal.y))
    h(n) = D_diagnoal * h_diagonal(n) + D_straight * (h_straight(n) - 2*h_diagonal(n)))
    :param start:
    :param goal:
    :return:
    """
    start_x = current.x
    start_y = current.y
    goal_x = neighbor.x
    goal_y = neighbor.y

    h_diagonal = min(np.abs(start_x - goal_x),np.abs(start_y - goal_y))
    h_straight = np.abs(start_x - goal_x) + np.abs(start_y - goal_y)
    h = d_diagnoal * h_diagonal + d_straight * (h_straight - 2 * h_diagonal)
    return h

def node_lowest_fScore(openSet):
    current = min(openSet, key=lambda o: o.f_value)
    return current

def current_in_cameFrom(current,cameFrom):
    for node in cameFrom:
        if(current.equal(node)):
            return True
    return False

def getNeighbors(current,width,height):
    """
     Get neighbors of current node
    :param current:
    :param width: default 60
    :param height: default 30
    :return:
    """
    x = current.x
    y = current.y
    neighbors = np.array([[x],[y]])
    if (x - 1) >= 0:
        t = np.array([[x - 1], [y]])
        neighbors = np.hstack((neighbors, t))
        if (y - 1) >= 0:
            t = np.array([[x - 1], [y - 1]])
            neighbors = np.hstack((neighbors,t))
        if (y + 1) < width:
            t = np.array([[x - 1], [y + 1]])
            neighbors = np.hstack((neighbors,t))
    if (y - 1) >= 0:
        t = np.array([[x], [y - 1]])
        neighbors = np.hstack((neighbors, t))
    if (y + 1) < width:
        t = np.array([[x], [y + 1]])
        neighbors = np.hstack((neighbors, t))
    if (x + 1) < height:
        t = np.array([[x + 1], [y]])
        neighbors = np.hstack((neighbors, t))
        if (y - 1) >= 0:
            t = np.array([[x + 1], [y - 1]])
            neighbors = np.hstack((neighbors,t))
        if (y + 1) < width:
            t = np.array([[x + 1], [y + 1]])
            neighbors = np.hstack((neighbors,t))
        neighbors = neighbors[:,1:]
    return neighbors

def neighbor_in_closedSet(neighbor,closedSet):
    for node in closedSet:
        if(neighbor.equal(node)):
            return True
    return False

def neighbor_not_in_openSet(neighbor,openSet):
    for node in openSet:
        if(neighbor.equal(node)):
            return False
    return True

def reconstruct_path(cameFrom, current):
    """
    Get path of A*
    """
    total_path = np.array([[current.x],[current.y]])
    while current_in_cameFrom(current,cameFrom):
        current = current.father
        node_x = current.x
        node_y = current.y
        node_pos = np.array([[node_x],[node_y]])
        total_path = np.hstack((total_path,node_pos))

    l1 = total_path[0,:]
    l1 = l1[::-1]
    l2 = total_path[1,:]
    l2 = l2[::-1]
    total_path = np.vstack((l1,l2))
    return total_path

def convertGridPathToReal(pathInGrid, sx, sy, gx, gy, grid_reso = 0.1):
    """
    convert the path in real to grid, e.g.   21 -> 2.15
    sx= ix * reso + reso/2
    :param pathReal:
    :param grid_reso: default 0.1[m]
    :return:
    """

    pathInReal = (pathInGrid * grid_reso + grid_reso / 2)
    stepNum = pathInReal.shape[1]
    # Replace head and tail
    pathInReal[:, 0] = [sx, sy]
    pathInReal[:, 0] = [sx, sy]
    pathInReal[:, stepNum - 1] = [gx, gy]
    pathInReal[:, stepNum - 1] = [gx, gy]

    return pathInReal

def realPosToGridPos(x_real, y_real, grid_reso = 0.1):
    ix = int(round((x_real - grid_reso/2) /grid_reso))
    iy = int(round((y_real - grid_reso/2) /grid_reso))
    return ix,iy

def gridPosToRealPos(x_grid, y_grid, grid_reso = 0.1):
    x_real = x_grid * grid_reso + grid_reso/2
    y_real = y_grid * grid_reso + grid_reso/2
    return x_real,y_real

def aStar(sx = 1.25, sy = 2.05, gx = 1.25, gy = 4.05, d_diagnoal = 14, d_straight = 10, grid_reso = 0.1, grid_width = 6, grid_height = 3):
    """
    A* algorithm
    A square 6m x 3m region
    :param startNode:
    :param goalNode:
    :param d_diagnoal:
    :param d_straight:
    :param grid_reso: default 0.1[m]
    :param grid_width: default 6[m]
    :param grid_height: default 3[m]
    :return:
    """
    width = int(grid_width/grid_reso)
    height = int(grid_height/grid_reso)
    #TODO
    A_sx, A_sy = realPosToGridPos(sx, sy, grid_reso = grid_reso)
    A_gx, A_gy = realPosToGridPos(gx, gy, grid_reso = grid_reso)
    startNode = Node(A_sx,A_sy,None,0,0,0)
    goalNode = Node(A_gx,A_gy,None,0,0,0)
    # The set of nodes already evaluated
    closedSet = set()
    # The set of currently discovered nodes that are not evaluated yet.
    openSet = set()
    # Initially, only the start node is known.
    openSet.add(startNode)
    # For each node, which node it can most efficiently be reached from.If a node can be reached from many nodes, cameFrom will eventually contain the most efficient previous step.
    cameFrom = []
    # For each node, the cost of getting from the start node to that node.
    gScore = create_gScore(width, height)
    start_x = startNode.x
    start_y = startNode.y
    # The cost of going from start to start is zero.
    startNode.g_value = 0
    gScore[start_x][start_y] = 0
    # For each node, the total cost of getting from the start node to the goal by passing by that node. That value is partly known, partly heuristic.
    fScore = create_fScore(width, height)
    # For the first node, that value is completely heuristic.
    # hStart = heuristic_cost_estimate_modified(startNode, goalNode, d_diagnoal, d_straight)
    hStart = heuristic_cost_estimate_modified(startNode, goalNode, d_diagnoal, d_straight, grid_reso, width, height)
    startNode.f_value = hStart
    fScore[start_x][start_y] = hStart

    while len(openSet) != 0:
        # current := the node in openSet having the lowest fScore[] value
        current = node_lowest_fScore(openSet)
        # If it is the item we want, retrace the path and return it
        if current.equal(goalNode):
            path = reconstruct_path(cameFrom, current) # path in real
            # print "path",path
            pathInReal = convertGridPathToReal(path, sx, sy, gx, gy, grid_reso = grid_reso) # path in grid
            # TODO------------------------------------------------
            Gmat = np.asarray(gScore)
            Fmat = np.asarray(fScore)
            smtf.saveMatToFile_G(Gmat)
            smtf.saveMatToFile_F(Fmat)
            # smtf. saveMatToFile_cond(accuracy_mat_scale)
            # TODO------------------------------------------------
            return pathInReal

        openSet.remove(current)
        closedSet.add(current)
        current_neighbors = getNeighbors(current, width, height)
        current_neighbors_num = current_neighbors.shape[1]
        # for neighbor in current_neighbors:
        for index in range(current_neighbors_num):
            [neighbor_x,neighbor_y] = current_neighbors[:,index]
            neighbor = Node(neighbor_x,neighbor_y,None,np.inf,np.inf,np.inf)
            if neighbor_in_closedSet(neighbor,closedSet):
                continue
            if neighbor_not_in_openSet(neighbor,openSet):	# Discover a new node
                openSet.add(neighbor)

            # The distance from start to a neighbor the "dist_between" function may vary as per the solution requirements.
            current_x = current.x
            current_y = current.y
            tentative_gScore = gScore[current_x][current_y] + dist_between(current, neighbor,d_diagnoal,d_straight)
            # print "[neighbor_x,neighbor_y]",neighbor_x,neighbor_y
            # print "tentative_gScore",tentative_gScore
            # neighbor_x = neighbor.x
            # neighbor_y = neighbor.y
            if tentative_gScore >= gScore[neighbor_x][neighbor_y]:
                continue		# This is not a better path.

            neighbor.father = current
            cameFrom.append(neighbor)
            gScore[neighbor_x][neighbor_y] = tentative_gScore
            neighbor.g_value = tentative_gScore
            # neighbor_f_value = gScore[neighbor_x][neighbor_y] + heuristic_cost_estimate_modified(neighbor, goalNode,d_diagnoal,d_straight)
            # TODO
            neighbor_f_value = gScore[neighbor_x][neighbor_y] + heuristic_cost_estimate_modified(neighbor, goalNode, d_diagnoal, d_straight, grid_reso, width, height)
            fScore[neighbor_x][neighbor_y] = neighbor_f_value
            neighbor.f_value = neighbor_f_value
    return False
# =================================Test========================================
# print create_gScore(2,2)
# -----------------------------------------------------------------------------
# start = Node(0,0,None,0,0)
# goal = Node(3,2,None,0,0)
# d_diagnoal = 14
# d_straight = 10
# print heuristic_cost_estimate(start, goal,d_diagnoal,d_straight)
# -----------------------------------------------------------------------------
# width = 5
# height = 5
# current = Node(1,4,None,0,0,0)
# print getNeighbors(current,width,height)
# -----------------------------------------------------------------------------
# width = 5
# height = 5
# d_diagnoal = 14
# d_straight = 10
# startNode = Node(20,20,None,0,0,0)
# goalNode = Node(20,30,None,0,0,0)
# path = aStar(sx = 1.25, sy = 2.05, gx = 1.25, gy = 4.05, d_diagnoal = 14, d_straight = 10, grid_reso = 0.1, grid_width = 6, grid_height = 3)
# print path
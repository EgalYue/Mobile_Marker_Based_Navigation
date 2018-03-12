#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 05.03.18 18:30
@File    : A_star.py
@author: Yue Hu

A Star algorithm
"""
import numpy as np
import math

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
    :param width:
    :param height:
    :return:
    """
    gScore = np.matrix(np.ones((width,height)) * np.inf)
    return gScore

def create_fScore(width,height):
    """
    For each node, the cost of getting from the start node to that node.
    :param width:
    :param height:
    :return:
    """
    fScore = np.matrix(np.ones((width,height)) * np.inf)
    return fScore

def heuristic_cost_estimate(start, goal,d_diagnoal,d_straight):
    """
    Manhattan distance
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

def dist_between(current, neighbor,d_diagnoal,d_straight):
    """
    Manhattan distance
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
    :return: set
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

def aStar(width,height,startNode,goalNode,d_diagnoal,d_straight):
    """
     A star algorithm
    :param start_x:
    :param start_y:
    :param end_x:
    :param end_y:
    :return:
    """
    # The set of nodes already evaluated
    closedSet = set()
    # The set of currently discovered nodes that are not evaluated yet.
    openSet = set()
    # Initially, only the start node is known.
    openSet.add(startNode)
    # For each node, which node it can most efficiently be reached from.If a node can be reached from many nodes, cameFrom will eventually contain the most efficient previous step.
    cameFrom = []
    # For each node, the cost of getting from the start node to that node.
    gScore = create_gScore(width,height)
    start_x = startNode.x
    start_y = startNode.y
    # The cost of going from start to start is zero.
    startNode.g_value = 0
    gScore[start_x,start_y] = 0
    # For each node, the total cost of getting from the start node to the goal by passing by that node. That value is partly known, partly heuristic.
    fScore = create_fScore(width,height)
    # For the first node, that value is completely heuristic.
    startNode.f_value = heuristic_cost_estimate(startNode, goalNode,d_diagnoal,d_straight)
    fScore[start_x,start_y] = heuristic_cost_estimate(startNode, goalNode,d_diagnoal,d_straight)
    while len(openSet) != 0:
        # current := the node in openSet having the lowest fScore[] value
        current = node_lowest_fScore(openSet)
        # If it is the item we want, retrace the path and return it
        if current.equal(goalNode):
            path = reconstruct_path(cameFrom, current)
            return path

        openSet.remove(current)
        closedSet.add(current)
        current_neighbors = getNeighbors(current,width,height)
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
            tentative_gScore = gScore[current_x,current_y] + dist_between(current, neighbor,d_diagnoal,d_straight)
            neighbor_x = neighbor.x
            neighbor_y = neighbor.y
            if tentative_gScore >= gScore[neighbor_x,neighbor_y]:
                continue		# This is not a better path.

            neighbor.father = current
            cameFrom.append(neighbor)
            gScore[neighbor_x,neighbor_y] = tentative_gScore
            neighbor.g_value = tentative_gScore
            neighbor_f_value = gScore[neighbor_x,neighbor_y] + heuristic_cost_estimate(neighbor, goalNode,d_diagnoal,d_straight)
            fScore[neighbor_x,neighbor_y] = neighbor_f_value
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
# startNode = Node(0,0,None,0,0,0)
# goalNode = Node(1,2,None,0,0,0)
# print aStar(width,height,startNode,goalNode,d_diagnoal,d_straight)
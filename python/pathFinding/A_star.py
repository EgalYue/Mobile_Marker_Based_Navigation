#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 05.03.18 18:30
@File    : A_star.py
@author: Yue Hu
"""
import numpy as np
import math

class Node:
    def __init__(self,x,y,father,g_value,h_value,f_value):
        self.x = x
        self.y = y
        self.father = father
        # TODO Value
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = f_value


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

def node_lowest_fScore(openSet):
    current = min(openSet, key=lambda o: o.f_value)
    return current

def reconstruct_path(cameFrom, current):
    # TODO Need to change
    total_path = [current]
    while current in cameFrom:
        current = cameFrom[current]
        total_path.append(current)
    return total_path

def getNeighbors(current):
    """

    :param current:
    :return: set
    """
    neighbors = set()
    # TODO NOT FINISH
    x = current.x
    y = current.y
    if (x - 1) >= 0 and (y - 1) >=0:
        # (self,x,y,father,g_value,h_value,f_value)




    return neighbors

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



def aStar(map,width,height,startNode,goalNode,d_diagnoal,d_straight):
    """

    :param map:
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
        if current == goalNode:
            return reconstruct_path(cameFrom, current)

        openSet.remove(current)
        closedSet.add(current)

        current_neighbors = getNeighbors(current)
        for neighbor in current_neighbors:
            if neighbor in closedSet:
                continue
            if neighbor not in openSet:	# Discover a new node
                openSet.add(neighbor)

            # The distance from start to a neighbor the "dist_between" function may vary as per the solution requirements.
            current_x = current.x
            current_y = current.y
            tentative_gScore = gScore[current] + dist_between(current, neighbor,d_diagnoal,d_straight)
            neighbor_x = neighbor.x
            neighbor_y = neighbor.y
            if tentative_gScore >= gScore[neighbor_x,neighbor_y]:
                continue		# This is not a better path.

            neighbor.father = current
            cameFrom.append(neighbor)
            gScore[neighbor_x,neighbor_y] = tentative_gScore
            neighbor.g_value = tentative_gScore
            neighbor_f_value = gScore[neighbor_x,neighbor_y] + heuristic_cost_estimate(neighbor, goalNode)
            fScore[neighbor_x,neighbor_y] = neighbor_f_value
            neighbor.f_value = neighbor_f_value

    return


            # =================================Test========================================
# print create_gScore(2,2)
# -----------------------------------------------------------------------------
# start = Node(0,0,None,0,0)
# goal = Node(3,2,None,0,0)
# d_diagnoal = 14
# d_straight = 10
# print heuristic_cost_estimate(start, goal,d_diagnoal,d_straight)
# -----------------------------------------------------------------------------
map = None
width = 5
height = 5
d_diagnoal = 14
d_straight = 10
startNode = Node(0,0,None,0,0,0)
goalNode = Node(3,2,None,0,0,0)
aStar(map,width,height,startNode,goalNode,d_diagnoal,d_straight)
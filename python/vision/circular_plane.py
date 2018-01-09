#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:04:45 2017

@author: lracuna
"""
import autograd.numpy as np
from rt_matrix import *
from conics import Circle
import matplotlib.pyplot as plt

class CircularPlane(object):
    """ Class for representing a 3D grid plane based on a point and a normal.

    Parameters:
        origin: 1x3 numpy array, position in world coordinates
        normal: 1x3 numpy array, normal vector of the plane in world coordinates
        radius: raidus of the circular plane (defines its limits)
        n: number of points defined inside the limits of the plane
        """
    def __init__(self, origin=np.array([0., 0., 0.]), normal = np.array([0, 0, 1]), radius=0.15, n = 4):
        self.origin = origin
        self.normal = normal
        self.n = n
        self.plane_points = None
        self.color = (0,0,0)
        self.angle = 0.
        self.R = np.eye(4)
        self.radius = radius
        self.size = (2.*radius,2.*radius)
        self.type = 'circular'
        self.circle = Circle((origin[0],origin[1]), radius)

    def clone(self):
        new_plane = CircularPlane()
        new_plane.origin = self.origin
        new_plane.normal = self.normal
        new_plane.size = self.size
        new_plane.n = self.n
        new_plane.color = self.color
        new_plane.angle = self.angle
        new_plane.R = self.R
        new_plane.radius = self.radius
        return new_plane

#    def get_corners(self):
#        x = self.size[0]/2.
#        y = self.size[1]/2.
#        corners = np.array([[x,  x, -x, -x],
#                            [y, -y, -y,  y],
#                            [0,  0,  0,  0],
#                            [1,  1,  1,  1]])
#        return corners

    def random(self, n = 4, r = 0.05, min_sep = 0.01):
        """
        n: amount of features on the plane
        r: radius of each feature (not including white borders for detection)
        min_sep: minimum distance from the border of the circle to assure detection
        """

        """
        This method is a little bit brute force approach. It is based on the Plane class.
        We first create a square of size RxR, in this square we create a grid, each cell of this
        grid has the size of a feature plus its min_separation between features.
        Then we randomly select a cell, if the center of the cell is inside the circle then
        we assigne a point to that cell. We continue with the same procedure for all the required poitns.
        """

        # create a equally spaced grid. Each cell with the size of a feature plus its min_separation
        cell_size = r + min_sep/2.0

        grid_size_RxR = int(round((self.radius*2)/cell_size))+1


        xy_range = range(grid_size_RxR)


        x_pos = np.array(xy_range, dtype=np.float64)*cell_size
        y_pos = np.array(xy_range, dtype=np.float64)*cell_size

        # center the grid
        x_pos = x_pos - x_pos[-1]/2.
        y_pos = y_pos - y_pos[-1]/2.
        #we create a boolean array
        # True means available
        available_grid = np.ones((grid_size_RxR, grid_size_RxR), dtype=bool)

        #Now we want to allocate the random points on the grid

        #we select an x,y position on the grid at random
        # and check if we already have a point there

        self.plane_points = np.zeros((4,n), dtype=np.float64)

        for i in range(n):
            from random import randint
            point_placed = False
            while not point_placed:
                x_grid_candidate = randint(0, grid_size_RxR-1)
                y_grid_candidate = randint(0, grid_size_RxR-1)

                if(available_grid[x_grid_candidate,y_grid_candidate]):
                    if self.inside_circle(x_pos[x_grid_candidate],y_pos[y_grid_candidate]):
                        self.plane_points[0,i] = x_pos[x_grid_candidate]
                        self.plane_points[1,i] = y_pos[y_grid_candidate]
                        self.plane_points[2,i] = 0
                        self.plane_points[3,i] = 1
                        available_grid[x_grid_candidate,y_grid_candidate] = False
                        point_placed = True
                    else:
                        available_grid[x_grid_candidate,y_grid_candidate] = False

        # translate
        self.plane_points[0] += self.origin[0]
        self.plane_points[1] += self.origin[1]
        self.plane_points[2] += self.origin[2]

    def inside_circle(self,x,y):
        """ Returns true if the coordinates defined by x,y are
        inside the circular plane """
        if np.sqrt(x**2+y**2) <= self.radius:
            return True
        else:
            return False

    def get_points(self):
        return np.copy(self.plane_points)

    def get_color(self):
        return self.color

    def set_origin(self, origin):
        self.origin = origin

    def set_normal(self, normal):
        self.normal = normal

    def set_radius(self, radius):
        self.radius = radius

    def set_color(self,color):
        self.color = color

    def rotate(self, axis, angle):
        """ rotate plane points around a given axis in world coordinates"""
        self.plane_points[0] -= self.origin[0]
        self.plane_points[1] -= self.origin[1]
        self.plane_points[2] -= self.origin[2]

        R = rotation_matrix(axis, angle)
        self.plane_points = np.dot(R, self.plane_points)

        # return translation
        self.plane_points[0] += self.origin[0]
        self.plane_points[1] += self.origin[1]
        self.plane_points[2] += self.origin[2]

    def rotate_x(self,angle):
        self.rotate(np.array([1,0,0],dtype=np.float32), angle)

    def rotate_y(self,angle):
        self.rotate(np.array([0,1,0],dtype=np.float32), angle)

    def rotate_z(self,angle):
        self.rotate(np.array([0,0,1],dtype=np.float32), angle)

    def plot_points(self):
        # show Image
        # plot projection
        #plt.figure("Plane points")
        plt.plot(self.plane_points[0],self.plane_points[1],'.', color = 'blue')
        #we add a key point to help us see orientation of the points
        plt.plot(self.plane_points[0,0],self.plane_points[1,0],'.',color = 'red')
        #plt.xlim(0,self.img_width)
        #plt.ylim(0,self.img_height)
        #plt.gca().invert_yaxis()
        plt.show()

    def plot_plane(self):
        circle = plt.Circle(self.origin, self.radius, color = self.color, fill = False, linestyle='--')
        plt.gcf().gca().add_artist(circle)
        #plt.gcf().gca().set_aspect('equal', 'datalim')



#from vision.circular_plane import CircularPlane
#cp = CircularPlane()
#cp.random(n=60)
#cp.plot_points()
#cp.plot_plane()






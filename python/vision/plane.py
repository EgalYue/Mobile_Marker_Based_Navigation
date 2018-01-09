#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:04:45 2017

@author: lracuna
"""
import autograd.numpy as np
from rt_matrix import *
import matplotlib.pyplot as plt

class Plane(object):
    """ Class for representing a 3D grid plane based on a point and a normal."""
    def __init__(self, origin=np.array([0., 0., 0.]), normal = np.array([0, 0, 1]), size=(0.4,0.4), n = (2,2)):
        self.origin = origin
        self.normal = normal
        self.size = size
        self.nx = n[0]
        self.ny = n[1]
        self.plane_points = None
        self.color = (1,0,0)
        self.angle = 0.
        self.R = np.eye(4)
        self.type = 'rectangular'

    def clone(self):
        new_plane = Plane()

        new_plane.origin = self.origin
        new_plane.normal = self.normal
        new_plane.size = self.size
        new_plane.nx = self.nx
        new_plane.ny = self.ny
        new_plane.color = self.color
        new_plane.angle = self.angle
        new_plane.R = self.R
        return new_plane

    def get_corners(self):
        x = self.size[0]/2.
        y = self.size[1]/2.
        corners = np.array([[x,  x, -x, -x],
                            [y, -y, -y,  y],
                            [0,  0,  0,  0],
                            [1,  1,  1,  1]])
        return corners
    def random(self, n = 4, r = 0.05, min_sep = 0.01):
        """
        n: ammount of features on the plane
        r: radius of each feature (not inluding white borders for detection)
        min_sep: minimum distance from the border of the circle to assure detection
        """
        # create a equally spaced grid. Each cell with the size of a feature plus its min_separation
        cell_size = r + min_sep/2.0

        grid_size_x = int(round(self.size[0]/cell_size))+1
        grid_size_y = int(round(self.size[1]/cell_size))+1

        x_range = range(grid_size_x)
        y_range = range(grid_size_y)

        x_pos = np.array(x_range, dtype=np.float64)*cell_size
        y_pos = np.array(y_range, dtype=np.float64)*cell_size

        # center the grid
        x_pos = x_pos - x_pos[-1]/2.
        y_pos = y_pos - y_pos[-1]/2.

        #we create a boolean array
        # True means available
        available_grid = np.ones((grid_size_x, grid_size_y), dtype=bool)

        #Now we want to allocate the random points on the grid

        #we select an x,y position on the grid at random
        # and check if we already have a point there

        self.plane_points = np.zeros((4,n), dtype=np.float64)

        for i in range(n):
            from random import randint
            point_placed = False
            while not point_placed:
                x_grid_candidate = randint(0, grid_size_x-1)
                y_grid_candidate = randint(0, grid_size_y-1)

                if(available_grid[x_grid_candidate,y_grid_candidate]):
                    self.plane_points[0,i] = x_pos[x_grid_candidate]

                    self.plane_points[1,i] = y_pos[y_grid_candidate]
                    self.plane_points[2,i] = 0
                    self.plane_points[3,i] = 1
                    available_grid[x_grid_candidate,y_grid_candidate] = False
                    point_placed = True

        # translate
        self.plane_points[0] += self.origin[0]
        self.plane_points[1] += self.origin[1]
        self.plane_points[2] += self.origin[2]



    def uniform(self):
        #we create a plane in the x-y plane
        """wx: plane width
           wy: plane heigth
           nx: ammount of points in x
           ny: ammount of points in y
        """
        wx = self.size[0]/2.0
        wy = self.size[1]/2.0
        nx = self.nx
        ny = self.ny

        x_range = np.linspace(-wx,wx,nx, dtype=np.float64)
        y_range = np.linspace(-wy,wy,ny, dtype=np.float64)
        xx, yy = np.meshgrid(x_range, y_range)



#        # create x,y
#        x_range = range(int(round(self.grid_size[0]/self.grid_step)))
#        y_range = range(int(round(self.grid_size[1]/self.grid_step)))
#        xx, yy = np.meshgrid(x_range, y_range)
#        # center the plane
#        xx = (xx.astype(np.float32))*self.grid_step - (x_range[-1]*self.grid_step/2.)
#        yy = (yy.astype(np.float32))*self.grid_step - (y_range[-1]*self.grid_step/2.)
#
        # calculate corresponding z
        hh = np.ones_like(xx, dtype=np.float32)
        zz = np.zeros_like(xx, dtype=np.float32)

        self.plane_points = np.array([xx.ravel(),yy.ravel(),zz.ravel(), hh.ravel()], dtype=np.float64)

        self.plane_points_basis = self.plane_points

#        #we rotate the plane around the normal axis by the given angle
#
#        if self.angle!=0.:
#            self.R = rotation_matrix(self.normal, self.angle)
#            self.plane_points = dot(self.R, self.plane_points)
#
#        #we now align the plane to the required normal
#
#        current_normal = array([1,0,0])
#        desired_normal = self.normal
#        if not (current_normal == desired_normal).all():
#            self.R = R = rotation_matrix_from_two_vectors(current_normal,desired_normal)
#            self.plane_points = dot(self.R, self.plane_points)
#
#
##
        # translate
        self.plane_points[0] += self.origin[0]
        self.plane_points[1] += self.origin[1]
        self.plane_points[2] += self.origin[2]


    def uniform_with_distortion(self, mean = 0, sd = 2):
        """ We modify the values of the uniform point distribution by adding
        a Gaussian noise
        mean: zero mean
        sd: pixels of standard deviation
        """
        self.uniform()
        gaussian_noise = np.random.normal(mean,sd,(2,self.plane_points.shape[1]))
        self.plane_points[:2,:] += gaussian_noise

        min_max_x = self.size[0]/2.
        min_max_y = self.size[1]/2.

        self.plane_points[0,:] = np.clip(self.plane_points[0,:],-min_max_x,min_max_x)
        self.plane_points[1,:] = np.clip(self.plane_points[1,:],-min_max_y,min_max_y)

    def get_points(self):
        return np.copy(self.plane_points)

    def get_points_basis(self):
        return self.plane_points_basis

    def get_mesh(self):
        return self.xx, self.yy, self.zz

    def get_color(self):
        return self.color

    def set_origin(self, origin):
        self.origin = origin

    def set_normal(self, normal):
        self.normal = normal

    def set_grid(self, grid_x, grid_y, grid_step):
        self.size = (grid_x, grid_y)
        self.grid_step = grid_step

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
        plt.figure("Plane points")
        plt.plot(self.plane_points[0],self.plane_points[1],'.', color = 'blue')
        #we add a key point to help us see orientation of the points
        plt.plot(self.plane_points[0,0],self.plane_points[1,0],'.',color = 'red')
        #plt.xlim(0,self.img_width)
        #plt.ylim(0,self.img_height)
        #plt.gca().invert_yaxis()
        plt.show()












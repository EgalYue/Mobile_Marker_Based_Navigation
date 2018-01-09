#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:04:45 2017

@author: lracuna
"""
from plane import Plane
from rt_matrix import *
import numpy as np
class Screen(Plane):
    """ Class for representing a 3D LCD screen"""
    def __init__(self, width = 0.5184, height = 0.324, diagonal_size = 0.61, pixel_pitch = 0.270, resolution = (1920,1200), aspect_ratio = (16,10), curvature_radius=4.0):
        #In meters
        self.width = width
        self.height = height
        self.diagonal_size = diagonal_size
        #In millimeters
        self.pixel_pitch = pixel_pitch
        #In pixels
        self.resolution = resolution
        #unitless
        self.aspect_ratio = aspect_ratio    
        #curvature radius in meters | a value of 0.0 means a plane
        self.curvature_radius = curvature_radius
        # TODO Yue: Plane dont have attribute grid_size and grid_step
        # Plane.__init__(self,grid_size=(1,2), grid_step = pixel_pitch/1000)

            
        
        
    
    def set_dimensions(self,width,height):
        """ Physicial dimensions in meters """
        self.width = width
        self.height = height
        self.grid_size = (self.width, self.height)
    
    def set_resolution_pixels(self, x,y):        
        self.resolution = (x,y)
    
    def set_pixel_pitch(self, pixel_pitch):
        """ In milimiters """
        self.pixel_pitch = pixel_pitch
        self.grid_step = pixel_pitch
    
    def update(self):
        if self.curvature_radius == 0:
            super(Screen, self).update()
        else:
            self.update_curved()
            print("curved screen")
            
    
    def update_curved(self):
        #we create a plane in the x-y plane        
        # create x,y
        x_range = range(int(round(self.grid_size[0]/self.grid_step)))
        y_range = range(int(round(self.grid_size[1]/self.grid_step)))
        xx, yy = np.meshgrid(x_range, y_range)
        # center the plane
        xx = (xx.astype(np.float32))*self.grid_step - (x_range[-1]*self.grid_step/2.)
        yy = (yy.astype(np.float32))*self.grid_step - (y_range[-1]*self.grid_step/2.)
        
        # calculate corresponding z        
        teta = np.arccos((xx)/(self.curvature_radius/2.0))
        zz = self.curvature_radius - self.curvature_radius*np.sin(teta)      
        
        
        hh = np.ones_like(xx, dtype=np.float32)
        self.plane_points = np.array([xx.ravel(),yy.ravel(),zz.ravel(), hh.ravel()], dtype=np.float32)         
        self.plane_points_basis = self.plane_points
#       
##        
        # translate        
        self.plane_points[0] += self.origin[0]
        self.plane_points[1] += self.origin[1]
        self.plane_points[2] += self.origin[2]
        
        
        self.xx = xx
        self.yy = yy
        self.zz = zz



# r = 2.0
# screen_size = 2.0
#
# grid_size = screen_size
# grid_step = 0.01
# x_range = range(int(round(grid_size/grid_step)))
# x = array(x_range).astype(np.float32)*grid_step - (x_range[-1]*grid_step/2.)
# teta = np.arccos((x)/(r/2.0))
# y = cy - r*np.sin(teta)
# plt.plot(x,y)
# plt.xlim(-2,2)
# plt.ylim(0,4)
#
# plt.figure()
# s1 = Screen()
# s1.grid_size = (1.5,2.0)
# s1.curvature_radius = 2.0
#
# s1.update_curved()
# plt.plot(s1.xx[0,:], s1.zz[0,:])
# plt.xlim(-2,2)
# plt.ylim(0,4)

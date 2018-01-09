#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:04:45 2017

@author: lracuna
"""
import autograd.numpy as np
from rt_matrix import *
import matplotlib.pyplot as plt
from conics import Conic

class Circle(Conic):
  #TODO Yue: There is already a Circle class in conics.py and in this class "project" method is not finish
  """ Class for representing a Circle on a plane on a center point and a radius."""
  def __init__(self, origin=(0.,0.), r = 0.1):
    self.origin = origin # Center of the circle
    self.r = r # Radiuus of the circle
    self.color = 'black'
    self.calculate_conic_matrix()

  def clone(self):
    new_circle = Circle()
    new_circle.origin = self.origin
    new_circle.r = self.r
    return new_circle

  def get_points(self):
    return None

  def get_color(self):
    return self.color

  def set_origin(self, origin):
    self.origin = origin

  def set_r(self, r):
    self.r = r

  def set_color(self,color):
    self.color = color

  def plot(self):
    circle = plt.Circle(self.origin, self.r, color = self.color)
    # show Image
    # plot projection
    #Asume that we have an existing Figure
    plt.gcf().gca().add_artist(circle)
    plt.gcf().gca().set_aspect('equal', 'datalim')
    plt.show()

  def contour(self):
    a = self.a
    b = self.b
    c = self.c
    d = self.d
    e = self.e
    f = self.f
    x = np.linspace(-9, 9, 400)
    y = np.linspace(-5, 5, 400)
    x, y = np.meshgrid(x, y)
    assert b**2 - 4*a*c < 0
    plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors='r', linestyles = 'dashed')
    #plt.gcf().gca().set_aspect('equal', 'datalim')
    plt.show()


  def calculate_conic_matrix(self):
    xo = self.origin[0]
    yo = self.origin[1]
    r = self.r
    self.Aq = np.mat([[1, 0, -xo],
                     [0, 1, -yo],
                     [-xo, -yo, xo**2+yo**2-r**2]])

    self.a = self.Aq[0,0]
    self.c = self.Aq[1,1]
    self.f = self.Aq[2,2]
    self.b = self.Aq[0,1]*2.
    self.d = self.Aq[0,2]*2.
    self.e = self.Aq[1,2]*2.

    return self.Aq


  def calculate_center(self):
    a = self.a
    b = self.b
    c = self.c
    d = self.d
    e = self.e
    f = self.f
    #https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
    Xc = (b*e-2*c*d)/(4*a*c-b**2)
    Yc = (d*b-2*a*e)/(4*a*c-b**2)

    return Xc, Yc

  def project(self,H):
    H = np.mat(H)
    Hinv = np.linalg.inv(H)
    self.Q = (Hinv.T)*self.Aq*Hinv
    return Q



















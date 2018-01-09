#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:04:45 2017

@author: lracuna
"""
import autograd.numpy as np
from rt_matrix import *  # TODO Yue: from vision.rt_matrix import *
import matplotlib.pyplot as plt


class Conic(object):
    """ Class for representing a conic on a plane

  Aq = [A, B/2, D/2]
       [B/2, C, E/2]
       [D/2, E/2, F]
  """

    def __init__(self, a=1, b=0, c=1, d=0, e=0, f=1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.r = None
        self.Aq = None
        self.update_conic_matrix()
        self.color = 'black'

    def update_conic_matrix(self):
        self.Aq = np.mat([[self.a, self.b / 2., self.d / 2.],
                          [self.b / 2., self.c, self.e / 2.],
                          [self.d / 2., self.e / 2., self.f]])

    def set_color(self, color):
        self.color = color


class Ellipse(Conic):
    def __init__(self, center=(0., 0.), semi_major_axis=2., semi_minor_axis=1., angle=90.):
        self.center = center  # Center of the circle
        a = semi_major_axis
        b = semi_minor_axis
        xc = center[0]
        yc = center[1]
        angle = np.deg2rad(angle)

        # https://en.wikipedia.org/wiki/Ellipse
        A = a ** 2 * (np.sin(angle) ** 2) + b ** 2 * (np.cos(angle) ** 2)
        B = 2 * (b ** 2 - a ** 2) * np.sin(angle) * np.cos(angle)
        C = a ** 2 * (np.cos(angle) ** 2) + b ** 2 * (np.sin(angle) ** 2)
        D = -2 * A * xc - B * yc
        E = -B * xc - 2 * C * yc
        F = A * xc ** 2 + B * xc * yc + C * yc ** 2 - (a ** 2) * (b ** 2)

        self.a = A
        self.b = B
        self.c = C
        self.d = D
        self.e = E
        self.f = F

        self.update_conic_matrix()

    def calculate_center(self):
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        e = self.e
        f = self.f
        # https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
        xc = (b * e - 2 * c * d) / (4 * a * c - b ** 2)
        yc = (d * b - 2 * a * e) / (4 * a * c - b ** 2)
        return xc, yc

    def major_axis_length(self):
        # https://math.stackexchange.com/questions/616645/determining-the-major-minor-axes-of-an-ellipse-from-general-form
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        e = self.e
        f = self.f
        #TODO Yue: following form is wrong, just use the form in https://en.wikipedia.org/wiki/Ellipse
        # q = 64. * (f * (4 * a * c - b ** 2) - a * e ** 2 + b * d * e - c * d ** 2) / ((4 * a * c - b ** 2) ** 2)
        # rmax = 1. / 8 * np.sqrt(np.abs(q) * np.sqrt(b ** 2 + (a - c) ** 2) - 2 * q * (a + c))
        rmax = -(np.sqrt(2*(a * e**2 + c * d**2 - b * d * e + (b**2 - 4 * a * c) * f)*(a + c + (np.sqrt((a - c)**2 + b**2))))) / (b**2 - 4 * a * c)
        return rmax

    def contour(self, grid_size=10):
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        e = self.e
        f = self.f
        ma = self.major_axis_length()
        xc, yc = self.calculate_center()
        x = np.linspace(-ma * 2 + xc, ma * 2 + xc, grid_size)
        y = np.linspace(-ma * 2 + yc, ma * 2 + yc, grid_size)
        x, y = np.meshgrid(x, y)
        # assert b**2 - 4*a*c < 0
        plt.contour(x, y, (a * x ** 2 + b * x * y + c * y ** 2 + d * x + e * y + f), [0], colors='k',
                    linestyles='dashed')
        plt.gcf().gca().set_aspect('equal', 'datalim')
        plt.show()


class Circle(Ellipse):
    """ Class for representing a Circle on a plane on a center point and a radius."""

    def __init__(self, center=(0., 0.), r=0.1):
        self.center = center  # Center of the circle
        self.r = r  # Radiuus of the circle
        self.color = 'black'
        self.update_conic_matrix()

    def clone(self):
        new_circle = Circle()
        new_circle.center = self.center
        new_circle.r = self.r
        return new_circle

    def get_points(self):
        return None

    def get_color(self):
        return self.color

    def set_origin(self, center):
        self.center = center

    def set_r(self, r):
        self.r = r

    def set_color(self, color):
        self.color = color

    def plot(self):
        circle = plt.Circle(self.center, self.r, color=self.color)
        # show Image
        # plot projection
        # Asume that we have an existing Figure
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
        x = np.linspace(-self.r * 2, self.r * 2, 100)
        y = np.linspace(-self.r * 2, self.r * 2, 100)
        x, y = np.meshgrid(x, y)
        assert b ** 2 - 4 * a * c < 0
        plt.contour(x, y, (a * x ** 2 + b * x * y + c * y ** 2 + d * x + e * y + f), [0], colors='k')
        plt.gcf().gca().set_aspect('equal', 'datalim')
        plt.show()

    def update_conic_matrix(self):
        xo = self.center[0]
        yo = self.center[1]
        r = self.r
        self.Aq = np.mat([[1, 0, -xo],
                          [0, 1, -yo],
                          [-xo, -yo, xo ** 2 + yo ** 2 - r ** 2]])

        self.a = self.Aq[0, 0]
        self.c = self.Aq[1, 1]
        self.f = self.Aq[2, 2]
        self.b = self.Aq[0, 1] * 2.
        self.d = self.Aq[0, 2] * 2.
        self.e = self.Aq[1, 2] * 2.

        return self.Aq

    def plot(self):
        circle = plt.Circle(self.center, self.r, color=self.color)
        # show Image
        # plot projection
        # Asume that we have an existing Figure
        plt.gcf().gca().add_artist(circle)
        plt.gcf().gca().set_aspect('equal', 'datalim')
        plt.show()

    def project(self, H):
        H = np.mat(H)
        Hinv = np.linalg.inv(H)
        Q = (Hinv.T) * self.Aq * Hinv

        projected_circle = Ellipse()
        projected_circle.a = Q[0, 0]
        projected_circle.c = Q[1, 1]
        projected_circle.f = Q[2, 2]
        projected_circle.b = Q[0, 1] * 2.
        projected_circle.d = Q[0, 2] * 2.
        projected_circle.e = Q[1, 2] * 2.
        projected_circle.Aq = Q
        projected_circle.r = self.r

        return projected_circle
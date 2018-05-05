#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 05.05.18 12:33
@File    : test_surf.py
@author: Yue Hu
"""
# --------------!!! important!!!----------
# from traits.etsconfig.api import ETSConfig
# ETSConfig.toolkit = 'wx'
#-----------------------------------------
import numpy
from mayavi import mlab

def test_surf():
    """Test surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        sin, cos = numpy.sin, numpy.cos
        return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

    x, y = numpy.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
    mlab.surf(x, y, f)
    mlab.show()
    #cs = contour_surf(x, y, f, contour_z=0)

#---------------------- Test -----------------------------
test_surf()
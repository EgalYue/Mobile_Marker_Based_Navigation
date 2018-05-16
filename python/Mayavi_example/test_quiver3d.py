#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 16.05.18 20:37
@File    : test_quiver3d.py
@author: Yue Hu
"""
# --------------!!! important!!!----------
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'wx'
#-----------------------------------------
import numpy
from mayavi import mlab

# def test_quiver3d():
#     x, y, z = numpy.mgrid[-2:3, -2:3, -2:3]
#     r = numpy.sqrt(x ** 2 + y ** 2 + z ** 4)
#     u = y * numpy.sin(r) / (r + 0.001)
#     v = -x * numpy.sin(r) / (r + 0.001)
#     w = numpy.zeros_like(z)
#     obj = quiver3d(x, y, z, u, v, w, line_width=3, scale_factor=1)
#     return obj

def test_quiver3d():
    x, y, z = numpy.mgrid[-2:3, -2:3, -2:3]
    r = numpy.sqrt(x ** 2 + y ** 2 + z ** 4)
    u = y * numpy.sin(r) / (r + 0.001)
    v = -x * numpy.sin(r) / (r + 0.001)
    w = numpy.zeros_like(z)
    mlab.quiver3d(x, y, z, u, v, w, line_width=3, scale_factor=1)
    mlab.show()

test_quiver3d()
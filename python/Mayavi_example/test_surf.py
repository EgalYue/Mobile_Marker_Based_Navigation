#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 05.05.18 12:33
@File    : test_surf.py
@author: Yue Hu
"""
# --------------!!! important!!!----------
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'wx'
#-----------------------------------------
import numpy as np
from mayavi import mlab

def test_surf():
    """Test surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        sin, cos = np.sin, np.cos
        return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

    x, y = np.mgrid[0:3:0.1, 0:6:0.1]
    print x.shape
    print y.shape
    z = np.zeros((30,60))
    z[1:20,10] = 9
    mlab.surf(x, y, z)
    mlab.show()
    #cs = contour_surf(x, y, f, contour_z=0)

#---------------------- Test -----------------------------
test_surf()
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 11.04.18 19:35
@File    : saveMatrixToFile.py
@author: Yue Hu
"""

import numpy as np


def saveMatToFile(matrix):
  mat = np.matrix(matrix)
  with open('accuracyMatrix.txt', 'wb') as f:
    for line in mat:
      np.savetxt(f, line, fmt='%d')


#=================Test=============================
# matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
# saveMatToFile(matrix)
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
      np.savetxt(f, line, fmt='%f') # Store the data as float instead of Integer %d

def saveMatToFile_G(matrix):
  mat = np.matrix(matrix)
  with open('pathGscore.txt', 'wb') as f:
    for line in mat:
      np.savetxt(f, line, fmt='%f')  # Store the data as float instead of Integer %d

def saveMatToFile_F(matrix):
  mat = np.matrix(matrix)
  with open('pathFscore.txt', 'wb') as f:
    for line in mat:
      np.savetxt(f, line, fmt='%f')  # Store the data as float instead of Integer %d

def saveMatToFile_cond(matrix):
  mat = np.matrix(matrix)
  with open('pathCondscore.txt', 'wb') as f:
    for line in mat:
      np.savetxt(f, line, fmt='%f')  # Store the data as float instead of Integer %d
#=================Test=============================
# matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
# saveMatToFile(matrix)
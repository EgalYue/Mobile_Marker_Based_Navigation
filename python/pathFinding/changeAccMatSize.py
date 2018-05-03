#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 03.05.18 00:57
@File    : changeAccMatSize.py
@author: Yue Hu
"""
from __future__ import division # set / as float!!!!
import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import os  # Read matrix form file

cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../CondNum_TheoryValidation_newAccMat/accuracyMatrix.txt', cur_path)
f = open(new_path, 'r')
l = [map(float, line.split(' ')) for line in f]
accuracy_mat = np.asarray(l)  # convert to matrix : 30 x 60

def changeAccMatSize(accuracy_mat, newRow, newCol):
    oldRow = accuracy_mat.shape[0]
    oldCol = accuracy_mat.shape[1]
    nRow = int(oldRow / newRow)
    nCol = int(oldCol / newCol)
    result = np.zeros((newRow,newCol),dtype=float)
    for i in range(newRow):
        for j in range(newCol):
            temMat = accuracy_mat[i*nRow:(i+1)*nRow - 1,j*nCol:(j+1)*nCol - 1]
            result[i,j] = temMat.sum() / (temMat.size)

    return result

#============================== Test =====================================
# print changeAccMatSize(accuracy_mat, 30, 60).shape
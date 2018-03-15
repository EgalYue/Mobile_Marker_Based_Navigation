#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 09.02.18 13:43
@File    : Rt_matrix_from_euler_zyx.py
@author: Yue Hu
"""
import numpy as np
from math import cos, sin

def R_matrix_from_euler_zyx(alpha,beta,gamma):
  # TODO Error
  """
  A=BCD. Z -> Y -> X
  :param alpha:
  :param beta:
  :param gamma:
  :return:
  """
  R = np.eye(4, dtype=np.float32)
  R[0,0] = cos(beta) * cos(alpha)
  R[1,0] = sin(gamma) * sin(beta) * cos(alpha) - cos(gamma) * sin(alpha)
  R[2,0] = cos(gamma) * sin(beta) * cos(alpha) + sin(gamma) * sin(alpha)

  R[0,1] = cos(beta) * sin(alpha)
  R[1,1] = sin(gamma) * sin(beta) * sin(alpha) + cos(gamma) * cos(alpha)
  R[2,1] = cos(gamma) * sin(beta) * sin(alpha) - sin(gamma) * cos(alpha)

  R[0,2] = -sin(beta)
  R[1,2] = sin(gamma) * cos(beta)
  R[2,2] = cos(gamma) * cos(beta)

  return R

# ===================Test==========================
print R_matrix_from_euler_zyx(0,np.deg2rad(-120),0)
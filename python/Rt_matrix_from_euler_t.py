#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:48:08 2017

@author: lracuna
"""
import numpy as np
from math import cos, sin

def R_matrix_from_euler_t(alpha,beta,gamma):
  """
  A=BCD. Z -> X -> Z
  1. the first rotation is by an angle phi about the z-axis using D,
  2. the second rotation is by an angle theta in [0,pi] about the former x-axis (now x^') using C, and
  3. the third rotation is by an angle psi about the former z-axis (now z^') using B.
  :param alpha:
  :param beta:
  :param gamma:
  :return:
  """
  R = np.eye(4, dtype=np.float32)
  R[0,0]=cos(alpha)*cos(gamma)-cos(beta)*sin(alpha)*sin(gamma)
  R[1,0]=cos(gamma)*sin(alpha)+cos(alpha)*cos(beta)*sin(gamma)
  R[2,0]=sin(beta)*sin(gamma)

  R[0,1]=-cos(beta)*cos(gamma)*sin(alpha)-cos(alpha)*sin(gamma)
  R[1,1]=cos(alpha)*cos(beta)*cos(gamma)-sin(alpha)*sin(gamma)
  R[2,1]=cos(gamma)*sin(beta)

  R[0,2]=sin(alpha)*sin(beta)
  R[1,2]=-cos(alpha)*sin(beta)
  R[2,2]=cos(beta)

  return R


# ======================Test==========================================
# print R_matrix_from_euler_t(0.0,np.deg2rad(180.0),0.0)




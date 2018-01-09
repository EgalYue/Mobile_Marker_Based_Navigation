#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:48:08 2017

@author: lracuna
"""
import numpy as np
from math import cos, sin

def R_matrix_from_euler_t(alpha,beta,gamma):
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






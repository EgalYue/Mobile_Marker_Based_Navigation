#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:08:12 2017

@author: lracuna
"""
from scipy.linalg import expm
import numpy as np
from math import cos, sin

def rotation_matrix(a, alpha):
    """  Creates a 3D [R|t] matrix for rotation
    around the axis of the vector a by an alpha angle."""
    #Normalize the rotation axis a
    a = a / np.linalg.norm(a)

    #Build the skew symetric
    a_skew = np.mat([[0,-a[2],a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(4)
    R[:3,:3] = expm(a_skew*alpha)
    return R

def translation_matrix(t):
  """  Creates a 3D [R|t] matrix with a translation t
  and an identity rotation """
  R = np.eye(4)
  R[:3,3] = np.array([t[0],t[1],t[2]])
  return R

def rotation_matrix_from_two_vectors(a,b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a,b)
    ssc = np.matrix(np.array([[0.0, -v[2], v[1]],
                 [v[2], 0.0, -v[0]],
                 [-v[1], v[0], 0]]))
    R = np.eye(4)
    #TODO Yue: Form wrong ??? https://gist.github.com/peteristhegreat/3b76d5169d7b9fc1e333
    R[:3,:3] = np.array(np.eye(3) + ssc + (ssc**2)*(1.0/(1.0+np.dot(a,b))))
    return R

def rot_matrix_error(R0, R1, method = 'unit_quaternion_product'):
    """ R0, R1 are 3x3 or 4x4 homogeneous Rotation matrixes
        returns: the value of the error depending on the method """

    if ((R0.shape != (4,4)) and (R0.shape != (3,3))):
        print ("Error in the R0 input rotation matrix shape, must be 3x3 or 4x4")
        print R0
        return -1
    if ((R1.shape != (4,4)) and (R1.shape != (3,3))):
        print ("Error in the R1 input rotation matrix shape, must be 3x3 or 4x4")
        print R1
        return -1

    if R0.shape == (3,3):
        R = np.eye(4)
        R[:3,:3] = R0
        R0 = R

    if R1.shape == (3,3):
        R = np.eye(4)
        R[:3,:3] = R1
        R1 = R



    if(method == 'unit_quaternion_product' ):
        ## From the paper "Metrics for 3D Rotations: Comparison and Analysis" D. Huynh
        # The 3D rotation error is computed using the inner product of unit quaterions


        #We use the ros library TF to convert rotation matrix into unit quaternions
        from tf import transformations
        q0 = transformations.quaternion_from_matrix(R0)
        q1 = transformations.quaternion_from_matrix(R1)

        # We convert into unit quaternions
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)

        #Find the error as defined in the paper
        rot_error = 1 - np.linalg.norm(np.dot(q0,q1))

    if(method == 'angle'):
        #option 2 find the angle of this rotation. In particular, the above is invalid
        #for very large error angles (error > 90 degrees) and is imprecise for large
        #error angles (angle > 45 degrees).

        E = R1.dot(R0.T)
        from cv2 import Rodrigues
        rot_vector, J = Rodrigues(E[:3,:3])

        angle = np.linalg.norm(rot_vector)

        rot_error = np.rad2deg(angle)
#        d = np.zeros(3)
#        d[0] = E[1,2] - E[2,1]
#        d[1] = E[2,0] - E[0,2]
#        d[2] = E[0,1] - E[1,0]
#
#        dmag = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
#
#        phi = np.rad2deg(asin(dmag/2))
#
#        rot_error = phi

    return rot_error



def R_matrix_from_euler_t(alpha,beta,gamma):
  R = np.eye(4, dtype=np.float32)
  # R[0,0]=cos(alpha)*cos(gamma)-cos(beta)*sin(alpha)*sin(gamma)
  # R[1,0]=cos(gamma)*sin(alpha)+cos(alpha)*cos(beta)*sin(gamma)
  # R[2,0]=sin(beta)*sin(gamma)
  #
  # R[0,1]=-cos(beta)*cos(gamma)*sin(alpha)-cos(alpha)*sin(gamma)
  # R[1,1]=cos(alpha)*cos(beta)*cos(gamma)-sin(alpha)*sin(gamma)
  # R[2,1]=cos(gamma)*sin(beta)
  #
  # R[0,2]=sin(alpha)*sin(beta)
  # R[1,2]=-cos(alpha)*sin(beta)
  # R[2,2]=cos(beta)
  #  TODO Wrong sequenz
  R[0,0]=cos(alpha)*cos(gamma)-cos(beta)*sin(alpha)*sin(gamma)
  R[0,1]=cos(gamma)*sin(alpha)+cos(alpha)*cos(beta)*sin(gamma)
  R[0,2]=sin(beta)*sin(gamma)

  R[1,0]=-cos(beta)*cos(gamma)*sin(alpha)-cos(alpha)*sin(gamma)
  R[1,1]=cos(alpha)*cos(beta)*cos(gamma)-sin(alpha)*sin(gamma)
  R[1,2]=cos(gamma)*sin(beta)

  R[2,0]=sin(alpha)*sin(beta)
  R[2,1]=-cos(alpha)*sin(beta)
  R[2,2]=cos(beta)

  return R





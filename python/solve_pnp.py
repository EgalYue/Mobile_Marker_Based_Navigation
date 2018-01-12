#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:33:14 2017

@author: lracuna
"""

import cv2
import numpy as np

def pose_pnp(objectPoints, imagePoints, K, debug = False, method = cv2.SOLVEPNP_ITERATIVE, Ransac = False):
    """ This function calculates the pose using the OpenCV solvePnP algorithm
    objectPoints:  4xn homogeneous 3D object coordinates
    normalizedimagePoints: 3xn homogeneous normalized pixel coordinates
    K: Camera matrix
    method: cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_DLS, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE
    """
    objPoints = objectPoints[:3,:].T
    imgPoints = imagePoints[:2,:].T

    objPoints = np.copy(objPoints.reshape([1, -1, 3]))
    imgPoints = np.copy(imgPoints.reshape([1, -1, 2]))



    if Ransac:
      retval, rvec, tvec, inliers = cv2.solvePnPRansac(objPoints, imgPoints, K, None, None, None, False ,method)
    else:
      retval, rvec, tvec = cv2.solvePnP(objPoints,imgPoints,K, None,None, None, False, method)

    if debug:
      print tvec, rvec

    if debug:
      print("solvePnP finished succesfully", retval)

    rmat, j = cv2.Rodrigues(rvec)

    #convert back to homgeneous coordinates
    pnp_tvec = np.zeros(4)
    pnp_tvec[3] = 1
    pnp_rmat = np.eye(4)
    pnp_tvec[:3] = tvec[:,0]
    pnp_rmat[:3,:3] = rmat

    return pnp_tvec,pnp_rmat

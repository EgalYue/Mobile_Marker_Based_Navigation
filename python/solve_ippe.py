#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:31:46 2017

@author: lracuna
"""
import numpy as np
from ippe import homo2d, ippe

def pose_ippe_both(objectPoints, normalizedimagePoints, debug = False):
    """ This function calculates the pose using the IPPE algorithm which
    returns to possible poses. It returns both poses for comparison.

    objectPoints:  4xn homogeneous 3D object coordinates
    normalizedimagePoints: 3xn homogeneous normalized pixel coordinates
    """
    if debug:
      print("Starting ippe pose calculation")
    x1 = objectPoints[:3,:] # 3D coordinates (assuming a plane Z = 0)
    x2 = normalizedimagePoints[:2,:]
    ippe_result = ippe.mat_run(x1,x2, hEstMethod='DLT')

    if debug:
      print("ippe finished succesfully")

    if ippe_result['reprojError1'] <= ippe_result['reprojError2']:
        ippe_best = '1'
        ippe_worst = '2'
    else:
        ippe_best = '2'
        ippe_worst = '1'

    #convert back to homgeneous coordinates
    ippe_tvec1 = np.zeros(4)
    ippe_tvec2 = np.zeros(4)
    ippe_tvec1[3] = 1
    ippe_tvec2[3] = 1
    ippe_rmat1 = np.eye(4)
    ippe_rmat2 = np.eye(4)

    ippe_tvec1[:3] = ippe_result['t'+ippe_best]
    ippe_rmat1[:3,:3] = ippe_result['R'+ippe_best]

    ippe_tvec2[:3] = ippe_result['t'+ippe_worst]
    ippe_rmat2[:3,:3] = ippe_result['R'+ippe_worst]

#    #hack to correct sign (check why does it happen)
#    #a rotation matrix has determinant with value equal to 1
#    if np.linalg.det(ippe_rmat) < 0:
#        ippe_rmat[:3,2] = -ippe_rmat[:3,2]


    return ippe_tvec1,ippe_rmat1,ippe_tvec2,ippe_rmat2



def pose_ippe_best(objectPoints, normalizedimagePoints, debug = False):
    """ This function calculates the pose using the IPPE algorithm which
    returns to possible poses. The best pose is then selected based on
    the reprojection error and that the objectPoints have to be in front of the
    camera in marker coordinates.

    objectPoints:  4xn homogeneous 3D object coordinates
    normalizedimagePoints: 3xn homogeneous normalized pixel coordinates
    """
    if debug:
      print("Starting ippe pose calculation")
    x1 = objectPoints[:3,:] # 3D coordinates (assuming a plane Z = 0)
    x2 = normalizedimagePoints[:2,:]
    ippe_result = ippe.mat_run(x1,x2, hEstMethod='DLT')

    if debug:
      print("ippe finished succesfully")

    if ippe_result['reprojError1'] <= ippe_result['reprojError2']:
        ippe_valid = '1'
    else:
        ippe_valid = '2'

    #convert back to homgeneous coordinates
    ippe_tvec = np.zeros(4)
    ippe_tvec[3] = 1
    ippe_rmat = np.eye(4)

    ippe_tvec[:3] = ippe_result['t'+ippe_valid]
    ippe_rmat[:3,:3] = ippe_result['R'+ippe_valid]

    #hack to correct sign (check why does it happen)
    #a rotation matrix has determinant with value equal to 1
    #if np.linalg.det(ippe_rmat) < 0:
    #    ippe_rmat[:3,2] = -ippe_rmat[:3,2]


    return ippe_tvec,ippe_rmat



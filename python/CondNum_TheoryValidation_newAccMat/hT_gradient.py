#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 19.12.17 12:45
@File    : hT_gradient.py
@author: Yue Hu
"""
import sys
sys.path.append("..")
from vision.camera import *
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

class Gradient(object):
  def __init__(self):

    self.dtx = None
    self.dty = None
    self.dtz = None

    self.dtx_eval = 0
    self.dty_eval = 0
    self.dtz_eval = 0

    self.dtx_eval_old = 0
    self.dty_eval_old = 0
    self.dtz_eval_old = 0

    self.n = None #step in gradient descent

    self.n_tx = self.n
    self.n_ty = self.n
    self.n_tz = self.n

  def set_n(self,n):
    #n = 0.000001 #condition number norm 4 points
    self.n = n
    self.n_pos = 0.02*n # for SuperSAB
    self.n_neg = 0.03*n # for SuperSAB

    self.n_tx = n
    self.n_ty = n
    self.n_tz = n



def calculate_A_matrix_autograd(x1,y1,x2,y2,x3,y3,x4,y4,K,R,tx,ty,tz,radius,normalize=False,):
  """ Calculate the A matrix for the DLT algorithm:  A.H = 0
  all coordinates are in object plane
  """
  X1 = np.array([[x1],[y1],[0.],[1.]]).reshape(4,1)
  X2 = np.array([[x2],[y2],[0.],[1.]]).reshape(4,1)
  X3 = np.array([[x3],[y3],[0.],[1.]]).reshape(4,1)
  X4 = np.array([[x4],[y4],[0.],[1.]]).reshape(4,1)

  t = np.eye(4)
  t[:3, 3] = np.array([np.copy(tx), np.copy(ty), np.copy(tz)])
  Rt = np.dot(t,np.copy(R))
  P = np.dot(np.copy(K),Rt[:3,:4])
  U1 = np.array(np.dot(P,X1)).reshape(3,1)
  U2 = np.array(np.dot(P,X2)).reshape(3,1)
  U3 = np.array(np.dot(P,X3)).reshape(3,1)
  U4 = np.array(np.dot(P,X4)).reshape(3,1)

  object_pts = np.hstack([X1,X2,X3,X4])
  image_pts = np.hstack([U1,U2,U3,U4])

  if normalize:
    object_pts_norm,T1 = normalise_points(object_pts)
    # image_pts_norm, T2 = normalise_points(image_pts)
    image_pts_norm,T2 = normalise_points_RadiusScale(image_pts,radius)

  else:
    object_pts_norm = object_pts[[0,1,3],:]
    image_pts_norm = image_pts

  x1 = object_pts_norm[0,0]/object_pts_norm[2,0]
  y1 = object_pts_norm[1,0]/object_pts_norm[2,0]

  x2 = object_pts_norm[0,1]/object_pts_norm[2,1]
  y2 = object_pts_norm[1,1]/object_pts_norm[2,1]

  x3 = object_pts_norm[0,2]/object_pts_norm[2,2]
  y3 = object_pts_norm[1,2]/object_pts_norm[2,2]

  x4 = object_pts_norm[0,3]/object_pts_norm[2,3]
  y4 = object_pts_norm[1,3]/object_pts_norm[2,3]


  u1 = image_pts_norm[0,0]/image_pts_norm[2,0]
  v1 = image_pts_norm[1,0]/image_pts_norm[2,0]

  u2 = image_pts_norm[0,1]/image_pts_norm[2,1]
  v2 = image_pts_norm[1,1]/image_pts_norm[2,1]

  u3 = image_pts_norm[0,2]/image_pts_norm[2,2]
  v3 = image_pts_norm[1,2]/image_pts_norm[2,2]

  u4 = image_pts_norm[0,3]/image_pts_norm[2,3]
  v4 = image_pts_norm[1,3]/image_pts_norm[2,3]

  A = np.array([    [ 0,  0, 0, -x1, -y1, -1,  v1*x1,  v1*y1,  v1],
                    [x1, y1, 1,   0,   0,  0, -u1*x1, -u1*y1, -u1],

                    [ 0,  0, 0, -x2, -y2, -1,  v2*x2,  v2*y2,  v2],
                    [x2, y2, 1,   0,   0,  0, -u2*x2, -u2*y2, -u2],

                    [ 0,  0, 0, -x3, -y3, -1,  v3*x3,  v3*y3,  v3],
                    [x3, y3, 1,   0,   0,  0, -u3*x3, -u3*y3, -u3],

                    [0,   0, 0, -x4, -y4, -1,  v4*x4,  v4*y4,  v4],
                    [x4, y4, 1,   0,   0,  0, -u4*x4, -u4*y4, -u4],
          ])
  return A

# TODO add radius in parameter list  used for scale
def matrix_condition_number_autograd(x1,y1,x2,y2,x3,y3,x4,y4,K,R,tx,ty,tz,radius, normalize = False):
  A = calculate_A_matrix_autograd(x1,y1,x2,y2,x3,y3,x4,y4,K,R,tx,ty,tz,radius, normalize)

  U, s, V = np.linalg.svd(A,full_matrices=False)

  greatest_singular_value = s[0]
#  rcond=1e-5
#  if s[-1] > rcond:
#    smalles_singular_value = s[-1]
#  else:
#    smalles_singular_value = s[-2]
  smallest_singular_value = s[-2]

  return greatest_singular_value/smallest_singular_value
  #return np.sqrt(greatest_singular_value)/np.sqrt(smalles_singular_value)

def repro_error_autograd(x1,y1,x2,y2,x3,y3,x4,y4,P, image_pts_measured, normalize = False):
  X1 = np.array([[x1],[y1],[0.],[1.]]).reshape(4,1)
  X2 = np.array([[x2],[y2],[0.],[1.]]).reshape(4,1)
  X3 = np.array([[x3],[y3],[0.],[1.]]).reshape(4,1)
  X4 = np.array([[x4],[y4],[0.],[1.]]).reshape(4,1)

  U1 = np.array(np.dot(P,X1)).reshape(3,1)
  U2 = np.array(np.dot(P,X2)).reshape(3,1)
  U3 = np.array(np.dot(P,X3)).reshape(3,1)
  U4 = np.array(np.dot(P,X4)).reshape(3,1)

  U1 = U1/U1[2,0]
  U2 = U2/U2[2,0]
  U3 = U3/U3[2,0]
  U4 = U4/U4[2,0]
#
#  u1_r = U1[0,0]/U1[2,0]
#  v1_r = U1[1,0]/U1[2,0]
#
#  u2_r = U2[0,1]/U2[2,1]
#  v2_r = U2[1,1]/U2[2,1]
#
#  u3_r = U3[0,2]/U3[2,2]
#  v3_r = U3[1,2]/U3[2,2]
#
#  u4_r = U4[0,3]/U4[2,3]
#  v4_r = U4[1,3]/U4[2,3]

  object_pts = np.hstack([X1,X2,X3,X4])
  image_pts_repro = np.hstack([U1,U2,U3,U4])

  x = image_pts_measured[:2,:]-image_pts_repro[:2,:]
  repro = np.sum(x**2)**(1./2)

  return repro
#  x1 = object_pts[0,0]/object_pts[2,0]
#  y1 = object_pts[1,0]/object_pts[2,0]
#
#  x2 = object_pts[0,1]/object_pts[2,1]
#  y2 = object_pts[1,1]/object_pts[2,1]
#
#  x3 = object_pts[0,2]/object_pts[2,2]
#  y3 = object_pts[1,2]/object_pts[2,2]
#
#  x4 = object_pts[0,3]/object_pts[2,3]
#  y4 = object_pts[1,3]/object_pts[2,3]
#
#
#  u1_r = image_pts_repro[0,0]/image_pts_repro[2,0]
#  v1_r = image_pts_repro[1,0]/image_pts_repro[2,0]
#
#  u2_r = image_pts_repro[0,1]/image_pts_repro[2,1]
#  v2_r = image_pts_repro[1,1]/image_pts_repro[2,1]
#
#  u3_r = image_pts_repro[0,2]/image_pts_repro[2,2]
#  v3_r = image_pts_repro[1,2]/image_pts_repro[2,2]
#
#  u4_r = image_pts_repro[0,3]/image_pts_repro[2,3]
#  v4_r = image_pts_repro[1,3]/image_pts_repro[2,3]


def hom_3d_to_2d(pts):
    pts = pts[[0,1,3],:]
    return pts

def hom_2d_to_3d(pts):
    pts = np.insert(pts,2,np.zeros(pts.shape[1]),0)
    return pts

def normalise_points(pts):
    """
    Function translates and normalises a set of 2D or 3d homogeneous points
    so that their centroid is at the origin and their mean distance from
    the origin is sqrt(2).  This process typically improves the
    conditioning of any equations used to solve homographies, fundamental
    matrices etc.


    Inputs:
    pts: 3xN array of 2D homogeneous coordinates

    Returns:
    newpts: 3xN array of transformed 2D homogeneous coordinates.  The
            scaling parameter is normalised to 1 unless the point is at
            infinity.
    T: The 3x3 transformation matrix, newpts = T*pts
    """
    if pts.shape[0] == 4:
        pts = hom_3d_to_2d(pts)

    if pts.shape[0] != 3 and pts.shape[0] != 4  :
        print "Shape error"


    finiteind = np.nonzero(abs(pts[2,:]) > np.spacing(1))

    if len(finiteind[0]) != pts.shape[1]:
        print('Some points are at infinity')

    dist = []
    pts = pts/pts[2,:]
    for i in finiteind:
        #Replaced below for autograd
#        pts[0,i] = pts[0,i]/pts[2,i]
#        pts[1,i] = pts[1,i]/pts[2,i]
#        pts[2,i] = 1;

        c = np.mean(pts[0:2,i].T, axis=0).T

        newp1 = pts[0,i]-c[0]
        newp2 = pts[1,i]-c[1]

        dist.append(np.sqrt(newp1**2 + newp2**2))

    dist = np.array(dist)

    meandist = np.mean(dist)

    scale = np.sqrt(2)/meandist

    T = np.array([[scale, 0, -scale*c[0]], [0, scale, -scale*c[1]], [0, 0, 1]])

    newpts = np.dot(T,pts)


    return newpts, T


def normalise_points_RadiusScale(pts,radius):
  """
  Function translates and normalises a set of 2D or 3d homogeneous points
  so that their centroid is at the origin
  Add radius(distance between camera and marker) as one influenced factor of Scale
  This process typically improves the
  conditioning of any equations used to solve homographies, fundamental
  matrices etc.


  Inputs:
  pts: 3xN array of 2D homogeneous coordinates

  Returns:
  newpts: 3xN array of transformed 2D homogeneous coordinates.  The
          scaling parameter is normalised to 1 unless the point is at
          infinity.
  """
  if pts.shape[0] == 4:
    pts = hom_3d_to_2d(pts)

  if pts.shape[0] != 3 and pts.shape[0] != 4:
    print "Shape error"

  finiteind = np.nonzero(abs(pts[2, :]) > np.spacing(1))

  if len(finiteind[0]) != pts.shape[1]:
    print('Some points are at infinity')

  dist = []
  pts = pts / pts[2, :]
  for i in finiteind:
    c = np.mean(pts[0:2, i].T, axis=0).T
    newp1 = pts[0, i] - c[0]
    newp2 = pts[1, i] - c[1]
    dist.append(np.sqrt(newp1 ** 2 + newp2 ** 2))

  dist = np.array(dist)
  meandist = np.mean(dist)
  # scale = np.sqrt(2) / meandist # translate and scaling
  # scale = 1 # Only translate, no scaling
  scale = radius*np.sqrt(2) / meandist # translate and set the scale increasing with the radius(distance from cam to marker)
  T = np.array([[scale, 0, -scale * c[0]], [0, scale, -scale * c[1]], [0, 0, 1]])
  newpts = np.dot(T, pts)
  return newpts,T

def create_gradient(metric='condition_number', n = 0.000001):
  """"
  metric: 'condition_number' (default)
          'volker_metric
  """
  if metric == 'condition_number':
    metric_function = matrix_condition_number_autograd
  elif metric == 'pnorm_condition_number':
    metric_function = matrix_pnorm_condition_number_autograd
  elif metric == 'volker_metric':
    metric_function = volker_metric_autograd
  elif metric == 'repro_error':
    metric_function = repro_error_autograd

  gradient = Gradient()
  gradient.set_n(n)
  gradient.dtx = grad(metric_function,0)
  gradient.dty = grad(metric_function,1)
  gradient.dtz = grad(metric_function,2)

  return gradient


def extract_objectpoints_vars(objectPoints):
  x1 = objectPoints[0,0]
  y1 = objectPoints[1,0]

  x2 = objectPoints[0,1]
  y2 = objectPoints[1,1]

  x3 = objectPoints[0,2]
  y3 = objectPoints[1,2]

  x4 = objectPoints[0,3]
  y4 = objectPoints[1,3]

  return [x1,y1,x2,y2,x3,y3,x4,y4]

def evaluate_gradient(gradient, objectPoints,K,R,tx,ty,tz, normalize = False, ):
  x1,y1,x2,y2,x3,y3,x4,y4 = extract_objectpoints_vars(objectPoints)

  gradient.dtx_eval_old = gradient.dtx_eval
  gradient.dty_eval_old = gradient.dty_eval
  gradient.dtz_eval_old = gradient.dtz_eval

  gradient.dtx_eval = gradient.dtx(x1,y1,x2,y2,x3,y3,x4,y4,K,R,tx,ty,tz,normalize)*gradient.n_tx
  gradient.dty_eval = gradient.dtx(x1, y1, x2, y2, x3, y3, x4, y4, K, R, tx, ty, tz, normalize) * gradient.n_ty
  gradient.dtz_eval = gradient.dtx(x1, y1, x2, y2, x3, y3, x4, y4, K, R, tx, ty, tz, normalize) * gradient.n_tz
  ## Limit
  limit = 0.1
  print "--gradient.dtz_eval before limit---"
  print gradient.dtz_eval
  print "-----"
  gradient.dtx_eval = np.clip(gradient.dtx_eval, -limit, limit)
  gradient.dty_eval = np.clip(gradient.dty_eval, -limit, limit)
  gradient.dtz_eval = np.clip(gradient.dtz_eval, -limit, limit)
  print "--gradient.dtz_eval after limit---"
  print gradient.dtz_eval
  print "-----"
  gradient.n_tx = supersab(gradient.n_tx,gradient.dtx_eval,gradient.dtx_eval_old,gradient.n_pos,gradient.n_neg)
  gradient.n_ty = supersab(gradient.n_ty, gradient.dty_eval, gradient.dty_eval_old, gradient.n_pos, gradient.n_neg)
  gradient.n_tz = supersab(gradient.n_tz, gradient.dtz_eval, gradient.dtz_eval_old, gradient.n_pos, gradient.n_neg)
  return gradient

def supersab(n, gradient_eval_current, gradient_eval_old, n_pos, n_neg):
  if np.sign(gradient_eval_current*gradient_eval_old) > 0:
    n = n + n_pos
  else:
    n = n - n_neg
  return n

def update_points(gradient, T, limitx=5,limity=5,limitz=5):
  t = np.copy(T)
  print "---t---"
  print t
  print "---gradient.dtx_eval , gradient.dty_eval , gradient.dtz_eval---"
  print gradient.dtx_eval, gradient.dty_eval, gradient.dtz_eval
  t[0] += - gradient.dtx_eval
  t[1] += - gradient.dty_eval
  t[2] += - gradient.dtz_eval
  print "----t new----"
  print t
  # TODO set limit distance for t
  t[0] = np.clip(t[0], -limitx, limitx)
  t[1] = np.clip(t[1], -limity, limity)
  t[2] = np.clip(t[2], -limitz, limitz)
  # circle = True
  # radius = 0.15
  # if (circle):
  #     for i in range(op.shape[1]):
  #         distance = np.sqrt(op[0,i]**2+op[1,i]**2)
  #         if distance > radius:
  #             op[:3,i] = op[:3,i]*radius/distance
  #
  # else:
  #     op[0,:] = np.clip(op[0,:], -limitx, limitx)
  #     op[1,:] = np.clip(op[1,:], -limity, limity)
  return t

# =======================================Test======================================
# pts = np.array([[100,200,300,400],[200,100,100,200],[1,1,1,1]])
# print normalise_points_withoutScale(pts)


def plotImagePointsDyn(imagePoints):
    # ------------- plot the image points dynamiclly-----------------
    print "imagePoints\n", imagePoints
    fig1 = plt.figure('Image points')
    ax_image = fig1.add_subplot(211)
    ax_image.cla()
    plt.sca(ax_image)
    plt.ion()
    ax_image.plot(imagePoints[0], imagePoints[1], '.', color='blue', )
    ax_image.set_xlim(0, 1280)
    ax_image.set_ylim(0, 960)
    ax_image.invert_yaxis()
    ax_image.set_title('Image Points')
    plt.show()
    plt.pause(0.001)

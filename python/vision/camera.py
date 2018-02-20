# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
sys.path.append("..")
from scipy.linalg import expm, rq, det, inv
import matplotlib.pyplot as plt

from math import atan
from rt_matrix import rotation_matrix
import autograd.numpy as np
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t

class Camera(object):
    """ Class for representing pin-hole cameras. """
    def __init__(self):
        """ Initialize P = K[R|t] camera model. """
        self.P = np.eye(3,4)
        self.K = np.eye(3, dtype=np.float32) # calibration matrix
        self.R = np.eye(4, dtype=np.float32) # rotation
        self.t = np.eye(4, dtype=np.float32) # translation
        self.Rt = np.eye(4, dtype=np.float32)
        self.fx = 1.
        self.fy = 1.
        self.cx = 0.
        self.cy = 0.
        self.img_width = 1280
        self.img_height = 960

    def clone_withPose(self, tvec, rmat):
        new_cam = Camera()
        new_cam.K = self.K
        new_cam.set_R_mat(rmat)
        new_cam.set_t(tvec[0], tvec[1],  tvec[2])
        new_cam.set_P()
        new_cam.img_height = self.img_height
        new_cam.img_width = self.img_width
        return new_cam

    def clone(self):
        new_cam = Camera()
        new_cam.P = self.P
        new_cam.K = self.K
        new_cam.R = self.R
        new_cam.t = self.t
        new_cam.Rt = self.Rt
        new_cam.fx = self.fx
        new_cam.fy = self.fy
        new_cam.cx = self.cx
        new_cam.cy = self.cy
        new_cam.img_height = self.img_height
        new_cam.img_width = self.img_width
        return new_cam


    def set_P(self):
        # P = K[R|t]
        # P is a 3x4 Projection Matrix (from 3d euclidean to image)
        #self.Rt = hstack((self.R, self.t))
        self.P = np.dot(self.K, self.Rt[:3,:4])

    def set_K(self, fx = 1, fy = 1, cx = 0,cy = 0):
        # K is the 3x3 Camera matrix
        # fx, fy are focal lenghts expressed in pixel units
        # cx, cy is a principal point usually at image center
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.mat([[fx, 0, cx],
                      [0,fy,cy],
                      [0,0,1.]], dtype=np.float32)
        self.set_P()

    def set_width_heigth(self,width, heigth):
        self.img_width = width
        self.img_height = heigth


    def update_Rt(self):
        self.Rt = np.dot(self.t,self.R)
        self.set_P()

    def set_R_axisAngle(self,x,y,z, alpha):
        """  Creates a 3D [R|t] matrix for rotation
        around the axis of the vector defined by (x,y,z)
        and an alpha angle."""
        #Normalize the rotation axis a
        a = np.array([x,y,z])
        a = a / np.linalg.norm(a)

        #Build the skew symetric
        a_skew = np.mat([[0,-a[2],a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(4)
        R[:3,:3] = expm(a_skew*alpha)
        self.R = R
        self.update_Rt()

    def set_R_mat(self,R):
        self.R = R
        self.update_Rt()


    def set_t(self, x,y,z, frame = 'camera'):
        #self.t = array([[x],[y],[z]])
        self.t = np.eye(4)
        if frame=='world':
          cam_world = np.array([x,y,z,1]).T
          cam_t = np.dot(self.R,-cam_world)
          self.t[:3,3] = cam_t[:3]
        else:
          self.t[:3,3] = np.array([x,y,z])
        self.update_Rt()

    def get_normalized_pixel_coordinates(self, X):
        """
        These are in normalised pixel coordinates. That is, the effects of the
        camera's intrinsic matrix and lens distortion are corrected, so that
        the Q projects with a perfect pinhole model.
        """
        return np.dot(inv(self.K), X)

    def addnoise_imagePoints(self, imagePoints, mean = 0, sd = 2):
        """ Add Gaussian noise to image points
        imagePoints: 3xn points in homogeneous pixel coordinates
        mean: zero mean
        sd: pixels of standard deviation
        """
        imagePoints = np.copy(imagePoints)
        if sd > 0:
            gaussian_noise = np.random.normal(mean,sd,(2,imagePoints.shape[1]))
            imagePoints[:2,:] = imagePoints[:2,:] + gaussian_noise
        return imagePoints


    def get_tvec(self):
        tvec = self.t[:,3]
        return tvec



    def get_world_position(self):
        t = np.dot(inv(self.Rt), np.array([0,0,0,1]))
        return t


    def project(self,X, quant_error=False):
        """  Project points in X (4*n array) and normalize coordinates. """
        self.set_P()
        x = np.dot(self.P,X)
        for i in range(x.shape[1]):
          x[:,i] /= x[2,i]
        if(quant_error):
            x = np.around(x, decimals=0)
        return x

    def project_circle(self, circle):
        C = circle.get_C
        H = self.homography_from_Rt()
        Q = None

    def plot_image(self, imgpoints, points_color = 'blue'):
        # show Image
        # plot projection
        plt.figure("Camera Projection")
        plt.plot(imgpoints[0],imgpoints[1],'.',color = points_color)
        #we add a key point to help us see orientation of the points
        plt.plot(imgpoints[0,0],imgpoints[1,0],'.',color = 'blue')
        plt.xlim(0,self.img_width)
        plt.ylim(0,self.img_height)
        plt.gca().invert_yaxis()
        plt.show()

    def plot_plane(self, plane):
        if plane.type == 'rectangular':
            corners = plane.get_corners()
            img_corners = np.array(self.project(corners))
            img_corners =np.c_[img_corners,img_corners[:,0]]
            plt.plot(img_corners[0],img_corners[1])
        elif plane.type == 'circular':
            c = plane.circle
            c_projected = c.project(self.homography_from_Rt())
            c_projected.contour(grid_size=100)




    def factor(self):
        """  Factorize the camera matrix into K,R,t as P = K[R|t]. """
        # factor first 3*3 part
        K,R = rq(self.P[:,:3])
        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if det(T) < 0:
            T[1,1] *= -1
        self.K = np.dot(K,T)
        self.R = np.dot(T,R) # T is its own inverse
        self.t = np.dot(inv(self.K),self.P[:,3])
        return self.K, self.R, self.t

    def fov(self):
        """ Calculate field of view angles (grads) from camera matrix """
        fovx = np.rad2deg(2 * atan(self.img_width / (2. * self.fx)))
        fovy = np.rad2deg(2 * atan(self.img_height / (2. * self.fy)))
        return fovx, fovy

    def move(self, x,y,z):
        Rt = np.identity(4);
        Rt[:3,3] = np.array([x,y,z])
        self.P = np.dot(self.K, self.Rt)

    def rotate_around_world(self,axis, angle):
        """ rotate camera around a given axis in WORLD coordinates"""
        R = rotation_matrix(axis, angle)
        self.Rt = np.dot(R, self.Rt)
        # newR = np.dot(R,self.R)
        # self.Rt = np.dot(self.t, newR)
        self.R[:3,:3] = self.Rt[:3,:3]
        self.t[:3,3] = self.Rt[:3,3]
        # DO NOT forget to set new P
        self.set_P()

    def rotate(self, axis, angle):
        """ rotate camera around a given axis in CAMERA coordinate, please use following Rt"""
        R = rotation_matrix(axis, angle)
        # self.Rt = np.dot(R, self.Rt)
        newR = np.dot(R,self.R)
        self.Rt = np.dot(self.t, newR)
        self.R[:3,:3] = self.Rt[:3,:3]
        self.t[:3,3] = self.Rt[:3,3]
        #TODO DO NOT forget to set new P
        self.set_P()

    def rotate_x(self,angle):
        self.rotate(np.array([1,0,0],dtype=np.float32), angle)

    def rotate_y(self,angle):
        self.rotate(np.array([0,1,0],dtype=np.float32), angle)

    def rotate_z(self,angle):
        self.rotate(np.array([0,0,1],dtype=np.float32), angle)

    def look_at(self, world_position):
      #%%
      world_position = self.get_world_position()[:3]
      eye = world_position
      target = np.array([0,0,0])
      up = np.array([0,1,0])

      zaxis = (target-eye)/np.linalg.norm(target-eye)
      xaxis = (np.cross(up,zaxis))/np.linalg.norm(np.cross(up,zaxis))
      yaxis = np.cross(zaxis, xaxis)

      # print "xaxis",xaxis
      # print "yaxis",yaxis
      # print "zaxis",zaxis
      R = np.eye(4)
      # TODO should use this R
      R = np.array([[xaxis[0], yaxis[0], zaxis[0], 0],
                   [xaxis[1], yaxis[1], zaxis[1], 0],
                   [xaxis[2], yaxis[2], zaxis[2], 0],
                   [       0,        0,        0, 1]]
          )

      # R = np.array([[xaxis[0], xaxis[1], xaxis[2], 0],
      #              [yaxis[0], yaxis[1], yaxis[2], 0],
      #              [zaxis[0], zaxis[1], zaxis[2], 0],
      #              [       0,        0,        0, 1]])

      # print (xaxis.T).dot(xaxis)
      # print (zaxis.T).dot(zaxis)
      # print (yaxis.T).dot(yaxis)
      # print "R",R
      t = np.eye(4, dtype=np.float32) # translation
      t[:3,3] = -eye

      self.R = R


      self.Rt = np.dot(R,t)
      self.t = np.eye(4, dtype=np.float32)
      self.t[:3,3] = self.Rt[:3,3]

    def homography_from_Rt(self):
      rt_reduced = self.Rt[:3,[0,1,3]]
      H = np.dot(self.K,rt_reduced)
      if H[2,2] != 0.:
        H = H/H[2,2]
      return H

      #%%
# cam = Camera()
#
# #Test that projection matrix doesnt change rotation and translation
#
# cam.set_world_position(0,0,-2.5)
# R1= cam.R
# t1 = cam.t
# Rt1 = cam.Rt
# pos1 = cam.get_world_position()
# cam.set_P()
# R2 = cam.R
# t2 = cam.t
# Rt2 = cam.Rt
# pos2 = cam.get_world_position()
# print pos1-pos2
# print R1 - R2
# print t1 - t2
# print Rt1 - Rt2
#
#
#print "------------------------------"
##Test that rotate function doesnt change translation matrix
#
#cam.set_world_position(0,0,-2.5)
#R1= cam.R
#t1 = cam.t
#Rt1 = cam.Rt
#pos1 = cam.get_world_position()
#cam.set_P()
#
#cam.rotate_y(deg2rad(+20.))
#cam.rotate_y(deg2rad(+20.))
#cam.set_P()
#R2 = cam.R
#t2 = cam.t
#Rt2 = cam.Rt
#pos2 = cam.get_world_position()
#print pos1-pos2
#print R1 - R2
#print t1 - t2
#print Rt1 - Rt2
# ======================Test===================================
# cam = Camera()
# f = 800
# cam.set_K(fx = f, fy = f, cx = 320, cy = 240)  #Camera Matrix
# cam.img_width = 320*2
# cam.img_height = 240*2
# cam.set_t(1,0,0,"world")
# cam.set_R_mat(Rt_matrix_from_euler_t.R_matrix_from_euler_t(0,np.deg2rad(0),0))
# cam.look_at([0,0,0])
# print "cam.R",cam.R
# print "cam.Rt",cam.Rt
# print "cam.P",cam.P



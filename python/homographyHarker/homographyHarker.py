#function [ H, Lh ] = homographyHarker( DataA, DataB, LA, LB)
#
# Purpose : Computes the general projective transformation between two sets
# of 2D data using a linear algorithm (the homography).
#
# Uses (syntax) :
#   H = homography( DataA, DataB ) 
#
# Input Parameters :
#   DataA, DataB := 2D homogeneous data sets in matrix form (3xn)
#
# Return Parameters :
#   H := the 3x3 homography
#
# Description and algorithms:
#   The algorithm is based on the Direct Linear Transform (DLT) method
#   outlined in Hartley et al.  The method uses orthogonal projections of
#   matrices, such that the vanishing line is treated as the principal
#   component of the reduction.  In this manner, the statistical behaviour
#   of the errors in variables are treated uniformly, see Harker and
#   O'Leary 2005.
#
# References :
#   Harker, M., O'Leary, P., Computation of Homographies, to appear in
#   Proceedings of the British Machine Vision Conference 2005, Oxford,
#   England.
#   Hartley, R., Zisserman, A., Multiple View Geometry in Computer Vision,
#   Cambridge University Press, Cambridge, 2001
#
# Cite this as :
#
# Author : Matthew Harker
# Date : July 25, 2005
# Version : 1.0
#--------------------------------------------------------------------------
# (c) 2005, O'Leary, Harker, University of Leoben, Leoben, Austria
# email: automation@unileoben.ac.at, url: automation.unileoben.ac.at
#--------------------------------------------------------------------------
# History:
#   Date:           Comment:
#   July 25, 2005   Original Version 1.0
#--------------------------------------------------------------------------
#
import numpy as np
# -*- coding: utf-8 -*-
def is2DData(Data):
    # Purpose : Tests if the input argument represents valid 2D homogeneous
    # coordinates.
    #
    # Uses (syntax) :
    #   is2DData( Data )
    #
    # Input Parameters :
    #   Data := the variable to be tested (should be 3xn, n greater than 0)
    #
    # Return Parameters :
    #   trueFalse := 0 or 1 (false or true)
    #
    # Description and algorithms:
    #   Tests the size of the input argument
    #
    # References :
    #
    # Cite this as :
    #
    # Author : Matthew Harker
    # Date : July 13, 2005
    # Version : 1.0
    #--------------------------------------------------------------------------
    # (c) 2005, O'Leary, Harker, University of Leoben, Leoben, Austria
    # email: automation@unileoben.ac.at, url: automation.unileoben.ac.at
    #--------------------------------------------------------------------------
    # History:
    #   Date:           Comment:
    #   July 13, 2005   Original Version 1.0
    #--------------------------------------------------------------------------
    #
    m = Data.shape[0]
    n = Data.shape[1]
        
    if (m == 3) and (n > 0):
        trueFalse = True
    else:
        trueFalse = False
    
    return trueFalse


def normalizeData(Data, L = None):
    #
    #function [DataN, T, Ti, LN] = normalizeData( Data, L ) ;
    #
    # Purpose : Computes a set of data corresponding to the input data with its
    # centroid subtracted, and scaled such that the root-mean-square distance
    # to the origin is sqrt(2).  The transformation T, carries the scaled data
    # back to its original form.  Optionally, the first order estimation of
    # covariance matrices are computed.
    #
    # Uses (syntax) :
    #   [DataN, T, LN] = normalizeData( Data, L ) Not yet implemented in python
    #   [DataN, T] = normalizeData( Data )
    #
    # Input Parameters :
    #   Data := a 3xn matrix of homogeneous points.
    #   L    := is a 3x3 covariance matrix (all points have identical covariance), or
    #           a 3x3xn array of n covariance matrices.
    #
    # Return Parameters :
    #   DataN := mean-free data scaled s.t. d_RMS = sqrt(2)
    #   T     := transformation to bring DataN to the Affine coordinates
    #            corresponding to Data (NOTE: T*DataN is in affine coords).
    #   LN    := the covariance of the scaled normalized data (size is
    #            generally 2x2xn, due to the normalization)
    #
    # Description and algorithms:
    #
    # References :
    #   Clarke, J.C., Modelling Uncertainty: A Primer, Dept. of Engineering
    #   Science, Oxford University, Technical Report.
    #
    # Cite this as :
    #
    # Author : Matthew Harker
    # Date : July 7, 2005
    # Version :
    #
    # (c) 2005, Institute for Automation, University of Leoben, Leoben, Austria
    # email: automation@unileoben.ac.at, url: automation.unileoben.ac.at
    #
    # History:
    #   Date:           Comment:
    #                   Original Version 1.0
    #--------------------------------------------------------------------------
    #
    # Check input arguments :
    #
    if L == None:
        if not is2DData( Data ):
            print 'Error:Input does not represent 2D data'
    else:
        print 'Error: covariance not yet implemented in Python version'
    
    Data = np.copy(Data)
    s = Data[0,:]
    t = Data[1,:]
    u = Data[2,:]
    
    x = s/u
    y = t/u
    
    xm = np.mean( x )
    ym = np.mean( y )
    
    xh = x - xm
    yh = y - ym
    
    n = len( xh )
    
    kappa = np.sum( xh**2 + yh**2 )
    
    beta = np.sqrt( 2*n / kappa ) ;
    
    xn = beta * xh
    yn = beta * yh
    
    DataN = np.vstack([xn,yn,np.ones(len(xn))])
    
    T = np.array([[ 1/beta,   0   , xm ],
                  [   0   , 1/beta, ym ],
                  [   0   ,   0   ,  1 ]])
    
    Ti = np.array([[ beta ,  0   , -beta * xm ],
                   [  0   , beta , -beta * ym ],
                   [  0   ,  0   ,       1    ]]) 
    
    return DataN, T, Ti
    
def homographyHarker(DataA, DataB, LA = None, LB = None):
    # Check input parameters:
    if LA == None and LB == None:
        if not is2DData( DataA ) or not is2DData( DataB ):
            print 'Error: Input does not represent 2D data'
        nA = DataA.shape[1]
        nB = DataB.shape[1]
        if (nA != nB):
            print 'Error: Data sets must be the same size'
    else:
        print 'Error: Input convariance data not yet implemented in Python'
        print 'Error propagation not implemented as of yet'

    # Normalize the input data:
    
    DataA,TA,TAi = normalizeData( DataA )
    DataB,TB,TBi = normalizeData( DataB )    
    
    # Construct the orthogonalized design matrix :
    
    C1 = -DataB[0,:] * DataA[0,:]
    C2 = -DataB[0,:] * DataA[1,:]
    
    C3 = -DataB[1,:] * DataA[0,:]
    C4 = -DataB[1,:] * DataA[1,:]
    
    mC1 = np.mean( C1 )
    mC2 = np.mean( C2 )
    mC3 = np.mean( C3 )
    mC4 = np.mean( C4 )
    
    Mx = np.column_stack([C1 - mC1, C2 - mC2, -DataB[0,:]])
    My = np.column_stack([C3 - mC3, C4 - mC4, -DataB[1,:]])
    
    Pp = np.linalg.pinv(DataA[0:2,:].conj().T)
    
    Bx = np.dot(Pp, Mx)
    By = np.dot(Pp, My)
    
    D = np.row_stack([Mx - np.dot(DataA[0:2,:].conj().T,Bx), My - np.dot(DataA[0:2,:].conj().T,By)])
    #% Find v_min and backsubstitute :
    #%
    U,S,Vh = np.linalg.svd( D )
    V = Vh.T    
    
    h789 = V[:,-1]
    h12 = -Bx.dot(h789)
    h45 = -By.dot(h789)
    h3 = -np.array([mC1, mC2]).dot(h789[0:2])
    h6 = -np.array([mC3, mC4]).dot(h789[0:2])
    
    # Reshape vector h to matrix H, and transform :
    
    H = np.hstack([h12, h3, h45, h6, h789]).reshape(3,3)
    
    H = TB.dot(H).dot(TAi)
    H = H / H[2,2]
    
    return H



def test():
    DataA = np.array([[1,2,3,4],[1,3,4,4],[1,1,1,1]])
    DataB = np.array([[2,4,6,8],[3,9,12,12],[1,1,1,1]])
    DataC = np.array([[2,2,3,1],[2,3,4,1],[1,1,1,1]])
    H = homographyHarker(DataA, DataB)


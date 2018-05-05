import numpy as np
import math


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# Test
# ---------------!!!! first rotation around Z -> Y -> X !!!!----------------------------
R = np.array([[ 0.92413276,  0.08257994, -0.3730405],[  0. ,        -0.97636294, -0.21613738],[-0.38207155,  0.19973963, -0.90228897]])
x,y,z = rotationMatrixToEulerAngles(R)
print np.rad2deg(x),np.rad2deg(y),np.rad2deg(z)

# --cam angle x Best-- -12.4822633668
# --cam angle y Best-- 22.4620582587
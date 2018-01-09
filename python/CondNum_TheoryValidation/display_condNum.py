import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


# sio.savemat('testpython.mat', {'data': [[1, 2, 3], [1, 2, 3], [9, 19, 29],[80000,60000,3000]]})
# data = sio.loadmat('testpython.mat')
# sio.whosmat('testpython.mat')


# -------------------------------------------------------------------------------
def Detectionplot(m):

    # data = sio.loadmat('testpython.mat')
    # m = data['data']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = m[0]
    y = m[1]
    z = m[2]
    cond_num = m[3]
    x = x.flatten('F')
    y = y.flatten('F')
    C = []

    for a in cond_num:
        if a > 70000.0:
            C.append('linen')
        elif a > 60000.0:
            C.append('antiquewhite')
        elif a > 50000.0:
            C.append('papayawhip')
        elif a > 40000.0:
            C.append('oldlace')
        elif a > 30000.0:
            C.append('cornsilk')
        elif a > 20000.0:
            C.append('palegoldenrod')
        elif a > 10000.0:
            C.append('yellow')
        elif a > 8000.0:
            C.append('lightblue')
        elif a > 6000.0:
            C.append('deepskyblue')
        elif a > 4000.0:
            C.append('red')
        elif a > 2000.0:
            C.append('darkred')
        elif a > 1000.0:
            C.append('maroon')
        else:
            C.append('black')

    dx = 0.1 * np.ones_like(x)
    dy = 0.1 * np.ones_like(y)
    dz = abs(z) * z.flatten()
    dz = dz.flatten() / abs(z)
    z = np.zeros_like(z)

    ax.set_xlabel('Xlabel')
    ax.set_ylabel('Ylabel')
    ax.set_zlabel('Zlabel')

    ax.bar3d(x, y, z, dx, dy, dz, color=C, zsort='average')
    plt.show()
    plt.pause(1000)

# plot angle
def anglePlot(m):

    # data = sio.loadmat('angle_data3.mat')
    # m = data['data']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = m[0] # angle rotation around x
    y = m[1] # angle rotation around y
    z = m[2] # cond num
    x = x.flatten('F')
    y = y.flatten('F')

    C = []
    for a in z:
        if a > 70000.0:
            C.append('linen')
        elif a > 60000.0:
            C.append('antiquewhite')
        elif a > 50000.0:
            C.append('papayawhip')
        elif a > 40000.0:
            C.append('oldlace')
        elif a > 30000.0:
            C.append('cornsilk')
        elif a > 20000.0:
            C.append('palegoldenrod')
        elif a > 10000.0:
            C.append('yellow')
        elif a > 8000.0:
            C.append('lightblue')
        elif a > 6000.0:
            C.append('deepskyblue')
        elif a > 4000.0:
            C.append('red')
        elif a > 2000.0:
            C.append('green')
        elif a > 1000.0:
            C.append('maroon')
        else:
            C.append('black')

    dx = 1 * np.ones_like(x)
    dy = 1 * np.ones_like(y)
    dz = abs(z) * z.flatten()
    dz = dz.flatten() / abs(z)
    z = np.zeros_like(z)

    ax.set_xlabel('Xlabel')
    ax.set_ylabel('Ylabel')
    ax.set_zlabel('Zlabel')
    ax.bar3d(x, y, z, dx, dy, dz, color=C, zsort='average')
    plt.show()
    plt.pause(1000)
#--------------------------------------------- Test--------------------------------------------------
# Detectionplot()
# anglePlot()
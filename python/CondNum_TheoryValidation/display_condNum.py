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
def getConNumColor(condNum):
        color = "white"
        if condNum > 70000.0:
            color = 'linen'
        elif condNum > 60000.0:
            color = 'antiquewhite'
        elif condNum > 50000.0:
            color = 'papayawhip'
        elif condNum > 40000.0:
            color = 'oldlace'
        elif condNum > 30000.0:
            color = 'cornsilk'
        elif condNum > 20000.0:
            color = 'palegoldenrod'
        elif condNum > 10000.0:
            color = 'yellow'
        elif condNum > 8000.0:
            color = 'lightblue'
        elif condNum > 6000.0:
            color = 'deepskyblue'
        elif condNum > 4000.0:
            color = 'red'
        elif condNum > 2000.0:
            color = 'green'
        elif condNum > 1000.0:
            color = 'maroon'
        else:
            color = 'black'

        return color


def displayCondNumDistribution(m):
    """"Display distribution of cond num for cam distribution in 3D"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100


    for i in range(0,m.shape[1]):
        x = m[0][i]
        y = m[1][i]
        z = m[2][i]
        condNum = m[3][i]
        color = getConNumColor(condNum)
        ax.scatter(x, y, z, s = 100,c = color, marker="o")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    plt.pause(1000)





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
# matrix = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1000,20000,5000,50000]])
# displayCondNumDistribution(matrix)
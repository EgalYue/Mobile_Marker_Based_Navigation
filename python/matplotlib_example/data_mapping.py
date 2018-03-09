#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 09.03.18 09:19
@File    : data_mapping.py
@author: Yue Hu
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
np.random.seed(19680801)

N = 50
x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]
colors = [1,2,3,4,5,6,7,8,9,10]

area = np.pi * (15 * np.random.rand(N))**2
# norm = colors.Normalize(vmin= -0.5, vmax= 1.0)
plt.scatter(x, y, s=area, c=colors, alpha=0.5,cmap="magma")
plt.colorbar()
plt.show()



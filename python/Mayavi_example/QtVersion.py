#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 05.05.18 00:18
@File    : QtVersion.py
@author: Yue Hu
"""
import inspect
from PyQt5 import Qt

vers = ['%s = %s' % (k,v) for k,v in vars(Qt).items() if k.lower().find('version') >= 0 and not inspect.isbuiltin(v)]
print('\n'.join(sorted(vers)))
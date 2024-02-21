# -*- coding: UTF-8 -*-
"""
@Project: MiSleep 
@File: sicpy_siganl_flitfilt_test.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2024/1/16 15:01 
@Description:  
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

t = np.linspace(0, 0.1, 2001)
low_hz = 5
high_hz = 250
# 5 Hz
x_low = np.sin(2 * np.pi * low_hz * t)
# 250 Hz
x_high = np.sin(2 * np.pi * high_hz * t)
x = x_low + x_high

b, a = signal.butter(8, 0.125)

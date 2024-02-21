# -*- coding: UTF-8 -*-
"""
@Project: MiSleep 
@File: scipy_signal_welch_test.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2024/1/16 9:45 
@Description:  
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

rng = np.random.default_rng()

# Sine wave's frequency is 1234 Hz, and
# data sampling frequency is 10e3 Hz, which is the sampling frequency of white noise
# totally 1e5 data points
# So totally 10 seconds
fs = 10e3
N = 1e5
# 2 Vrms sine wave
amp = 2*np.sqrt(2)
freq = 1234.
# Corrupted by 0.001 V**2/Hz of white noise sampled at 10 kHz
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
# 2*pi is 1 Hz
x = amp*np.sin(2*np.pi*freq*time)
# Add white noise
x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

# Compute and plot the power spectral density
f, Pxx_den = signal.welch(x, fs, nperseg=10000)


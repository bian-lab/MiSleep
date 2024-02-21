# -*- coding: UTF-8 -*-
"""
@Project: MiSleep 
@File: feature_extraction.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2023/10/11 15:30 
@Description:  extract features from raw eeg data and emg data, compare their power spectrum and each components'
                features
"""
from hdf5storage import loadmat

from utils import get_sleep_stage_labels
import matplotlib.pyplot as plt
import numpy as np

data = list(loadmat(
    r'E:\workplace\EEGProcessing\00_DATA\20230922_EEG_test\20230922_grounding_EEGdata.mat').values())[-1].T

# Get sleep stage label into sec format
sleep_stages_label = get_sleep_stage_labels("E:/test_label.txt")

EEG = data[1]
EMG = data[2]
label = sleep_stages_label

epoch_length = 5
SR = 305

# Get epoch data and epoch sleep stage label
epoch_EEG = [EEG[i:i + epoch_length * SR] for i in range(0, len(EEG), epoch_length * SR)][:-1]
epoch_EMG = [EMG[i:i + epoch_length * SR] for i in range(0, len(EMG), epoch_length * SR)][:-1]
epoch_label = [label[i:i + epoch_length] for i in range(0, len(label), epoch_length)][:-1]
# Use the highest occurrence label during the epoch length as the epoch label
epoch_label = [max(set(each), key=each.count) for each in epoch_label]

# ----Calculate standard deviation for EEG and EMG----
epoch_EEG_SD = [np.var(each) for each in epoch_EEG]
epoch_EMG_SD = [np.var(each) for each in epoch_EMG]

plt.boxplot(epoch_EEG_SD)
plt.scatter(epoch_EEG_SD, epoch_EMG_SD)
plt.show()

plt.step(range(len(epoch_label)), epoch_label)
plt.show()



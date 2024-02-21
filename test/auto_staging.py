# -*- coding: UTF-8 -*-
"""
@Project: MiSleep 
@File: auto_staging.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2023/10/9 16:00 
@Description:  auto sleep staging
"""
import numpy as np
from hdf5storage import loadmat
import matplotlib.pyplot as plt

from utils import get_epoch_spectrum, get_ave_bands

data = list(loadmat(
    r'E:\workplace\EEGProcessing\EEG script & dataset\20230922_non-shielding wires w plug & grounding\20230922_grounding_EEGdata.mat').values())[
    -1].T
label_file = open(r'E:/test_label.txt', 'r')
labels = [each.replace("\n", "") for each in label_file.readlines()]
sleep_stages = labels[labels.index("==========Sleep stage==========") + 1:]
temp_sleep_stage_labels = [[int(each.split(', ')[1]), int(each.split(', ')[4]), int(each.split(', ')[6])]
                           for each in sleep_stages]
sleep_stage_labels = []
for each in temp_sleep_stage_labels:
    sleep_stage_labels += [each[2] for sec in range(each[0], each[1] + 1)]

EEG = data[1]
EMG = data[2]
labels = sleep_stage_labels

# construct training data and label, 5-s epoch
epoch_length = 20
SR = 305

# [:-1] can delete the incomplete data in the end of list
epoch_label = [labels[i:i + epoch_length] for i in range(0, len(labels), epoch_length)][:-1]
# Use the highest occurrence label during the epoch length as the epoch label
epoch_label = [max(set(each), key=each.count) for each in epoch_label]
epoch_EEG = [EEG[i:i + epoch_length * SR] for i in range(0, len(EEG), epoch_length * SR)][:-1]
epoch_EMG = [EMG[i:i + epoch_length * SR] for i in range(0, len(EMG), epoch_length * SR)][:-1]

# epoch eeg spectrum
# split to 5 bands:
# Delta: 0.5~4 HZ
# Theta: 4~7 HZ
# Alpha: 7~12 HZ
# Beta: 12~30 HZ
# Gamma: 30~70 HZ
# epoch_EEG_spectrum =
epoch_eeg_spectrum = get_epoch_spectrum(data=epoch_EEG, SR=SR)
epoch_band_power_ave_percentage = np.array([get_ave_bands(each[0], each[1]) for each in epoch_eeg_spectrum]).T

figure = plt.figure(figsize=(30, 40))
axs = figure.subplots(nrows=3, ncols=1)
x = list(range(len(epoch_label)))

axs[0].plot(x, epoch_band_power_ave_percentage[0], linewidth=1, color='blue')
axs[0].plot(x, epoch_band_power_ave_percentage[1], linewidth=1, color='orange')
axs[0].plot(x, epoch_band_power_ave_percentage[2], linewidth=1, color='green')
axs[0].plot(x, epoch_band_power_ave_percentage[3], linewidth=1, color='red')
axs[0].plot(x, epoch_band_power_ave_percentage[4], linewidth=1, color='purple')
axs[0].legend(['delta', 'theta', 'alpha', 'beta', 'gamma'], loc='upper right')

p_delta = axs[1].bar(x, epoch_band_power_ave_percentage[0], color='blue')
p_theta = axs[1].bar(x, epoch_band_power_ave_percentage[1], bottom=sum(epoch_band_power_ave_percentage[0:1]),
                     color='orange')
p_alpha = axs[1].bar(x, epoch_band_power_ave_percentage[2], bottom=sum(epoch_band_power_ave_percentage[0:2]),
                     color='green')
p_beta = axs[1].bar(x, epoch_band_power_ave_percentage[3], bottom=sum(epoch_band_power_ave_percentage[0:3]),
                    color='red')
p_gamma = axs[1].bar(x, epoch_band_power_ave_percentage[4], bottom=sum(epoch_band_power_ave_percentage[0:4]),
                     color='purple')
# axs[0].set_xticks([])
# axs[0].set_yticks([])
axs[2].step(x, epoch_label, where="mid", linewidth=1)
# axs[1].set_xticks([each*epoch_length for each in x], roration=45)
# axs[1].set_yticks(range(1, 5), ['NREM', 'REM', 'Wake', 'INIT'])
figure.savefig('E:/aaaaaaa.eps', dpi=350)
# plt.show()

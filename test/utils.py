# -*- coding: UTF-8 -*-
"""
@Project: MiSleep 
@File: utils.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2023/10/11 15:36 
@Description:  
"""

import numpy as np
from scipy.signal import welch


def get_epoch_spectrum(data, SR):
    # lowpass the data to 70HZ
    # fnorm = np.array(70 / (.5 * SR))
    # b, a = butter(3, fnorm, btype='lowpass')
    #
    # filtered_data = signal.filtfilt(b, a, data)
    epoch_spectrum = []
    for each in data:
        spectrum_F, spectrum_P = welch(each, fs=SR, nperseg=SR * 4)
        epoch_spectrum.append([spectrum_F, spectrum_P])
    return epoch_spectrum


def get_ave_bands(F, P):
    """
    Get average band spectrum for each band
    # Delta: 0.5~4 HZ
    # Theta: 4~7 HZ
    # Alpha: 7~12 HZ
    # Beta: 12~30 HZ
    # Gamma: 30~70 HZ
    :param F: frequency list
    :param P: power spectrum list
    :return:
    """

    band_dict = {
        'delta': [0.5, 4],
        'theta': [4, 7],
        'alpha': [7, 12],
        'beta': [12, 30],
        'gamma': [30, 70]
    }

    band_ave_power_dict = {}
    for key, value in band_dict.items():
        _idx = [np.where(F == value[0])[0][0], np.where(F == value[1])[0][0]]
        band_ave_power_dict[key] = np.average(P[_idx[0]: _idx[1]])

    band_ave_power = list(band_ave_power_dict.values())
    return np.array([each / sum(band_ave_power) for each in band_ave_power])


def get_sleep_stage_labels(label_path):
    """
    Get labels as the standard format, label per second
    :param label_path:
    :return:
    """

    label_file = open(label_path, 'r')
    labels = [each.replace("\n", "") for each in label_file.readlines()]
    sleep_stages = labels[labels.index("==========Sleep stage==========") + 1:]
    temp_sleep_stage_labels = [[int(each.split(', ')[1]), int(each.split(', ')[4]), int(each.split(', ')[6])]
                               for each in sleep_stages]
    sleep_stage_labels = []
    for each in temp_sleep_stage_labels:
        sleep_stage_labels += [each[2] for sec in range(each[0], each[1] + 1)]

    return sleep_stage_labels

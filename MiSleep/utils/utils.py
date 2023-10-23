# -*- coding: UTF-8 -*-
"""
@Project: EEGProcessing_V3 
@File: utils.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2023/9/1 15:14 
@Description:  
"""

import datetime
from itertools import groupby

import numpy as np
from scipy.signal import welch


def second2time(second, ac_time):
    """
    Pass second, return time format %d:%H:%M:%S from acquisition time
    :param second:
    :param ac_time: acquisition time
    :return:
    """

    return (ac_time + datetime.timedelta(seconds=second)).strftime("%d:%H:%M:%S")


def time_delta2second(str_time1, str_time2):
    str_time1 = [int(each) for each in str_time1.split(":")]
    str_time2 = [int(each) for each in str_time2.split(":")]
    return int((datetime.datetime(2000, 1, str_time2[0], str_time2[1], str_time2[2], str_time2[3]) -
                datetime.datetime(2000, 1, str_time1[0], str_time1[1], str_time1[2], str_time1[3])).total_seconds())


def str_time2second(str_time):
    """

    :param str_time: dd:HH:MM:SS
    :return:
    """

    return int((datetime.datetime.strptime(str_time, "y%/m%/%d %H:%M:%S") -
                datetime.datetime(1900, 1, 1, 0, 0, 0)).total_seconds())


def lst2group(pre_lst):
    """
    Convert a list like [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 1], [7, 1], [8, 1], [9, 3], [10, 3]] to
    [[1, 5, 2], [6, 8, 1], [9, 10, 3]]
    :param pre_lst:
    :return:
    """

    # Convert to [[[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]], [[6, 1], [7, 1], [8, 1]], [[9, 3], [10, 3]]]
    pre_lst = [list(group) for idx, group in groupby(pre_lst, key=lambda x: x[1])]

    # Convert to [[1, 5, 2], [6, 8, 1], [9, 10, 3]]
    return [[each[0][0], each[-1][0], each[0][1]] for each in pre_lst]


def get_4_stages(sleep_label_lst, data, SR):
    """
    Divide the selected data into 3 stages
    :param sleep_label_lst: Sleep stage label, format: [[1, 2], [2, 2], [3, 2], ...]
    :param data: selected data, format: [[channel_1 data], [channel_2 data], ...]
    :param SR: Sampling data, int
    :return:
    """

    # Group labels
    labels = lst2group(sleep_label_lst)
    channel_num = len(data)

    NREM_data = [[] for _ in range(channel_num)]
    REM_data = [[] for _ in range(channel_num)]
    Wake_data = [[] for _ in range(channel_num)]
    Init_data = [[] for _ in range(channel_num)]

    for each in labels:
        # NREM sleep stage
        if each[2] == 1:
            for i in range(channel_num):
                NREM_data[i] += list(data[i][each[0] * SR: (each[1] + 1) * SR])

        # REM sleep stage
        if each[2] == 2:
            for i in range(channel_num):
                REM_data[i] += list(data[i][each[0] * SR: (each[1] + 1) * SR])

        # Wake sleep stage
        if each[2] == 3:
            for i in range(channel_num):
                Wake_data[i] += list(data[i][each[0] * SR: (each[1] + 1) * SR])

        # Initial sleep stage
        if each[2] == 4:
            for i in range(channel_num):
                Init_data[i] += list(data[i][each[0] * SR: (each[1] + 1) * SR])

    return NREM_data, REM_data, Wake_data, Init_data


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
    return band_ave_power

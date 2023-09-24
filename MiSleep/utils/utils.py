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
import scipy


def second2time(second):
    """
    Pass second, return time format %d:%H:%M:%S from 00:00:00:00
    :param second:
    :return:
    """

    return (datetime.datetime(2000, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=second)).strftime("%d:%H:%M:%S")


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


def get_3_stages(sleep_label_lst, data, SR):
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

    return NREM_data, REM_data, Wake_data

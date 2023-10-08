# -*- coding: UTF-8 -*-
"""
@Project: MiSleep 
@File: test_transfer.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2023/10/7 17:06 
@Description: Transfer the label.txt into .xlsx file
"""
from datetime import datetime
import pandas as pd

label_path = r'E:\workplace\EEGProcessing\MiSleep\MiSleep\test\test_label.txt'

label_file = open(label_path, 'r+')
labels = [each.replace("\n", "") for each in label_file.readlines()]
ac_time = datetime.strptime(labels[3].split(": ")[1], "%Y-%m-%d %H:%M:%S")
sleep_stages = labels[labels.index("==========Sleep stage==========") + 1:]
sleep_stages = [each.split(', ') for each in sleep_stages]
sleep_stages_df = pd.DataFrame(sleep_stages,
                               columns=['start_time', 'start_sec', 'start/end', 'end_time', 'end_sec', 'start/end',
                                        'phase_code', 'phase'])

sleep_stages_df['epoch_duration(s)'] = sleep_stages_df.apply(lambda x: int(x[4])-int(x[1]), axis=1)

group_ = sleep_stages_df.groupby('phase')
stage_dict = {}
for key, value in group_:
    stage_dict[key] = value

wake_df = stage_dict['Wake']

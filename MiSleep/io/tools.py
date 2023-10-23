# -*- coding: UTF-8 -*-
"""
@Project: MiSleep 
@File: tools.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2023/10/9 13:52 
@Description:  
"""
import copy
import datetime

import pandas as pd
import matplotlib.pyplot as plt


def add_hour_marker(df, ac_time):
    """
    Add hour marker into dataframe
    :param df:
    :param ac_time:
    :return:
    """
    datetime_lst = pd.date_range(start=df['start_time'].iloc[0].replace(minute=0, second=0),
                                 end=(df['end_time'].iloc[-1] + datetime.timedelta(hours=1)).replace(
                                     minute=0, second=0), freq='H')
    idx_lst = 0
    temp_df = pd.DataFrame([], columns=df.columns)
    for idx_df, row in df.iterrows():
        # insert before current line
        if row['start_time'] > datetime_lst[idx_lst]:
            _time = datetime_lst[idx_lst]
            _sec = int((_time - ac_time).total_seconds())
            _row = pd.DataFrame([[_time, _sec, 5, _time, _sec, 5, ' ', 'Marker', ' ']], columns=df.columns)
            temp_df = pd.concat([temp_df, _row], ignore_index=True)
            temp_df = pd.concat([temp_df, pd.DataFrame(row).T], ignore_index=True)
            idx_lst += 1

        # add a new line and insert between them, if the bout cross the hour
        elif row['start_time'] < datetime_lst[idx_lst] < row['end_time']:
            _time = datetime_lst[idx_lst]
            _sec = int((_time - ac_time).total_seconds())
            _row = pd.DataFrame([[_time, _sec, 5, _time, _sec, 5, ' ', 'Marker', ' ']], columns=df.columns)

            temp_row = copy.deepcopy(row)

            temp_row['end_time'] = _time
            temp_row['end_sec'] = _sec
            temp_df = pd.concat([temp_df, pd.DataFrame(temp_row).T], ignore_index=True)
            temp_df = pd.concat([temp_df, _row], ignore_index=True)

            temp_row = copy.deepcopy(row)
            temp_row['start_time'] = _time
            temp_row['start_sec'] = _sec
            temp_df = pd.concat([temp_df, pd.DataFrame(temp_row).T], ignore_index=True)
            idx_lst += 1
        else:
            temp_df = pd.concat([temp_df, pd.DataFrame(row).T], ignore_index=True)

    # Add last line
    _time = datetime_lst[-1]
    _sec = int(abs((_time - ac_time).total_seconds()))

    _row = pd.DataFrame([[_time, _sec, 5, _time, _sec, 5, ' ', 'Marker', ' ']], columns=df.columns)
    temp_df = pd.concat([temp_df, _row], ignore_index=True)
    return temp_df


class Transfer:
    def __init__(self, label_file_path=None):
        """
        Transfer the label file into operability excel format
        :param label_file: label file is .txt format label file, should create by MiSleep
        """

        self.label_file_path = label_file_path
        self.ac_time = None
        self.sleep_stages_df = pd.DataFrame()
        self.group_stages = None
        self.sleep_stages_label = []

    def get_params(self):
        """
        Get params from pass in data
        :return:
        """

        label_file = open(self.label_file_path, 'r+')
        labels = [each.replace("\n", "") for each in label_file.readlines()]
        label_file.close()
        self.ac_time = datetime.datetime.strptime(labels[3].split(": ")[1], "%Y-%m-%d %H:%M:%S")
        sleep_stages = labels[labels.index("==========Sleep stage==========") + 1:]
        self.sleep_stages_label = [each.split(', ') for each in sleep_stages]
        self.sleep_stages_df = pd.DataFrame(self.sleep_stages_label,
                                            columns=['start_time', 'start_sec', 'start/end', 'end_time', 'end_sec',
                                                     'start/end',
                                                     'phase_code', 'phase'])

        self.sleep_stages_df['epoch_duration(s)'] = self.sleep_stages_df.apply(lambda x: int(x[4]) - int(x[1]), axis=1)
        self.sleep_stages_df['start_time'] = self.sleep_stages_df['start_sec'].apply(
            lambda x: self.ac_time + datetime.timedelta(seconds=int(x)))
        self.sleep_stages_df['end_time'] = self.sleep_stages_df['end_sec'].apply(
            lambda x: self.ac_time + datetime.timedelta(seconds=int(x)))
        self.sleep_stages_df['start_sec'] = self.sleep_stages_df['start_sec'].astype(int)
        self.sleep_stages_df['end_sec'] = self.sleep_stages_df['end_sec'].astype(int)
        self.sleep_stages_df['start/end'] = self.sleep_stages_df['start/end'].astype(int)
        self.sleep_stages_df['phase_code'] = self.sleep_stages_df['phase_code'].astype(int)

        self.group_stages = self.sleep_stages_df.groupby('phase')

    def analyze_stage_df(self, df):
        df = df.reset_index(drop=True)
        bout_number = df.shape[0]
        total_time = sum(df['epoch_duration(s)'])
        ave_epoch_length = round(total_time / bout_number, 4)

        features = ['Bout number', bout_number, ' ', 'Total time', total_time, ' ', 'Ave epoch length',
                    ave_epoch_length]

        df = add_hour_marker(df, ac_time=self.ac_time)
        try:
            df['features'] = features + [' '] * (df.shape[0] - len(features))
        except Exception as e:
            print(e)
            df['features'] = [' '] * df.shape[0]

        return df

    def save_transfer_results(self, save_path):
        """
        Save the analyzed results
        :return:
        """

        sleep_stages_df = add_hour_marker(self.sleep_stages_df, ac_time=self.ac_time)

        # write into Excel
        writer = pd.ExcelWriter(save_path, datetime_format='yyyy-mm-dd hh:mm:ss')
        sleep_stages_df.to_excel(excel_writer=writer, sheet_name='All', index=False)
        for df_name, df in self.group_stages:
            _df = self.analyze_stage_df(df)
            _df.to_excel(excel_writer=writer, sheet_name=df_name, index=False)

        writer.close()


class Hypnogram:
    """
    Draw hypnogram
    """

    def __init__(self, label_file_path=None, line_color=None, start_sec=0, end_sec=0, title=''):
        """
        Pass in label file path, draw hypnogram
        Can custom some features
        1. Define line color
        2. Select plot range by entering start and end second
        3. Define title

        """

        self.label_file_path = label_file_path
        self.sleep_stages_label = []
        self.line_color = line_color
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.title = title

    def get_params(self):
        """
        Get pass in params
        :return:
        """

        label_file = open(self.label_file_path, 'r+')
        labels = [each.replace("\n", "") for each in label_file.readlines()]
        label_file.close()
        sleep_stages = labels[labels.index("==========Sleep stage==========") + 1:]
        self.sleep_stages_label = [each.split(', ') for each in sleep_stages]

    def draw_save_hypnogram(self, save_path):
        """
        Use sleep stage labels for hypnogram
        :return:
        """

        # Draw hypnogram
        temp_sleep_stage_labels = [[int(each[1]), int(each[4]), int(each[6])]
                                   for each in self.sleep_stages_label]
        _stage_labels = []
        for each in temp_sleep_stage_labels:
            _stage_labels += [[sec, each[2]] for sec in range(each[0], each[1] + 1)]

        # end sec = 99 means plot all data
        if self.end_sec != 99:
            _stage_labels = _stage_labels[self.start_sec: self.end_sec]

        total_second = len(_stage_labels)

        figure = plt.figure(figsize=(30, 5))
        ax = figure.subplots(nrows=1, ncols=1)
        ax.step(range(total_second), [each[1] for each in _stage_labels], where="mid", linewidth=1,
                color=self.line_color)
        ax.set_ylim(0.9, 3.1)
        ax.set_xlim(0, total_second - 1)
        ax.set_xticks([each for each in range(0, total_second, int(total_second / 10))],
                      [each for each in range(0, total_second, int(total_second / 10))], fontsize=15, weight='bold')
        ax.set_yticks(range(1, 4), ['NREM', 'REM', 'Wake'], fontsize=15, weight='bold')
        ax.set_xlabel("Time(Sec)", fontsize=15, weight='bold')
        ax.set_ylabel("Stage", fontsize=15, weight='bold')
        ax.set_title(self.title, fontsize=15, weight='bold')
        # plt.show()
        figure.savefig(save_path)

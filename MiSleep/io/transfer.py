# -*- coding: UTF-8 -*-
"""
@Project: MiSleep 
@File: transfer.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2023/10/9 13:52 
@Description:  
"""
import copy
import datetime

import pandas as pd


class Transfer:
    def __init__(self, label_file=None):
        """
        Transfer the label file into operability excel format
        :param label_file: label file is .txt format label file, should create by MiSleep
        """

        self.label_file = label_file
        self.ac_time = None
        self.sleep_stages_df = pd.DataFrame()
        self.group_stages = None

    def get_params(self):
        """
        Get params from pass in data
        :return:
        """

        labels = [each.replace("\n", "") for each in self.label_file.readlines()]
        self.ac_time = datetime.datetime.strptime(labels[3].split(": ")[1], "%Y-%m-%d %H:%M:%S")
        sleep_stages = labels[labels.index("==========Sleep stage==========") + 1:]
        sleep_stages = [each.split(', ') for each in sleep_stages]
        self.sleep_stages_df = pd.DataFrame(sleep_stages,
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

    def add_hour_marker(self, df, ac_time):
        """
        Add hour marker into dataframe
        :param df:
        :param ac_time:
        :return:
        """
        datetime_lst = pd.date_range(start=df['start_time'].iloc[0].replace(minute=0, second=0),
                                     end=df['end_time'].iloc[-1].replace(hour=df['end_time'].iloc[-1].hour + 1,
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

    def analyze_df(self, df):
        df = df.reset_index(drop=True)
        bout_number = df.shape[0]
        total_time = sum(df['epoch_duration(s)'])
        ave_epoch_length = round(total_time / bout_number, 4)

        features = ['Bout number', bout_number, ' ', 'Total time', total_time, ' ', 'Ave epoch length',
                    ave_epoch_length]

        df = self.add_hour_marker(df, ac_time=self.ac_time)
        try:
            df['features'] = features + [' '] * (df.shape[0] - len(features))
        except Exception as e:
            print(e)
            df['features'] = [' '] * df.shape[0]

        return df

    def save(self, save_path):
        """
        Save the analyzed results
        :return:
        """

        sleep_stages_df = self.add_hour_marker(self.sleep_stages_df, ac_time=self.ac_time)

        # write into Excel
        writer = pd.ExcelWriter(save_path, datetime_format='yyyy-mm-dd hh:mm:ss')
        sleep_stages_df.to_excel(excel_writer=writer, sheet_name='All', index=False)
        for df_name, df in self.group_stages:
            _df = self.analyze_df(df)
            _df.to_excel(excel_writer=writer, sheet_name=df_name, index=False)

        writer.close()

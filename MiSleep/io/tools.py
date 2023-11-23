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
pd.options.mode.chained_assignment = None


def insert_row(df, idx, row):
    """
    Insert a row in the specific index "idx" for df
    :param df:
    :type df:
    :param idx:
    :type idx:
    :param row:
    :type row:
    :return:
    :rtype:
    """
    above = df[:idx]
    below = df[idx:]
    return pd.concat([above, row, below], axis=0)


def add_hour_marker(df, ac_time):
    """
    Add hour marker into dataframe
    :param df:
    :param ac_time:
    :return:
    """

    date_range_df = pd.DataFrame(columns=df.columns)
    # Construct a date range with Marker, then sort by start sec, to locate the marker position
    date_range_df["start_time"] = pd.date_range(start=df["start_time"].iloc[0].replace(minute=0, second=0),
                                                end=(df['end_time'].iloc[-1] + datetime.timedelta(
                                                    hours=1)).replace(
                                                    minute=0, second=0), freq='H')
    date_range_df["end_time"] = date_range_df["start_time"]
    date_range_df["start_sec"] = date_range_df["start_time"].apply(lambda x: int((x - ac_time).total_seconds()))
    date_range_df["end_sec"] = date_range_df["end_time"].apply(lambda x: int((x - ac_time).total_seconds()))
    date_range_df["phase_code"] = [5] * date_range_df.shape[0]
    date_range_df["phase"] = ["Marker"] * date_range_df.shape[0]

    new_df = pd.concat([df, date_range_df], ignore_index=True)
    del df
    new_df = new_df.sort_values(["start_sec"]).reset_index(drop=True)


    # Compare the previous row with the Marker row, if the end_sec larger than the Marker row, add a row after the
    # Marker row
    changed = 1
    while changed == 1:
        changed = 0
        for idx, row in new_df[new_df["phase"] == "Marker"].iterrows():
            # If
            if idx >= 1 and row["start_sec"] < new_df["end_sec"].iloc[idx - 1]:
                # Add a new row
                row_ = copy.deepcopy(new_df.iloc[idx - 1])
                row_["start_time"] = row["start_time"]
                row_["start_sec"] = row["start_sec"]
                new_df = insert_row(new_df, idx + 1, pd.DataFrame(row_).T)

                # Update the previous row
                new_df["end_sec"].iloc[idx - 1] = row["start_sec"]
                new_df["end_time"].iloc[idx - 1] = row["start_time"]
                changed = 1

                new_df = new_df.reset_index(drop=True)

                # If inserted a row, reloop the whole
                break

    new_df['epoch_duration(s)'] = new_df.apply(lambda x: int(x[4]) - int(x[1]), axis=1)
    return new_df

def analyze_phases(hour_marker_df):
    """
    Use the dataframe with hour marker, analyse each phase per hour
    :param hour_marker_df:
    :type hour_marker_df:
    :return: analysed dataframe, with NREM, REM, WAKE, INIT duration, bout number, ave bout duration
    :rtype:
    """
    # Analyse duration and bout number for each phase
    analyse_df = pd.DataFrame()
    analyse_df['date_time'] = pd.date_range(start=hour_marker_df['start_time'].iloc[0],
                                            end=hour_marker_df['end_time'].iloc[-1], freq='H')[:-1]
    temp_df = hour_marker_df.drop(hour_marker_df[hour_marker_df['phase'] == "Marker"].index)
    temp_df["hour_marker"] = temp_df['start_time'].apply(lambda x: x.day * 100 + x.hour)
    features = []
    # NREM_duration, NREM_bout, REM_duration, REM_bout, WAKE_duration, WAKE_bout, INIT_duration, INIT_bout
    for each in temp_df.groupby('hour_marker'):
        df = each[1]
        temp_lst = []
        for phase in ["NREM", "REM", "Wake", "INIT"]:
            _duration = df[df["phase"] == phase]["epoch_duration(s)"].sum()
            _bout = df[df["phase"] == phase]["epoch_duration(s)"].count()
            temp_lst += [_duration, _bout, round(_duration / _bout, 2) if _bout != 0 else 0, round(_duration / 3600, 2)]
        features.append(temp_lst)

    analyse_df[['NREM_duration', 'NREM_bout', "NREM_ave", "NREM_percentage", 'REM_duration', 'REM_bout', "REM_ave",
                "REM_percentage", 'WAKE_duration', 'WAKE_bout', "WAKE_ave", "WAKE_percentage", 'INIT_duration',
                'INIT_bout', "INIT_ave", "INIT_percentage"]] = features
    analyse_df[
        ['NREM_duration', 'NREM_bout', 'REM_duration', 'REM_bout', 'WAKE_duration', 'WAKE_bout', 'INIT_duration',
         'INIT_bout']] = analyse_df[
        ['NREM_duration', 'NREM_bout', 'REM_duration', 'REM_bout', 'WAKE_duration', 'WAKE_bout', 'INIT_duration',
         'INIT_bout']].astype(int)

    return analyse_df


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

        analyse_df = analyze_phases(hour_marker_df=sleep_stages_df)

        # write into Excel
        writer = pd.ExcelWriter(save_path, datetime_format='yyyy-mm-dd hh:mm:ss')
        pd.concat([sleep_stages_df, analyse_df], axis=1).to_excel(excel_writer=writer, sheet_name='All', index=False)
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

# -*- coding: UTF-8 -*-
"""
@Project: MiSleep 
@File: tools.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2023/10/7 17:06 
@Description: Transfer the label.txt into .xlsx file
"""
import copy
from datetime import datetime, timedelta
import pandas as pd
from matplotlib import pyplot as plt


def insert_df(df, idx, df_add):
    df1 = df.iloc[:idx, :]
    df2 = df.iloc[idx:, :]
    df_new = pd.concat([df1, df_add, df2], ignore_index=True)
    return df_new


def add_hour_marker(df, ac_time):
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


def analyze_df(df):
    df = df.reset_index(drop=True)
    bout_number = df.shape[0]
    total_time = sum(df['epoch_duration(s)'])
    ave_epoch_length = round(total_time / bout_number, 4)

    features = ['Bout number', bout_number, ' ', 'Total time', total_time, ' ', 'Ave epoch length', ave_epoch_length]

    df = add_hour_marker(df, ac_time=ac_time)
    try:
        df['features'] = features + [' '] * (df.shape[0] - len(features))
    except Exception as e:
        print(e)
        df['features'] = [' '] * df.shape[0]

    return df


label_path = r'E:\workplace\EEGProcessing\00_DATA\20230922_non-shielding wires w plug & grounding\Label\20230922_scoring.txt'

label_file = open(label_path, 'r+')
labels = [each.replace("\n", "") for each in label_file.readlines()]
ac_time = datetime.strptime(labels[3].split(": ")[1], "%Y-%m-%d %H:%M:%S")
sleep_stages = labels[labels.index("==========Sleep stage==========") + 1:]
sleep_stages = [each.split(', ') for each in sleep_stages]
sleep_stages_df = pd.DataFrame(sleep_stages,
                               columns=['start_time', 'start_sec', 'start/end', 'end_time', 'end_sec', 'start/end',
                                        'phase_code', 'phase'])

sleep_stages_df['epoch_duration(s)'] = sleep_stages_df.apply(lambda x: int(x[4]) - int(x[1]), axis=1)
sleep_stages_df['start_time'] = sleep_stages_df['start_sec'].apply(lambda x: ac_time + timedelta(seconds=int(x)))
sleep_stages_df['end_time'] = sleep_stages_df['end_sec'].apply(lambda x: ac_time + timedelta(seconds=int(x)))
sleep_stages_df['start_sec'] = sleep_stages_df['start_sec'].astype(int)
sleep_stages_df['end_sec'] = sleep_stages_df['end_sec'].astype(int)
sleep_stages_df['start/end'] = sleep_stages_df['start/end'].astype(int)
sleep_stages_df['phase_code'] = sleep_stages_df['phase_code'].astype(int)

group_ = sleep_stages_df.groupby('phase')
sleep_stages_df = add_hour_marker(sleep_stages_df, ac_time=ac_time)

# write into Excel
writer = pd.ExcelWriter("E:/data.xlsx", datetime_format='yyyy-mm-dd hh:mm:ss')
sleep_stages_df.to_excel(excel_writer=writer, sheet_name='All', index=False)
stage_dict = {}
for df_name, df in group_:
    _df = analyze_df(df)
    _df.to_excel(excel_writer=writer, sheet_name=df_name, index=False)

# writer.save()
writer.close()

# Draw hypnogram
temp_sleep_stage_labels = [[int(each[1]), int(each[4]), int(each[6])]
                           for each in sleep_stages]
sleep_stage_labels = []
for each in temp_sleep_stage_labels:
    sleep_stage_labels += [[sec, each[2]] for sec in range(each[0], each[1] + 1)]
total_second = len(sleep_stage_labels[:1000])
sleep_stage_labels = sleep_stage_labels[:1000]

figure = plt.figure(figsize=(30, 5))
ax = figure.subplots(nrows=1, ncols=1)
ax.step(range(total_second), [each[1] for each in sleep_stage_labels], where="mid", linewidth=1)
ax.set_ylim(0.9, 3.1)
ax.set_xlim(0, total_second - 1)
ax.set_xticks([each for each in range(0, total_second, int(total_second / 10))],
              [each for each in range(0, total_second, int(total_second / 10))], fontsize=15, weight='bold')
ax.set_yticks(range(1, 4), ['NREM', 'REM', 'Wake'], fontsize=15, weight='bold')
ax.set_xlabel("Time(Sec)", fontsize=15, weight='bold')
ax.set_ylabel("Stage", fontsize=15, weight='bold')
ax.set_title("Hypnogram", fontsize=15, weight='bold')
# plt.show()
figure.savefig("E:/hypnogram.pdf")

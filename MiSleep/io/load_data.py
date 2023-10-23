# -*- coding: UTF-8 -*-
"""
@Project: EEGProcessing_V2 
@File: load_data.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2023/7/29 17:37 
@Description:  
"""

import datetime
import sys

from math import ceil

from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from hdf5storage import loadmat
from MiSleep.gui.load_data.load_data import Ui_MiSleep
from MiSleep.io.auto_stage import Auto_stage
from MiSleep.plot.MiSleep import sleep
from MiSleep.utils.utils import second2time, lst2group
from MiSleep.io.tools import Transfer, Hypnogram
import joblib


class load_gui(QMainWindow, Ui_MiSleep):
    def __init__(self, parent=None):
        super(load_gui, self).__init__(parent)
        self.setupUi(self)

        self.dateTimeEdit.setDateTime(datetime.datetime.now())
        self.ASDateTimeEditor.setDateTime(datetime.datetime.now())

        self.data_path = ''
        self.label_path = ''
        self.SR = 256
        self.epoch_length = 5
        self.channel_num = 2
        self.acquisition_time = ''
        self.total_seconds = 0

        self.dataSelectBt.clicked.connect(self.get_data_path)
        self.labelSelectBt.clicked.connect(self.get_label_path)

        self.auto_stage_data_path = None
        self.ASdataSelectBt.clicked.connect(self.get_data_for_auto_staging)
        self.AS_data = None
        self.autoStagingBt.clicked.connect(self.auto_staging)

        self.selectFileBt.clicked.connect(self.get_label_for_transfer)
        self.transferBt.clicked.connect(self.transfer)
        self.hypnogramBt.clicked.connect(self.hypnogram)

        # Press check button and check data
        self.checkBt.clicked.connect(self.check)

        self.data = []
        self.label_file = []  # Labels for saving format
        self.labels = []  # Labels format, including three part, marker label, start end label, sleep stage label
        self.data_length = 0

        # Model list for auto staging
        self.model_lst = {
            0: 'lightGBM_1EEG_1EMG'
        }

        self.stage_type_dict = {1: 'NREM', 2: 'REM', 3: 'Wake', 4: 'INIT'}

    def get_data_path(self):
        """
        get data path from QFileDialog, select a path and add to the dataPathEdit
        :return:
        """

        self.data_path, _ = QFileDialog.getOpenFileName(self, 'Select data file',
                                                        r'E:/', 'Matlab Files (*.mat *.MAT)')
        self.dataPathEdit.setText(self.data_path)

    def get_label_path(self):
        """
        get label path from QFileDialog, same with self.get_data_path, the label file can be empty
        :return:
        """

        self.label_path, _ = QFileDialog.getOpenFileName(self, 'Select label file',
                                                         r'E:/', 'txt Files (*.txt *.TXT)')
        self.labelPathEdit.setText(self.label_path)

    def check(self):
        """
        Initialize check class, check the following input features:
        1. Whether data path exist
        2. Whether label path exist
        3. If channel number same with data channel numbers

        :return:
        """

        # get value from editors
        # self.acquisition_time = self.dateTimeEdit.dateTime()
        self.SR = self.SREdit.value()
        self.channel_num = self.channelNumEdit.value()
        self.epoch_length = self.epochLengthEdit.value()

        if self.data_path == '':
            # Alert warning box
            QMessageBox.about(self, "Error", "Please select a data file!")
            return
        elif self.label_path == '':
            QMessageBox.about(self, "Error", "Please select or set a label file!")
            return

        try:
            # Read data from data path
            self.data = list(loadmat(self.data_path).values())[-1]
            if self.data.shape[0] > 20:
                self.data = self.data.transpose()
            self.data_length = self.data.shape[1]

            self.total_seconds = ceil(self.data_length / self.SR)

            # Do not accept the data is less than 30 s
            if self.total_seconds < 30:
                QMessageBox.about(self, "Error", "Do not accept data less than 30 seconds!")
                # self.dataPathEdit.clear()
                # self.data_path = ''
                return
            if len(self.data) != self.channel_num:
                QMessageBox.about(self, "Error",
                                  "Number of channels in data file "
                                  + str(self.data.shape[0]) + " does not match the entered channel number "
                                  + str(self.channel_num) + "!")
                # self.dataPathEdit.clear()
                # self.data_path = ''
                return

            self.channel_num = self.data.shape[0]

            # load label from label path
            f = open(self.label_path, 'r+')
            self.label_file = [each.replace("\n", "") for each in f.readlines()]

            if len(self.label_file) == 0:

                self.acquisition_time = self.dateTimeEdit.dateTime().toPyDateTime()
                # Initialize the labels
                self.label_file.append("READ ONLY! DO NOT EDIT!\n4-INIT 3-Wake 2-REM 1-NREM")
                self.label_file.append("\nSave time: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                self.label_file.append("\nAcquisition time: " +
                                       self.acquisition_time.strftime("%Y-%m-%d %H:%M:%S"))
                self.label_file.append("\nSampling rate: " + str(self.SR))
                self.label_file.append("\n==========Marker==========")
                self.label_file.append("\n==========Start-End==========")
                self.label_file.append("\n==========Sleep stage==========\n")
                label_start_time = 0
                label_end_time = self.total_seconds - 1

                # Three types label format 2023-09-11
                # Marker labels:
                # [[timestamp, time_sec, 1, timestamp, time_sec, 0, label_type, label_name], ..]
                # Start end labels:
                # [[start_timestamp, start_sec, 1, end_timestamp, end_sec, 0, label_type, label_name], ..]
                # Sleep stage labels:
                # [[start_timestamp, start_sec, 1, end_timestamp, end_sec, 0, label_type, label_name], ..]
                # The '1' and '0' above are stand for 'start' and 'end' respectively

                sleep_stage = [(label_start_time, label_end_time, '4', 'INIT')]
                sleep_stage_format = "\n".join(
                    [", ".join([second2time(second=each[0], ac_time=self.acquisition_time), str(each[0]), '1',
                                second2time(second=each[1], ac_time=self.acquisition_time), str(each[1]), '0', each[2],
                                each[3]]) for each in sleep_stage])
                self.label_file += sleep_stage_format

                f.write("".join(self.label_file))
                f.close()

                self.labels = [[], [], sleep_stage]

            else:
                # check label valid
                # 1. sleep start time
                # 2. sleep end time

                SR_ = int(self.label_file[self.label_file.index("==========Marker==========") - 1].split(": ")[1])
                label_start_time = self.label_file[
                    self.label_file.index("==========Sleep stage==========") + 1].split(", ")[1]
                label_end_time = self.label_file[-1].split(", ")[4]
                if SR_ != self.SR:
                    QMessageBox.about(self, "Error",
                                      "Sampling rate " + self.SR +
                                      " does not match the one (" + str(SR_) + ") in label file")
                # -1 because the time start from 0 and total seconds is start from 1
                if self.total_seconds - 1 != (int(label_end_time) - int(label_start_time)):
                    QMessageBox.about(self, "Error",
                                      "Invalid label file, please check it or create a new one.")
                    # self.labelPathEdit.clear()
                    # self.label_path = ''
                    return

                self.acquisition_time = datetime.datetime.strptime(self.label_file[3].split(": ")[1],
                                                                   "%Y-%m-%d %H:%M:%S")
                self.dateTimeEdit.setDateTime(self.acquisition_time)
            f = open(self.label_path, 'r+')
            self.label_file = [each.replace("\n", "") for each in f.readlines()]
            # If not empty or error, load the three types of labels
            mark_label_idx = self.label_file.index("==========Marker==========")
            start_end_label_idx = self.label_file.index("==========Start-End==========")
            sleep_stage_label_idx = self.label_file.index("==========Sleep stage==========")
            self.labels = [self.label_file[mark_label_idx + 1: start_end_label_idx],
                           self.label_file[start_end_label_idx + 1: sleep_stage_label_idx],
                           self.label_file[sleep_stage_label_idx + 1:]]

        except Exception as e:
            QMessageBox.about(self, "Error", "Invalid data file, please check the data format!")
            # self.dataPathEdit.clear()
            return

        win_plot.__init__(data=self.data, labels=self.labels, label_file=self.label_path,
                          SR=self.SR, epoch_length=self.epoch_length, acquisition_time=self.acquisition_time)

        del self.data
        print("Check finish!")
        win_plot.my_sleep()
        win_plot.show()

    def get_label_for_transfer(self):
        """
        Get label file for transfer
        :return:
        """

        self.label_path, _ = QFileDialog.getOpenFileName(self, 'Select label file',
                                                         r'E:/', 'txt Files (*.txt *.TXT)')
        self.labelFileEditor.setText(self.label_path)

    def get_data_for_auto_staging(self):
        """
        Open fileDialog for data file selection
        :return:
        :rtype:
        """

        self.auto_stage_data_path, _ = QFileDialog.getOpenFileName(self, 'Select data file',
                                                                   r'E:/', 'matlab Files (*.mat *.MAT)')
        self.dataFileEditor.setText(self.auto_stage_data_path)

        if self.dataFileEditor.text() != '':
            data = list(loadmat(self.dataFileEditor.text()).values())[-1]
            if data.shape[0] > 20:
                data = data.transpose()

            self.AS_data = data

            qList = [str(each) for each in range(1, len(data) + 1)]
            channel_slm = QStringListModel()
            # Set up model
            channel_slm.setStringList(qList)
            self.channelListView.setModel(channel_slm)

    def auto_staging(self):
        """
        Auto staging function, use auto stage class
        :return:
        :rtype:
        """

        if self.dataFileEditor.text() == '':
            # Alert warning box
            QMessageBox.about(self, "Error", "Please select a data file!")
            return

        selected_channels = [each.row() for each in self.channelListView.selectedIndexes()]

        # May need to sort selected channels
        if len(selected_channels) != 2:
            # Alert warning box
            QMessageBox.about(self, "Error", "We recommend 1 EEG and 1 EMG for prediction")
            return
        selected_data = [self.AS_data[i] for i in selected_channels]
        del self.AS_data
        selected_model_name = self.model_lst[self.modelSelectorCombo.currentIndex()]
        selected_model = joblib.load(f'../models/{selected_model_name}.pkl')
        # Get model type according to the model name
        model_type = selected_model_name.split('_')[0]

        epoch_length = self.ASEpochLengthEditor.value()
        SR = self.ASSREditor.value()

        auto_stage = Auto_stage(data=selected_data, model=selected_model, model_type=model_type,
                                epoch_length=epoch_length, SR=SR)

        predicted_label = auto_stage.predicted_label
        del auto_stage
        # Expand predicted labels to per second
        predicted_label = [each for each in predicted_label for _ in range(epoch_length)]
        predicted_label = [[idx, value] for idx, value in enumerate(predicted_label)]
        acquisition_time = self.ASDateTimeEditor.dateTime().toPyDateTime()

        sleep_stage_labels = lst2group(predicted_label)
        # Padding the end with INIT stage
        if sleep_stage_labels[-1][1] != ceil(len(selected_data[0]) / SR)-1:
            sleep_stage_labels.append([sleep_stage_labels[-1][1]+1, ceil(len(selected_data[0]) / SR)-1, 4])
        sleep_stage_labels = [', '.join([second2time(each[0], ac_time=acquisition_time), str(each[0]), '1',
                                         second2time(each[1], ac_time=acquisition_time), str(each[1]),
                                         '0', str(each[2]), self.stage_type_dict[each[2]]])
                              for each in sleep_stage_labels]

        labels = ["READ ONLY! DO NOT EDIT!\n4-INIT 3-Wake 2-REM 1-NREM",
                  "Save time: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Acquisition time: " +
                  acquisition_time.strftime("%Y-%m-%d %H:%M:%S"), "Sampling rate: " + str(SR),
                  "==========Marker==========" + '\n'.join([]),
                  "==========Start-End==========" + '\n'.join([]),
                  "==========Sleep stage==========", '\n'.join(sleep_stage_labels)]

        save_path, _ = QFileDialog.getOpenFileName(self, 'Save predicted labels',
                                                   f'{self.auto_stage_data_path.split("/")[0]}',
                                                   'txt Files (*.txt *.TXT)')
        if save_path == "":
            return

        with open(save_path, 'w') as f:
            f.write('\n'.join(labels))

    def transfer(self):
        """
        Transfer bar, use transfer class
        :return:
        """

        if self.labelFileEditor.text() == '':
            # Alert warning box
            QMessageBox.about(self, "Error", "Please select a label file!")
            return
        label_file_path = self.labelFileEditor.text()
        transfer = Transfer(label_file_path=label_file_path)
        try:
            transfer.get_params()
            fd, type_ = QFileDialog.getSaveFileName(self, "Save results",
                                                    'E:/transfer_results', "*.xlsx;;")
            if fd == '':
                del transfer
                return

            transfer.save_transfer_results(fd)

            del transfer
        except Exception as e:
            print(e)
            QMessageBox.about(self, "Error", "Invalid label file, please ensure the label file was create by MiSleep!")
            del transfer
            return

    def hypnogram(self):
        """
        Draw hypnogram and save
        :return:
        """

        if self.labelFileEditor.text() == '':
            # Alert warning box
            QMessageBox.about(self, "Error", "Please select a label file!")
            return

        color_map = {0: 'blue', 1: 'red', 2: 'purple', 3: 'orange', 4: 'green', 5: 'yellow'}
        # Get params from GUI
        line_color = color_map[self.colorSelector.currentIndex()]
        start_sec = self.startSecEditor.value()
        end_sec = self.endSecEditor.value()

        if end_sec != 99 and end_sec - start_sec < 100:
            # Alert warning box
            QMessageBox.about(self, "Warning", "Hypnogram time duration should be 100 seconds at least!")
            return
        title = self.titleEditor.text()
        label_file_path = self.labelFileEditor.text()

        hypnogram = Hypnogram(label_file_path=label_file_path, line_color=line_color, start_sec=start_sec,
                              end_sec=end_sec, title=title)
        try:
            hypnogram.get_params()
            fd, type_ = QFileDialog.getSaveFileName(self, "Save hypnogram",
                                                    f'E:/{title}', "*.png;;*.pdf;;*.eps;;")
            if fd == '':
                del hypnogram
                return

            hypnogram.draw_save_hypnogram(save_path=fd)
            del hypnogram
        except Exception as e:
            print(e)
            QMessageBox.about(self, "Error", "Invalid label file, please ensure the label file was create by MiSleep!")
            del hypnogram
            return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = load_gui()
    win_plot = sleep()
    myWin.show()
    sys.exit(app.exec_())

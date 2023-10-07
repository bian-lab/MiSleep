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
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from hdf5storage import loadmat
from MiSleep.gui.load_data.load_data import Ui_Load_data
from MiSleep.plot.MiSleep import sleep
from MiSleep.utils.utils import second2time


class load_gui(QMainWindow, Ui_Load_data):
    def __init__(self, parent=None):
        super(load_gui, self).__init__(parent)
        self.setupUi(self)

        self.dateTimeEdit.setDateTime(datetime.datetime.now())

        self.data_path = ''
        self.label_path = ''
        self.SR = 256
        self.epoch_length = 5
        self.channel_num = 2
        self.acquisition_time = ''
        self.total_seconds = 0

        self.dataSelectBt.clicked.connect(self.get_data_path)
        self.labelSelectBt.clicked.connect(self.get_label_path)

        # Press check button and check data
        self.checkBt.clicked.connect(self.check)

        self.data = []
        self.label_file = []  # Labels for saving format
        self.labels = []  # Labels format, including three part, marker label, start end label, sleep stage label
        self.data_length = 0

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
        self.acquisition_time = self.dateTimeEdit.dateTime()
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
                # Initialize the labels
                self.label_file.append("READ ONLY! DO NOT EDIT!\n4-INIT 3-Wake 2-REM 1-NREM")
                self.label_file.append("\nSave time: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                self.label_file.append("\nAcquisition time: " +
                                       self.acquisition_time.toPyDateTime().strftime("%Y-%m-%d %H:%M:%S"))
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

                acquisition_time = datetime.datetime.strptime(self.label_file[3].split(": ")[1], "%Y-%m-%d %H:%M:%S")
                self.dateTimeEdit.setDateTime(acquisition_time)
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = load_gui()
    win_plot = sleep()
    myWin.show()
    sys.exit(app.exec_())

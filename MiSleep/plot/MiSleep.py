# -*- coding: UTF-8 -*-
"""
@Project: EEGProcessing_V3 
@File: MiSleep.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2023/9/6 11:28 
@Description:  
"""
import copy
import datetime
from math import ceil

import numpy as np
import scipy
from PyQt5 import QtCore
from PyQt5.QtCore import QStringListModel, Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QMainWindow, QDialog, QFileDialog, QMessageBox, QShortcut
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy import signal
from scipy.signal import butter, welch

from MiSleep.gui.MiSleep.labels import Ui_label
from MiSleep.gui.MiSleep.sleep import Ui_sleep
from MiSleep.gui.MiSleep.spectrum import Ui_spectrum
from MiSleep.utils.utils import second2time, lst2group, get_4_stages


class sleep(QMainWindow, Ui_sleep):
    """
    Sleep class, including plots and analysis
    """

    def __init__(self, parent=None, data=None, labels=None, label_file=None,
                 SR=256, epoch_length=5, acquisition_time=None):
        """
        :param parent:
        :param data: All channel data, depends on the numbers of its column
        :param labels: Label data, including Marker label, start-end label and sleep stage label
                        labels data format: [Marker label, start-end label, sleep stage label]
                        Marker label data format:
                        ['timestamp, time_sec, timestamp, time_sec, label_type, label_name', ...]
                        start-end label data format:
                        ['start_timestamp, start_sec, end_timestamp, end_sec, label_type, label_name', ...]
                        sleep stage label data format:
                        ['start_timestamp, start_sec, end_timestamp, end_sec, label_type, label_name', ...]
        :param label_file: The file path of label data, can get other information from it,
                            and save it to the path after labeling
        :param SR: Sampling rate
        :param epoch_length:
        :param acquisition_time:
        """

        # The first time load this class, there is no data pass in, check it and pass if the data is None
        if data is None:
            return

        # Initialize window and setup widgets
        super(sleep, self).__init__(parent=parent)
        self.setupUi(self)

        # Receive the data passed in
        self.data = data
        self.label_file = label_file
        # Three types of label, convert labels to manipulation format, mainly the time format
        # Change the sleep_stage_labels to sec-label format, saving the time for labeling
        # After converting, the labels' format is:
        # marker_labels:
        # [[timestamp, time_sec, 1, timestamp, time_sec, 0, label_type, label_name], ..]
        # --> [[time_sec, label_name], ...]
        # start_end_labels:
        # [[start_timestamp, start_sec, 1, end_timestamp, end_sec, 0, label_type, label_name],..]
        # --> [[start_sec, end_sec, label_name], ...]
        # sleep_stage_labels:
        # [[start_timestamp, start_sec, 1, end_timestamp, end_sec, 0, label_type, label_name], ..]
        # --> [[start_sec, end_sec, label_type]]
        self.marker_labels = [[int(each.split(', ')[1]), each.split(', ')[7]] for each in labels[0]]
        self.start_end_labels = [[int(each.split(', ')[1]), int(each.split(', ')[4]), each.split(', ')[7]]
                                 for each in labels[1]]
        temp_sleep_stage_labels = [[int(each.split(', ')[1]), int(each.split(', ')[4]), int(each.split(', ')[6])]
                                   for each in labels[2]]
        self.sleep_stage_labels = []
        for each in temp_sleep_stage_labels:
            self.sleep_stage_labels += [[sec, each[2]] for sec in range(each[0], each[1] + 1)]

        self.SR = SR
        self.epoch_length = epoch_length
        self.stage_type_dict = {1: 'NREM', 2: 'REM', 3: 'Wake', 4: 'INIT'}
        self.acquisition_time = acquisition_time.toPyDateTime()

        # Initialize some widgets' initial value and setup
        self.epochCustomRadio.setChecked(False)
        self.epochShow.setDisabled(True)
        self.total_seconds = ceil(self.data.shape[1] / self.SR)
        self.epochShow.setMaximum(int(self.total_seconds / self.epoch_length))
        self.epoch_selector_dict = {0: 30, 1: 60, 2: 300, 3: 1800, 4: 3600, 5: self.epoch_length,
                                    6: 3 * self.epoch_length, 7: 5 * self.epoch_length, 8: 9 * self.epoch_length}
        self.autoScrollCheckBox.setChecked(False)  # Select auto scroll
        self.markerRadio.setChecked(True)  # Select mark radio

        # Initialize other attributes
        self.is_saved = True  # Check whether the labels are saved into the label_file
        self.channel_num = self.data.shape[0]
        self.channel_list = [str(each + 1) for each in range(self.channel_num)]  # Name of each channel in a list
        # Channels index to be show in the signal area, can select by user
        self.channel_show = list(range(self.channel_num))
        self.data_show = self.data  # Which channel data to show, connect to self.channel_show
        self.sample_num = self.data.shape[1]  # Number of samples
        self.samples_x = range(self.sample_num)  # For building figure x-axis

        # About epoch and current window attributes
        self.position_sec = 0  # Current position of signal figures
        self.epoch_show = 5  # Number of epoch to be shown in current window
        # Current window size
        self.x_window_size_sec = self.epoch_selector_dict[self.epochSelector.currentIndex()]
        self.x_window_size = int(self.x_window_size_sec * self.SR)  # Current window size of sampling points
        self.y_lims = [max(self.data[each][:30]) for each in range(self.channel_num)]  # y-axis lim of each channel data

        # Initialize signalArea figure
        self.signal_figure = plt.figure()
        # +1 is for time-frequency MiSleep
        self.signal_ax = self.signal_figure.subplots(nrows=self.channel_num + 1, ncols=1)
        self.signal_figure.set_tight_layout(True)
        self.signal_figure.tight_layout(pad=0, w_pad=0, h_pad=0)
        self.signal_figure.subplots_adjust(hspace=0, left=0.03, right=0.97, top=0.97, bottom=0.07)  # Adjust subplots
        # Put the figure to a canvas
        self.signal_canvas = FigureCanvas(self.signal_figure)
        # Add button click release event for signal canvas
        self.signal_canvas.mpl_connect("button_release_event", self.click_signal)
        # Put signal canvas to signalScrollArea
        self.signalScrollArea.setWidget(self.signal_canvas)
        # Add axvline to signal figure for visualizing epoch
        self.signal_axvline = None

        # Initialize sleep stage area figure
        self.sleep_stage_figure = plt.figure(layout='constrained')
        self.sleep_ax = self.sleep_stage_figure.subplots(nrows=1, ncols=1)
        self.sleep_stage_canvas = FigureCanvas(self.sleep_stage_figure)
        self.sleepStageArea.setWidget(self.sleep_stage_canvas)
        # Click sleep stage area, jump to that time
        self.sleep_stage_canvas.mpl_connect('button_press_event', self.press_sleep_stage)
        # Add axvline for current time on sleep stage figure
        self.sleep_axvline = self.sleep_ax.axvline(self.position_sec, color='gray', alpha=0.8)

        # Start-end label, if there is two values in the list, then store them to start_end_labels list
        self.start_end = []
        self.start_end_vline = None
        # Initialize labels to display in the label dialog for selecting
        self.marker_label_list = ['label' + str(each) for each in range(1, 21)]
        self.start_end_label_list = ['Slow Wave', 'Spindle'] + ['start-end label' + str(each) for each in range(1, 19)]
        # Initialize label dialog
        self.label_dialog = label_dialog(marker_label=self.marker_label_list, start_end_label=self.start_end_label_list)
        self.spectrum_dialog = spectrum_dialog()

        # Set default channel for time frequency analysis
        self.default_TF_channel = 0

        # Label current window with different sleep stage, only with shortcut
        self.nremSc = QShortcut(QKeySequence('Shift+1'), self)
        self.nremSc.activated.connect(self.NREM_window)
        self.remSc = QShortcut(QKeySequence('Shift+2'), self)
        self.remSc.activated.connect(self.REM_window)
        self.wakeSc = QShortcut(QKeySequence('Shift+3'), self)
        self.wakeSc.activated.connect(self.Wake_window)
        self.initSc = QShortcut(QKeySequence('Shift+4'), self)
        self.initSc.activated.connect(self.INIT_window)

        # Set a timer
        self.save_timer = QTimer(self)
        self.save_timer.timeout.connect(self.auto_save)

        # Set slm for channel list
        self.channel_slm = QStringListModel()

        # Percentile for spectrum colorbar
        self.spectrum_percentile = 99.7

    def my_sleep(self):
        """
        Mainly for setup widgets
        :return:
        """

        # Time slider time range
        self.timeSlider.setRange(0, self.total_seconds)
        # Time slider click and scroll event
        self.timeSlider.valueChanged.connect(self.slider_change)
        self.timeSlider.setSingleStep(self.epoch_length)
        self.timeSlider.setPageStep(self.x_window_size_sec)

        # Two time editor area, seconds and time stamp
        self.secTimeEdit.setMaximum(self.total_seconds - 1)
        self.dateTimeEdit.setMaximumDateTime(self.acquisition_time + datetime.timedelta(
            seconds=self.total_seconds - 1))
        # Time editor time to go
        self.dateTimeEdit.dateTimeChanged.connect(self.data_time_go)
        self.secTimeEdit.valueChanged.connect(self.sec_time_go)

        # spectrum percentile changes
        self.spectrumPercentileEdit.valueChanged.connect(self.reset_spectrum_percentile)

        # Next and previous button
        # self.previousWinBt.clicked.connect(self.update_previous_position)
        # self.nextWinBt.clicked.connect(self.update_next_position)

        # Scaler buttons, operate the selected channel
        self.reductionBt.clicked.connect(self.reduction_y_lim)
        self.amplifyBt.clicked.connect(self.amplify_y_lim)

        # Add channel to channel list container
        self.add_channel()

        # Epoch selector or custom
        self.epochShow.valueChanged.connect(self.set_x_window_sec)
        self.epochSelector.currentIndexChanged.connect(self.set_x_window_sec)
        self.epochCustomRadio.clicked.connect(self.check_epoch_custom)

        # Channel operation
        self.channelShowBt.clicked.connect(self.show_channel)
        self.channelHideBt.clicked.connect(self.hide_channel)
        self.channelDeleteBt.clicked.connect(self.delete_channel)

        # Filter function
        self.hz_edit_ability()
        self.passSelecter.currentIndexChanged.connect(self.hz_edit_ability)
        self.filterBt.clicked.connect(self.filter)

        # Spectrum
        self.spectrumBt.clicked.connect(self.draw_fft_freq)

        # Select default channel for time_frequency
        self.defaultTFBt.clicked.connect(self.set_default_TF_channel)

        # Function for labelBt
        self.labelBt.clicked.connect(self.label_start_end)

        # Label selected start end area with different sleep stage, with three buttons or shortcut
        self.remBt.clicked.connect(self.REM_start_end)
        self.nremBt.clicked.connect(self.NREM_start_end)
        self.wakeBt.clicked.connect(self.Wake_start_end)
        self.initBt.clicked.connect(self.Init_start_end)

        # Save labels
        self.saveBt.clicked.connect(self.save)

        # Draw figure
        self.window_plot()
        self.update_sleep_stage()

        # Set a timer observe the save event, every 5 mins auto save
        self.save_timer.start(60 * 5 * 1000)

        # Save selected channels' data
        self.saveDataBt.clicked.connect(self.save_selected_data)
        self.saveSleepStageDataBt.clicked.connect(self.merge_selected_data)
        self.saveAllBt.clicked.connect(self.save_all_data)
        self.mergeAllBt.clicked.connect(self.merge_all_data)

        # Listen to label dialog list changes, if data changed, call update_label_list function to update
        self.label_dialog.slm.dataChanged.connect(self.update_label_list)

        # If channel list name is edited, update the channel list
        self.channel_slm.dataChanged.connect(self.update_channel_names)

    def window_plot(self, redraw_spectrum=False):
        """
        Main function, MiSleep and update all the figures
        :param redraw_spectrum: Adjust colorbarPercentile, redraw spectrum ax
        :return:
        """

        # Update sleep figure's avvline
        self.sleep_axvline.remove()
        self.sleep_axvline = self.sleep_ax.axvline(self.position_sec, color='gray', alpha=0.8)
        self.sleep_stage_figure.canvas.draw()
        self.sleep_stage_figure.canvas.flush_events()

        position = int(self.position_sec * self.SR)  # Convert to point

        if position + self.x_window_size > self.sample_num:
            # Jump to end
            position = self.sample_num - self.x_window_size
        elif position < 0:
            position = 0

        self.position_sec = int(position / self.SR)

        x = list(self.samples_x[position: position + self.x_window_size])  # Construct x-axis index

        # get sleep stage
        sleep_labels = lst2group(self.sleep_stage_labels)

        # Plot time-frequency figure at signal_ax[0]
        self.signal_ax[0].clear()
        # F is frequency resolution, T is time resolution, and Sxx is power matrix,
        # the three variables' format show as below:
        # F: from 0 to 152.5 (Half of sampling rate), contains 129 data points
        # T: time resolution, depends on window size
        # Sxx: power density, shape of (F.shape, T.shape)

        # Filter the data to reduce data point
        fnorm = np.array(30 / (.5 * self.SR))
        b, a = butter(3, fnorm, btype='lowpass')

        filtered_data = signal.filtfilt(b, a,
                                        self.data[self.default_TF_channel][position: position + self.x_window_size])
        F, T, Sxx = signal.spectrogram(filtered_data, fs=self.SR, noverlap=0, nperseg=self.SR)
        # Sxx = numpy.log(Sxx)
        cmap = plt.cm.get_cmap('jet')

        self.signal_ax[0].set_ylim(0, 30)
        # cmap = plt.cm.S
        self.signal_ax[0].pcolormesh(T, F, Sxx, cmap=cmap, vmax=np.percentile(Sxx, self.spectrum_percentile))
        self.signal_ax[0].set_xticks([])

        if redraw_spectrum:
            # flush and return
            self.signal_figure.canvas.draw()  # Redraw canvas
            self.signal_figure.canvas.flush_events()  # Flush canvas
            return

        # If marker label or start-end label is in current window, show them
        show_labels_mark = [each for each in self.marker_labels if int(each[0] * self.SR) in x]
        show_labels_start_end = []
        for each in self.start_end_labels:
            if int(each[0] * self.SR) in x:
                show_labels_start_end.append([each[0], 's-' + each[2]])
            if int(each[1] * self.SR) in x:
                show_labels_start_end.append([each[1], 'e-' + each[2]])

        # Traverse all channels and MiSleep them
        for i in range(1, len(self.channel_show) + 1):
            self.signal_ax[i].clear()

            # Add axvline of each label in this window
            for vline_pos in show_labels_mark:
                self.signal_ax[i].axvline(vline_pos[0] * self.SR, color='red', alpha=1)
            for vline_pos in show_labels_start_end:
                self.signal_ax[i].axvline(vline_pos[0] * self.SR, color='dodgerblue', alpha=1)
            if self.start_end and self.start_end[0] * self.SR - 1 in x:
                self.signal_ax[i].axvline(self.start_end[0] * self.SR, color='lime', alpha=1)
            if len(self.start_end) == 2 and self.start_end[1] * self.SR in x:
                self.signal_ax[i].axvline(self.start_end[1] * self.SR, color='lime', alpha=1)
            y = self.data_show[i - 1][position: position + self.x_window_size]

            for each in sleep_labels:
                if each[1] * self.SR in x:
                    self.signal_ax[i].axvline((each[1] + 1) * self.SR, color='orange', alpha=0.3)

            self.signal_ax[i].set_ylim(ymin=-self.y_lims[self.channel_show[i - 1]],
                                       ymax=self.y_lims[self.channel_show[i - 1]])
            self.signal_ax[i].set_xlim(xmin=position, xmax=position + self.x_window_size)
            self.signal_ax[i].plot(x, y, color='black', linewidth=0.5)

            self.signal_ax[i].xaxis.set_ticks([])
            # self.signal_ax[i].yaxis.set_ticks([0], [self.channel_list[self.channel_show[i]]])
            # self.signal_ax[i].ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
            self.signal_ax[i].yaxis.set_ticks([0], ['{:.2e}'.format(self.y_lims[self.channel_show[i - 1]])],
                                              rotation=90)
            self.signal_ax[i].set_ylabel(self.channel_list[self.channel_show[i - 1]])

            # Add axvline of each epoch
            if len(x) < self.SR * 300:
                for axvline_positions in range(
                        self.position_sec, self.position_sec + self.x_window_size_sec, self.epoch_length):
                    self.signal_ax[i].axvline(axvline_positions * self.SR, color='green', alpha=0.1)

        # Add annotation of each axvline (except epoch line)
        for each in show_labels_mark:
            self.signal_ax[1].text(x=each[0] * self.SR,
                                   y=self.y_lims[self.channel_show[0]],
                                   s=each[1],
                                   verticalalignment="top",
                                   color='red')
        for each in show_labels_start_end:
            self.signal_ax[1].text(x=each[0] * self.SR,
                                   y=self.y_lims[self.channel_show[0]],
                                   s=each[1],
                                   verticalalignment="top",
                                   color='dodgerblue')
        if self.start_end and self.start_end[0] * self.SR in x:
            self.signal_ax[-1].text(x=self.start_end[0] * self.SR,
                                    y=-self.y_lims[self.channel_show[-1]],
                                    s="S", color='lime')
        if len(self.start_end) == 2 and self.start_end[1] * self.SR - 1 in x:
            self.signal_ax[-1].text(x=self.start_end[1] * self.SR,
                                    y=-self.y_lims[self.channel_show[-1]],
                                    s="E",
                                    horizontalalignment="right",
                                    color='lime')

        # Add annotation for sleep stage
        # Only add annotation but no axvline in the bottom of last figure
        for each in sleep_labels:
            if each[0] * self.SR in x:
                self.signal_ax[-1].text(
                    x=each[0] * self.SR,
                    y=-self.y_lims[self.channel_show[-1]],
                    s="s-" + self.stage_type_dict[each[2]],
                    color='orange')
            if each[1] * self.SR in x:
                self.signal_ax[-1].text(
                    x=(each[1] + 1) * self.SR,
                    y=-self.y_lims[self.channel_show[-1]],
                    s="e-" + self.stage_type_dict[each[2]],
                    horizontalalignment="right",
                    color='orange')

        # Set xtick for the last figure, because all the figures share the same x-axis
        if len(x) < self.SR * 450:

            self.signal_ax[-1].set_xticks(
                [each * self.SR for each in
                 range(self.position_sec, self.position_sec + self.x_window_size_sec + 1, self.epoch_length)],
                [each for each in
                 range(self.position_sec, self.position_sec + self.x_window_size_sec + 1, self.epoch_length)],
                rotation=45)
            self.signal_ax[-1].set_xticks(
                [each * self.SR for each in
                 range(self.position_sec, self.position_sec + self.x_window_size_sec + 1)],
                # [each for each in
                #  range(self.position_sec, self.position_sec + self.x_window_size_sec + 1)],
                minor=True)
            # self.signal_ax[-1].set_xticks(
            #     [each * self.SR for each in
            #      range(self.position_sec, self.position_sec + self.x_window_size_sec + 1, self.epoch_length)],
            #     [each for each in
            #      range(self.position_sec, self.position_sec + self.x_window_size_sec + 1, self.epoch_length)],
            #     rotation=45
            # )
        else:
            self.signal_ax[-1].set_xticks(
                [each * self.SR for each in
                 range(self.position_sec, self.position_sec + self.x_window_size_sec + 1, 50)],
                [each for each in
                 range(self.position_sec, self.position_sec + self.x_window_size_sec + 1, 50)],
                rotation=45
            )
        self.signal_figure.canvas.draw()  # Redraw canvas
        self.signal_figure.canvas.flush_events()  # Flush canvas

        # Update the value of secTimeEdit and dateTimeEdit
        self.secTimeEdit.setValue(self.position_sec)
        date_time = self.acquisition_time + datetime.timedelta(seconds=self.position_sec)
        self.dateTimeEdit.setDateTime(date_time)

    def update_sleep_stage(self):
        """
        Update sleep stage figure, in the bottom
        :return:
        """

        self.sleep_ax.clear()
        # Flush sleep stage axvlines
        self.sleep_axvline.remove()
        self.sleep_axvline = self.sleep_ax.axvline(self.position_sec, color='gray', alpha=0.8)

        # Show marker labels and start-end labels in sleep stage figure
        show_labels_mark = [each[0] for each in self.marker_labels]
        for vline_pos in show_labels_mark:
            self.sleep_ax.axvline(vline_pos, color='red', alpha=1)
        if self.start_end:
            self.sleep_ax.axvline(self.start_end[0], color='lime', alpha=1)
            if len(self.start_end) == 2:
                self.sleep_ax.axvline(self.start_end[1], color='lime', alpha=1)

        self.sleep_ax.step(range(self.total_seconds), [each[1] for each in self.sleep_stage_labels], where="mid",
                           linewidth=1)
        self.sleep_ax.set_ylim(0.5, 4.5)
        self.sleep_ax.set_xlim(0, self.total_seconds - 1)
        self.sleep_ax.xaxis.set_ticks([])
        self.sleep_ax.yaxis.set_ticks(range(1, 5), ['NREM', 'REM', 'Wake', 'INIT'])

        self.sleep_stage_figure.canvas.draw()
        self.sleep_stage_figure.canvas.flush_events()

    def slider_change(self):
        """
        Slider value changes, redraw
        :return:
        """

        self.position_sec = self.timeSlider.value()
        self.window_plot()

    def check_epoch_custom(self):
        """
        If select custom epoch radio, disable the selector
        :return:
        """

        if self.epochCustomRadio.isChecked():
            self.epochSelector.setDisabled(True)
            self.epochShow.setEnabled(True)

        else:
            self.epochSelector.setEnabled(True)
            self.epochShow.setDisabled(True)

    def set_x_window_sec(self):
        """
        Check is epochSelector or epoch custom, and pass value to x_window_size_sec
        :return:
        """
        if self.epochCustomRadio.isChecked():
            epoch_show = self.epochShow.value()
            # if the x_window_size_sec is larger than the total data points, then don't replot and reset the window size
            if epoch_show * self.epoch_length > self.total_seconds:
                self.x_window_size_sec = self.total_seconds
            else:
                self.epoch_show = epoch_show
                self.x_window_size_sec = epoch_show * self.epoch_length
        else:
            if self.epoch_selector_dict[self.epochSelector.currentIndex()] > self.total_seconds:
                self.x_window_size_sec = self.total_seconds
            else:
                self.x_window_size_sec = self.epoch_selector_dict[self.epochSelector.currentIndex()]
        self.x_window_size = int(self.x_window_size_sec * self.SR)
        self.window_plot()

        self.timeSlider.setPageStep(self.x_window_size_sec)

    def click_signal(self, event):
        """
        Click signalScrollArea and will trigger this function
        Three radios correspond three different type of marker:
        Marker: label the second clicked, popup a dialog to select a label,
                will show axvline and annotation both in signal MiSleep area and sleep stage MiSleep area
        Start-end: click to activate start axvline and annotation, and click again to activate end axvline
                    and annotation, then popup a dialog to select a label type
        Start-end for stage: similar to Start-end, but won't popup a dialog, can choose which sleep stage to
                                label by clicking button, or use shortcut

        :return:
        """

        # Get current second
        try:
            sec = int(event.xdata / self.SR)
        except TypeError as e:
            # When click the area not including in the MiSleep, will call this exception
            return

        # If clicked right button and there is a label, alert and delete this label
        if event.button == 3:
            # marker_secs = [each[0] for each in self.marker_labels]
            # if sec in marker_secs:
            #     self.marker_labels.remove(self.marker_labels[marker_secs.index(sec)])
            for idx, label in enumerate(self.marker_labels):
                if sec in label:
                    self.marker_labels.pop(idx)

            for idx, label in enumerate(self.start_end_labels):
                if sec in label:
                    self.start_end_labels.pop(idx)

            # Also remove the start end label
            if len(self.start_end) >= 1 and sec == self.start_end[0]:
                self.start_end = []
            elif len(self.start_end) == 2 and sec == self.start_end[1]:
                self.start_end.pop(1)

            self.update_sleep_stage()
            self.window_plot()
            return

        # If markerRadio selected
        elif self.markerRadio.isChecked():

            # Popup dialog for selection, cancel to quit
            self.show_dialog(type_=0)
            if self.label_dialog.closed:
                return
            label_name = self.label_dialog.label_name

            # If the second point already exist in marker_labels, remove it
            if (sec, label_name) in self.marker_labels:
                self.marker_labels.remove([sec, label_name])
            else:
                # Add current second to marker_labels
                self.marker_labels.append([sec, label_name])
                self.marker_labels = sorted(list(self.marker_labels), key=lambda x: x[0])
            self.is_saved = False

        # If startEndRadio selected
        elif self.startEndRadio.isChecked():
            if not self.start_end:
                self.start_end.append(sec)
            elif len(self.start_end) == 2:
                # Clear
                self.start_end = []
                self.start_end.append(sec)
            else:
                sec = ceil(event.xdata / self.SR)

                # If end second is less than start second, alert
                if sec <= self.start_end[0]:
                    QMessageBox.about(self, "Error", "End should be larger than Start!")
                    return

                self.start_end.append(sec)

        self.update_sleep_stage()
        self.window_plot()

    def show_dialog(self, type_=0):
        """
        Show dialog for label
        :return:
        """

        self.label_dialog.type_ = type_
        self.label_dialog.show_contents()
        self.label_dialog.exec()

    def draw_fft_freq(self):
        """
        Draw spectrum with FFT, show after click 'spectrum' button
        https://vimsky.com/examples/usage/python-scipy.signal.welch.html
        :return:
        """

        selected_channel = [each.row() for each in self.channelList.selectedIndexes()]
        if len(selected_channel) != 1:
            QMessageBox.about(self, "Info", "Please select only(at least) one channel!")
            return
        elif len(self.start_end) != 2:
            QMessageBox.about(self, "Info", "Please select a start end area!")
            return

        #  Construct data
        data = self.data[selected_channel[0]][
               int(self.start_end[0] * self.SR): int(self.start_end[1] * self.SR)]

        # Call spectrum dialog
        self.spectrum_dialog.data = data
        self.spectrum_dialog.start_end = self.start_end
        self.spectrum_dialog.SR = self.SR
        self.spectrum_dialog.epoch_length = self.epoch_length
        self.spectrum_dialog.spectrum_percentile = self.spectrum_percentile
        self.spectrum_dialog.draw()
        self.spectrum_dialog.activateWindow()
        self.spectrum_dialog.setWindowState(self.spectrum_dialog.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        self.spectrum_dialog.showNormal()

    def press_sleep_stage(self, event):
        """
        Press sleep stage figure and will jump to that time
        :return:
        """

        self.position_sec = int(event.xdata)
        self.window_plot()

    def reduction_y_lim(self):
        """
        Reduction y lim of selected channel
        :return:
        """

        selected_channel = [each.row() for each in self.channelList.selectedIndexes()]
        if len(selected_channel) == 0:
            QMessageBox.about(self, "Info", "Please select at least one channel to reduction!")
            return
        self.y_lims = [self.y_lims[each] * 1.1 if each in selected_channel else self.y_lims[each]
                       for each in range(self.channel_num)]
        self.window_plot()

    def amplify_y_lim(self):
        selected_channel = [each.row() for each in self.channelList.selectedIndexes()]
        if len(selected_channel) == 0:
            QMessageBox.about(self, "Info", "Please select at least one channel to amplify!")
            return
        self.y_lims = [self.y_lims[each] * 0.9 if each in selected_channel else self.y_lims[each]
                       for each in range(self.channel_num)]
        self.window_plot()

    def label_start_end(self):
        """
        Selected a start_end area, click label button will call this function, popup a dialog to select a label
        :return:
        """

        if len(self.start_end) != 2:
            QMessageBox.about(self, "Info", "Please select a start end area!")
            return
        # Popup dialog for selection, cancel to quit
        self.show_dialog(type_=1)
        if self.label_dialog.closed:
            return
        label_name = self.label_dialog.label_name

        start_sec = self.start_end[0]
        end_sec = self.start_end[1]

        # If the second point already exist in marker_labels, remove it
        if (start_sec, end_sec, label_name) in self.start_end_labels:
            pass
        else:
            # Add current second to marker_labels
            self.start_end_labels.append([start_sec, end_sec, label_name])
            self.start_end_labels = sorted(list(self.start_end_labels), key=lambda x: x[0])
        self.is_saved = False

        self.window_plot()

    def sec_time_go(self):
        """
        Jump to time specified, trigger by time change of secTimeEditor
        :return:
        """

        # Get second
        self.position_sec = self.secTimeEdit.value()
        # Redraw MiSleep
        self.timeSlider.setValue(self.position_sec)

    def data_time_go(self):
        """
        Jump to time specified, trigger by time change of dateTimeEditor, data format is DD:HH:MM:SS
        :return:
        """

        date_time = self.dateTimeEdit.dateTime().toPyDateTime()
        # Change time to second format and jump
        self.position_sec = int((date_time - self.acquisition_time).total_seconds())
        self.timeSlider.setValue(self.position_sec)

    def reset_spectrum_percentile(self):
        """
        Reset spectrum color percentile
        :return:
        """

        self.spectrum_percentile = self.spectrumPercentileEdit.value()
        self.window_plot(redraw_spectrum=True)

    def update_next_position(self):
        """
        Next window
        :return:
        """

        if self.position_sec + self.x_window_size_sec >= self.total_seconds:
            self.position_sec = self.total_seconds - self.x_window_size_sec
        else:
            self.position_sec = int(self.position_sec + self.x_window_size_sec)
        self.timeSlider.setValue(self.position_sec)

    def update_previous_position(self):
        """
        Previous window
        :return:
        """
        if self.position_sec < self.x_window_size_sec:
            self.position_sec = 0
        else:
            self.position_sec = int(self.position_sec - self.x_window_size_sec)
        self.timeSlider.setValue(self.position_sec)

    def add_channel(self):
        """
        Add and update channels to channel list container, update from self.channel_show
        :return:
        """

        qList = self.channel_list
        # Set up model
        self.channel_slm.setStringList(qList)
        self.channelList.setModel(self.channel_slm)

    def show_channel(self):
        """
        Select channel and click 'Show' button, show selected channels in th signal area
        :return:
        """

        selected_channel = [each.row() for each in self.channelList.selectedIndexes()]
        if len(selected_channel) == 0:
            return
        self.channel_show = sorted(selected_channel)

        self.channel_changed_show()

    def delete_channel(self):
        """
        Delete the channel selected
        :return:
        """

        box = QMessageBox.question(self, 'Warning', 'You are deleting data, please confirm!',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if box == QMessageBox.Yes:
            # Get selected channels
            selected_channel = [each.row() for each in self.channelList.selectedIndexes()]
            # If selected all channels, can't operate
            if len(selected_channel) == len(self.channel_list):
                # Alert
                QMessageBox.about(self, "Error", "You can't delete all channels!")

            # If selected channels is less than the number of all channels
            else:
                # selected_channel = sorted(selected_channel, reverse=True)
                for each in sorted(selected_channel, reverse=True):
                    if each in self.channel_show:
                        self.channel_show.remove(each)
                    self.channel_list.pop(each)
                    self.data = np.delete(self.data, each, axis=0)

                self.channel_show = list(range(len(self.channel_list)))
                self.data_show = self.data

                self.add_channel()
                self.channel_changed_show()

        else:
            pass

    def hide_channel(self):
        """
        Hide selected channel
        :return:
        """

        # Get channel selected to hide
        channels = [each.row() for each in self.channelList.selectedIndexes()]
        # Deepcopy, or it will change the original data
        channel_show_temp = list(copy.deepcopy(self.channel_show))
        # Remove selected channel from channel_show
        for each in channels:
            if each in channel_show_temp:
                channel_show_temp.remove(each)
            else:
                pass

        # Can't hide all channels
        if not channel_show_temp:
            # Alert
            QMessageBox.about(self, "Error", "You can't hide all channels!")
            channel_show_temp = self.channel_show

        self.channel_show = channel_show_temp

        self.channel_changed_show()

    def channel_changed_show(self):
        """
        If channel show changed, use this function to redraw plots
        :return:
        """

        self.data_show = [self.data[each] for each in self.channel_show]
        self.signal_figure.clf()
        self.signal_ax = self.signal_figure.subplots(nrows=len(self.channel_show) + 1, ncols=1)
        # if len(self.channel_show) == 1:
        #     self.signal_ax = [self.signal_ax]
        self.window_plot()

    def hz_edit_ability(self):
        """
        Select low pass or high pass, and disable the other editor
        :return:
        """

        if self.passSelecter.currentIndex() == 0:
            # Lowpass
            self.hzLowEdit.setDisabled(True)
            self.hzHighEdit.setEnabled(True)
        elif self.passSelecter.currentIndex() == 1:
            # HighPass
            self.hzHighEdit.setDisabled(True)
            self.hzLowEdit.setEnabled(True)
        else:
            self.hzLowEdit.setEnabled(True)
            self.hzHighEdit.setEnabled(True)

    def filter(self):
        """
        Click filter button and call this function to filter, add a new channel for filtered data
        :return:
        """

        # Operate only one channel at a time
        selected_channel = [each.row() for each in self.channelList.selectedIndexes()]
        low = self.hzLowEdit.value()
        high = self.hzHighEdit.value()
        if len(selected_channel) != 1:
            QMessageBox.about(self, "Error", "You can only(at least) select one channel to filter!")
        else:

            # Get data for filter
            wait_data = self.data[selected_channel[0]]

            # Lowpass
            if self.passSelecter.currentIndex() == 0:
                # filter
                fnorm = np.array(high / (.5 * self.SR))
                b, a = butter(3, fnorm, btype='lowpass')
                name = str(high) + '_LP'

            # HighPass
            elif self.passSelecter.currentIndex() == 1:
                # filter
                fnorm = np.array(low / (.5 * self.SR))
                b, a = butter(3, fnorm, btype='highpass')
                name = str(low) + '_HP'
            # BindPass
            else:
                if low >= high or low == 0:
                    QMessageBox.about(self, "Error", "Please reset value for filter")
                    return
                fnorm = np.divide([low, high], .5 * self.SR)
                b, a = butter(3, fnorm, btype='bandpass')
                name = str(low) + '~' + str(high) + '_BP'

            filtered_data = signal.filtfilt(b, a, wait_data)

            # Add a channel for filtered data
            self.channel_list.append(self.channel_list[selected_channel[0]] + '_' + name)
            self.channel_num += 1
            self.y_lims.append(max(filtered_data[:30]))
            self.channel_show.append(len(self.channel_list) - 1)

            # Merge the filtered data into the entire data list
            self.data = np.r_[self.data, np.array([filtered_data])]
            self.add_channel()

            self.channel_changed_show()

    def set_default_TF_channel(self):
        """
        Click defaultTFBt and call this function, set the default time frequency channel
        :return:
        """

        selected_channel = [each.row() for each in self.channelList.selectedIndexes()]
        if len(selected_channel) != 1:
            QMessageBox.about(self, "Error", "You can only(at least) select one channel for time frequency analysis!")
        else:
            self.default_TF_channel = selected_channel[0]
            self.window_plot()

    def REM_window(self):
        """
        Label current window as REM sleep stage, only call with shortcut CTRL+NUM1
        :return:
        """

        self.label_all(sleep_type=2)

    def NREM_window(self):
        """
        Label current window as NREM sleep stage, only call with shortcut CTRL+NUM2
        :return:
        """

        self.label_all(sleep_type=1)

    def Wake_window(self):
        """
        Label current window as wake sleep stage, only call with shortcut CTRL+NUM3
        :return:
        """

        self.label_all(sleep_type=3)

    def INIT_window(self):
        """
        Label current window as INIT state
        :return:
        """

        self.label_all(sleep_type=4)

    def REM_start_end(self):
        """
        Label selected start end area as REM sleep stage, call by button remBt or shortcut NUM1
        :return:
        """

        self.label_all(sleep_type=2, start_end=True)

    def NREM_start_end(self):
        """
        Label selected start end area as NREM sleep stage, call by button nremBt or shortcut NUM2
        :return:
        """

        self.label_all(sleep_type=1, start_end=True)

    def Wake_start_end(self):
        """
        Label selected start end area as Wake sleep stage, call by button wakeBt or shortcut NUM3
        :return:
        """

        self.label_all(sleep_type=3, start_end=True)

    def Init_start_end(self):
        """
        Label as INIT state, means not any kinds of label
        :return:
        """

        self.label_all(sleep_type=4, start_end=True)

    def label_all(self, sleep_type=3, start_end=False):
        """
        Label current window's sleep stage, call by REM_window, NREM_window, Wake_window
        :param start_end: Check whether is start_end type label or current window label
        :param sleep_type: sleep type to label
        :return:
        """

        if start_end:
            if len(self.start_end) != 2:
                QMessageBox.about(self, "Info", "Please select a start end area!")
                return
            self.sleep_stage_labels[self.start_end[0]: self.start_end[1]] = \
                [[each[0], sleep_type] for each in self.sleep_stage_labels[self.start_end[0]: self.start_end[1]]]
        else:
            self.position_sec = int(self.position_sec)
            self.sleep_stage_labels[self.position_sec: self.position_sec + self.x_window_size_sec] = \
                [[each[0], sleep_type] for each in
                 self.sleep_stage_labels[self.position_sec: self.position_sec + self.x_window_size_sec]]

        if self.autoScrollCheckBox.isChecked():
            if start_end:
                self.position_sec = self.start_end[1]
            else:
                self.position_sec = int(self.position_sec + self.x_window_size_sec)
            self.window_plot()

        self.is_saved = False
        self.window_plot()
        self.update_sleep_stage()

    def auto_save(self):
        """
        Auto save labels every 5 minutes if is_saved if False
        :return:
        """

        if not self.is_saved:
            self.save()

    def save(self):
        """
        Save labels into labels file, call by Save button or CTRL+S shortcut
        :return:
        """

        self.saveBt.setDisabled(True)
        # Preprocess labels in three label lists, convert to the format in the label file
        marker_labels = [
            ', '.join([second2time(each[0], ac_time=self.acquisition_time), str(each[0]), '1',
                       second2time(each[0], ac_time=self.acquisition_time), str(each[0]), '0', '1', each[1]])
            for each in self.marker_labels]
        start_end_labels = [', '.join([second2time(each[0], ac_time=self.acquisition_time), str(each[0]), '1',
                                       second2time(each[1], ac_time=self.acquisition_time), str(each[1]), '0',
                                       '1', each[2]]) for each in self.start_end_labels]

        # Sleep stage data is [[sec, label_type], ...], need to sort by label_type and construct a label list,
        # and convert to the format [[start_sec, end_sec, label_type]]
        sleep_stage_labels = lst2group(self.sleep_stage_labels)
        sleep_stage_labels = [', '.join([second2time(each[0], ac_time=self.acquisition_time), str(each[0]), '1',
                                         second2time(each[1], ac_time=self.acquisition_time), str(each[1]),
                                         '0', str(each[2]), self.stage_type_dict[each[2]]])
                              for each in sleep_stage_labels]

        if len(marker_labels) > 0:
            marker_labels = [''] + marker_labels
        if len(start_end_labels) > 0:
            start_end_labels = [''] + start_end_labels
        labels = ["READ ONLY! DO NOT EDIT!\n4-INIT 3-Wake 2-REM 1-NREM",
                  "Save time: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Acquisition time: " +
                  self.acquisition_time.strftime("%Y-%m-%d %H:%M:%S"), "Sampling rate: " + str(self.SR),
                  "==========Marker==========" + '\n'.join(marker_labels),
                  "==========Start-End==========" + '\n'.join(start_end_labels),
                  "==========Sleep stage==========", '\n'.join(sleep_stage_labels)]

        with open(self.label_file, 'w') as f:
            f.write('\n'.join(labels))
        self.is_saved = True
        self.saveBt.setEnabled(True)

    def closeEvent(self, event):
        """
        Check if labels are saved when close the window
        :param event:
        :return:
        """

        if not self.is_saved:
            box = QMessageBox.question(self, 'Warning',
                                       'Your labels haven\'t been saved, discard?\n'
                                       'Yes: Save and quit\nNo: Discard\nCancel: Back to sleep window',
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)

            if box == QMessageBox.Yes:
                self.save()

                event.accept()
            elif box == QMessageBox.No:

                event.accept()
            else:
                event.ignore()

    def save_all_data(self):
        """
        Save all data ignore whether selected channel(s), call by saveAllBt
        :return:
        """

        self.save_selected_data(save_all=True)

    def merge_all_data(self):
        """
        merge all data 
        :return:
        """

        self.merge_selected_data(merge_all=True)

    def save_selected_data(self, save_all=False):
        """
        Save selected data to matlab file format
        :return:
        """

        if save_all:
            save_data = self.data
        else:
            selected_channel = [each.row() for each in self.channelList.selectedIndexes()]
            if len(selected_channel) == 0:
                QMessageBox.about(self, "Error", "Please select at least 1 channel to save!")
                return

            save_data = [self.data[idx] for idx in selected_channel]

        fd, type_ = QFileDialog.getSaveFileName(self, "Save data", "E:/SelectedData", "*.mat;;*.MAT;;")
        if fd == '':
            return
        scipy.io.savemat(fd, mdict={'data': save_data})

    def merge_selected_data(self, merge_all=False):
        """
        Save selected data to different sleep stages
        :return:
        """

        self.setDisabled(True)
        if merge_all:
            save_data = self.data
        else:
            # Get selected channels' data
            selected_channel = [each.row() for each in self.channelList.selectedIndexes()]
            if len(selected_channel) == 0:
                QMessageBox.about(self, "Error", "Please select at least 1 channel to save!")
                self.setDisabled(False)
                return

            save_data = [self.data[idx] for idx in selected_channel]

        # Get each time duration's sleep labels
        nrem_data, rem_data, wake_data, init_data = get_4_stages(sleep_label_lst=self.sleep_stage_labels,
                                                                 data=save_data,
                                                                 SR=self.SR)
        # construct 3 stages' label to save
        nrem_labels = ', '.join([second2time(0, ac_time=self.acquisition_time), str(0), '1',
                                 second2time(ceil(len(nrem_data[0]) / self.SR), ac_time=self.acquisition_time),
                                 str(ceil(len(nrem_data[0]) / self.SR) - 1), '0', '1', 'NREM'])
        rem_labels = ', '.join([second2time(0, ac_time=self.acquisition_time), str(0), '1',
                                second2time(ceil(len(rem_data[0]) / self.SR), ac_time=self.acquisition_time),
                                str(ceil(len(rem_data[0]) / self.SR) - 1), '0', '2', 'REM'])
        wake_labels = ', '.join([second2time(0, ac_time=self.acquisition_time), str(0), '1',
                                 second2time(ceil(len(wake_data[0]) / self.SR), ac_time=self.acquisition_time),
                                 str(ceil(len(wake_data[0]) / self.SR) - 1), '0', '3', 'Wake'])
        init_labels = ', '.join([second2time(0, ac_time=self.acquisition_time), str(0), '1',
                                 second2time(ceil(len(init_data[0]) / self.SR), ac_time=self.acquisition_time),
                                 str(ceil(len(init_data[0]) / self.SR) - 1), '0', '4', 'INIT'])
        labels = ["READ ONLY! DO NOT EDIT!\n4-INIT 3-Wake 2-REM 1-NREM",
                  "Save time: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Acquisition time: " +
                  self.acquisition_time.strftime("%Y-%m-%d %H:%M:%S"), "Sampling rate: " + str(self.SR),
                  "==========Marker==========",
                  "==========Start-End==========",
                  "==========Sleep stage=========="]

        # Get file path
        path_ = QFileDialog.getExistingDirectory(self, "Select a folder to save 4 stages' data", "E:/")
        if path_ == '':
            self.setDisabled(False)
            return

        if nrem_data[0]:
            scipy.io.savemat(path_ + "/nrem_data.mat", mdict={'data': nrem_data})
            with open(path_ + "/nrem_labels.txt", 'w') as f:
                _ = copy.deepcopy(labels)
                _.append(nrem_labels)
                f.write('\n'.join(_))

        if rem_data[0]:
            scipy.io.savemat(path_ + "/rem_data.mat", mdict={'data': rem_data})
            with open(path_ + "/rem_labels.txt", 'w') as f:
                _ = copy.deepcopy(labels)
                _.append(rem_labels)
                f.write('\n'.join(_))

        if wake_data[0]:
            scipy.io.savemat(path_ + "/wake_data.mat", mdict={'data': wake_data})
            with open(path_ + "/wake_labels.txt", 'w') as f:
                _ = copy.deepcopy(labels)
                _.append(wake_labels)
                f.write('\n'.join(_))

        if init_data[0]:
            scipy.io.savemat(path_ + "/init_data.mat", mdict={'data': init_data})
            with open(path_ + "/init_labels.txt", 'w') as f:
                _ = copy.deepcopy(labels)
                _.append(init_labels)
                f.write('\n'.join(_))

        self.setEnabled(True)

    def update_label_list(self):
        """
        Update labels if label dialog changed the StringListModel
        :return:
        """

        # Get the changed StringList
        string_list = self.label_dialog.slm.stringList()

        # If it is marker labels, then update the marker list
        if self.markerRadio.isChecked():
            self.marker_label_list = string_list
            self.label_dialog.marker_label = string_list
        elif self.startEndRadio.isChecked():
            self.start_end_label_list = string_list
            self.label_dialog.start_end_label = string_list

    def update_channel_names(self):
        """
        If user edit the channel's name, update it in the list, and save in to the label file
        :return:
        """

        # Get the changed StringList
        string_list = self.channel_slm.stringList()
        self.channel_list = string_list
        self.is_saved = False
        self.window_plot()

    def hover_show(self, event):
        """
        When hover on the figure, show the y coordinate
        :return:
        """

    def draw_spectrum(self):
        """
        Plot spectrum and call spectrum dialog, also pass the data to save
        :return:
        """


class label_dialog(QDialog, Ui_label):
    def __init__(self, marker_label=None, start_end_label=None):
        """
        Pass in marker labels and start end labels to display for selection
        :param marker_label: a list contains several marker labels
        :param start_end_label: a list contains start end labels
        """
        super().__init__()
        self.setupUi(self)

        self.type_ = 0
        self.OkBt.clicked.connect(self.submit_label)
        self.cancelBt.clicked.connect(self.cancelEvent)

        self.slm = QStringListModel()

        # QListView can not select the setting default line when first time initialize this Dialog
        self.slm.setStringList(['1'])
        self.listView.setModel(self.slm)
        idx = self.slm.index(0)
        self.listView.setCurrentIndex(idx)
        self.marker_label = marker_label
        self.start_end_label = start_end_label
        self.label_name = ''
        self.closed = False

    def show_contents(self):
        """
        Add the labels into the label list area in the label dialog
        :return:
        """

        self.closed = False
        idx = self.slm.index(0)
        if self.type_ == 0:
            # Marker type

            # Set model list, load data
            self.slm.setStringList(self.marker_label)
            self.listView.setModel(self.slm)
            # Select the first item by default
            self.listView.setCurrentIndex(idx)

        else:
            # start-end type
            self.slm.setStringList(self.start_end_label)
            self.listView.setModel(self.slm)
            # Select the first one by default
            self.listView.setCurrentIndex(idx)

    def submit_label(self):
        """
        Click the 'OK' button, return the selected label's index
        :param:
        :return:
        """

        if self.type_ == 0:
            self.label_name = self.marker_label[self.listView.selectedIndexes()[0].row()]
        else:
            self.label_name = self.start_end_label[self.listView.selectedIndexes()[0].row()]

        # Hide the dialog window but not close it
        self.hide()

    def cancelEvent(self):
        """
        Click the 'Cancel' button will call this function
        :return:
        """

        self.closed = True
        self.hide()

    def closeEvent(self, event):
        """
        Rewrite the close function of QDialog, aim to hide but not close the dialog
        :param event: QDialog event
        :return:
        """
        event.ignore()
        self.closed = True
        self.hide()


class spectrum_dialog(QDialog, Ui_spectrum):

    def __init__(self):
        """
        Pass in the x data and y data to MiSleep the spectrum
        """

        super().__init__()
        self.setupUi(self)

        self.data = None
        self.start_end = []
        self.SR = 256
        self.epoch_length = 5
        self.spectrum_percentile = 99.7

        self.spectrum_figure = plt.figure()
        self.spectrum_ax = self.spectrum_figure.subplots(nrows=1, ncols=1)
        self.spectrum_canvas = FigureCanvas(self.spectrum_figure)
        self.spectrumArea.setWidget(self.spectrum_canvas)
        self.saveSpectrumBt.clicked.connect(self.save_spectrum)

        self.tf_figure = plt.figure()
        self.tf_ax = self.spectrum_figure.subplots(nrows=1, ncols=1)
        self.tf_canvas = FigureCanvas(self.tf_figure)
        self.timeFrequencyArea.setWidget(self.tf_canvas)
        self.saveTFBt.clicked.connect(self.save_TF)

        # value
        self.spectrum_F = None
        self.spectrum_P = None
        self.time_frequency_T = None
        self.time_frequency_F = None
        self.time_frequency_P = None

        # Add maximum and minimum for spectrum dialog
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint)

    def draw(self):
        """
        Plot the spectrum and put it in the dialog area
        :return:
        """

        self.refresh_canvas()
        self.setWindowTitle('Spectrum: ' + str(self.start_end[0]) + '~' + str(self.start_end[1]))

        # Get Spectrum
        fnorm = np.array(50 / (.5 * self.SR))
        b, a = butter(3, fnorm, btype='lowpass')

        filtered_data = signal.filtfilt(b, a, self.data)
        self.spectrum_F, self.spectrum_P = welch(filtered_data, self.SR, nperseg=self.epoch_length * self.SR)

        # Get time frequency
        self.time_frequency_F, self.time_frequency_T, self.time_frequency_P = \
            signal.spectrogram(self.data, fs=self.SR, noverlap=0, nperseg=self.SR)
        cmap = plt.cm.get_cmap('jet')

        # plot time frequency
        pcm = self.tf_ax.pcolormesh(self.time_frequency_T, self.time_frequency_F[:31], self.time_frequency_P[:31],
                                    cmap=cmap, vmax=np.percentile(self.time_frequency_P,
                                                                  self.spectrum_percentile))
        self.tf_figure.colorbar(pcm, ax=self.tf_ax)
        self.tf_ax.set_ylim(0, 30)
        self.tf_ax.set_xticks([each for each in range(0, self.start_end[1] - self.start_end[0],
                                                      int((self.start_end[1] - self.start_end[0]) / 5))],
                              [each + self.start_end[0] for each in range(0, self.start_end[1] - self.start_end[0],
                                                                          int((self.start_end[1] - self.start_end[
                                                                              0]) / 5))])
        self.tf_ax.set_xlabel("Time (S)")
        self.tf_ax.set_ylabel("Frequency (HZ)")

        # MiSleep spectrum
        # self.spectrum_ax[1].clear()
        major_ticks_top = np.linspace(0, 50, 26)
        minor_ticks_top = np.linspace(0, 50, 51)

        self.spectrum_ax.xaxis.set_ticks(major_ticks_top)
        self.spectrum_ax.xaxis.set_ticks(minor_ticks_top, minor=True)
        self.spectrum_ax.grid(which="major", alpha=0.6)
        self.spectrum_ax.grid(which="minor", alpha=0.3)

        self.spectrum_ax.set_xlim(0, 50)
        self.spectrum_ax.plot(self.spectrum_F, self.spectrum_P)
        self.spectrum_ax.set_xlabel("Frequency (Hz)")
        self.spectrum_ax.set_ylabel("Power spectral density (Power/Hz)")

        self.spectrum_figure.canvas.draw()
        self.spectrum_figure.canvas.flush_events()
        self.tf_figure.canvas.draw()
        self.tf_figure.canvas.flush_events()

    def refresh_canvas(self):
        """
        Color bar cannot be deleted, refresh the canvas every time
        :return:
        """

        self.spectrum_figure = plt.figure()
        self.spectrum_figure.set_tight_layout(True)
        self.spectrum_ax = self.spectrum_figure.subplots(nrows=1, ncols=1)
        self.spectrum_canvas = FigureCanvas(self.spectrum_figure)
        self.spectrumArea.setWidget(self.spectrum_canvas)

        self.tf_figure = plt.figure()
        self.tf_figure.set_tight_layout(True)
        self.tf_ax = self.tf_figure.subplots(nrows=1, ncols=1)
        self.tf_canvas = FigureCanvas(self.tf_figure)
        self.timeFrequencyArea.setWidget(self.tf_canvas)

    def save_spectrum(self):
        """
        save figure to local path, popup a dialog for path selecting
        :return:
        """

        # save spectrum
        fd, type_ = QFileDialog.getSaveFileName(self, "Save figure and data",
                                                'E:/spectrum_' + str(self.start_end[0]) + '_'
                                                + str(self.start_end[1]), "*.eps;;*.png;;*.tif;;*.pdf;;")
        if fd == '':
            return

        self.setDisabled(True)
        self.spectrum_figure.savefig(fd, dpi=300)
        # extent = self.spectrum_ax[0].get_window_extent()
        # self.spectrum_figure.savefig(fd, bbox_inches=extent)
        # .savefig(fd, dpi=350)
        data_path = fd[:-4]
        fd = data_path + '_data.csv'

        data_arr = np.array([self.spectrum_F, self.spectrum_P]).transpose()
        np.savetxt(fd, X=data_arr, delimiter=',')

        self.setEnabled(True)

    def save_TF(self):
        """
        save figure to local path, popup a dialog for path selecting
        :return:
        """

        # save spectrum
        fd, type_ = QFileDialog.getSaveFileName(self, "Save figure and data",
                                                'E:/time_frequency_' + str(self.start_end[0]) + '_'
                                                + str(self.start_end[1]), "*.png;;*.tif;;*.pdf;;*.eps;;")
        if fd == '':
            return

        self.setDisabled(True)
        self.tf_figure.savefig(fd, dpi=300)
        # extent = self.spectrum_ax[0].get_window_extent()
        # self.spectrum_figure.savefig(fd, bbox_inches=extent)
        # .savefig(fd, dpi=350)
        # data_path = fd[:-4]
        # fd = data_path + '_data.csv'

        # data_arr = np.array([self.spectrum_F, self.spectrum_P]).transpose()
        # np.savetxt(fd, X=data_arr, delimiter=',')

        self.setEnabled(True)

    def closeEvent(self, event):
        """
        Rewrite close function of dialog
        :param event:
        :return:
        """
        event.ignore()
        self.hide()

# -*- coding: UTF-8 -*-
"""
@Project: MiSleep 
@File: auto_stage.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2023/10/23 9:34 
@Description:  Auto stage with models
"""
import copy

import antropy
import numpy as np
import pandas as pd
from scipy.stats import stats
import joblib

from MiSleep.utils.utils import get_epoch_spectrum, get_ave_bands


def constrains(y_prob, y_label, NREM_trans_threshold, REM_trans_threshold, WAKE_trans_threshold, REM_threshold):
    # if there is a transition, the probility must higher than transition threshold
    for idx, label in enumerate(y_label):
        if y_prob[idx][1] >= REM_threshold:
            y_label[idx] = 2
        if idx >= 1 and y_label[idx] != y_label[idx - 1]:
            if y_label[idx] == 1 and y_prob[idx][label - 1] < NREM_trans_threshold:
                y_label[idx] = y_label[idx - 1]
            elif y_label[idx] == 2 and y_prob[idx][label - 1] < REM_trans_threshold:
                y_label[idx] = y_label[idx - 1]
            elif y_label[idx] == 3 and y_prob[idx][label - 1] < WAKE_trans_threshold:
                y_label[idx] = y_label[idx - 1]

    for idx, label in enumerate(y_label):
        if 1 <= idx <= len(y_label) - 2 and y_label[idx - 1] == y_label[idx + 1]:
            y_label[idx] = y_label[idx + 1]
        # No REM after WAKE, set it to NREM
        if idx >= 1 and y_label[idx] == 2 and y_label[idx - 1] == 3:
            y_label[idx] = 1

    return y_label


class Auto_stage:
    """
    Should have the same input format, contains data(1 EEG and 1 EMG data), specific model type
    And the output is a list of label and  transfer to MiSleep label format
    """

    def __init__(self, data, model, model_type, epoch_length, SR):
        self.model = model
        self.model_type = model_type
        self.epoch_length = epoch_length
        self.SR = SR
        self.EEG = data[0]
        self.EMG = data[1]

        self.epoch_EEG = None
        self.epoch_EMG = None
        self.construct_epoch_data()
        self.epoch_features_df = pd.DataFrame()

        self.predicted_label = []

        if self.model_type == "lightGBM":
            self.lightGBM_model()

    def construct_epoch_data(self):
        self.epoch_EEG = [self.EEG[i:i + self.epoch_length * self.SR] for i in
                          range(0, len(self.EEG), self.epoch_length * self.SR)][:-1]
        self.epoch_EMG = [self.EMG[i:i + self.epoch_length * self.SR] for i in
                          range(0, len(self.EMG), self.epoch_length * self.SR)][:-1]

    def extract_features(self):
        # Calculation different time feature of EEG and EMG based on epoch
        epoch_features_df = pd.DataFrame({'epoch': range(len(self.epoch_EEG))})
        # Standard deviation
        epoch_EEG_SD = [np.var(each) for each in self.epoch_EEG]
        epoch_EMG_SD = [np.var(each) for each in self.epoch_EMG]
        epoch_features_df['EEG_SD'] = epoch_EEG_SD
        epoch_features_df['EMG_SD'] = epoch_EMG_SD

        # Set those abnormal points to the 0.9 of max
        EEG_SD_upper_quartile = epoch_features_df['EEG_SD'].quantile(0.9)
        EMG_SD_upper_quartile = epoch_features_df['EMG_SD'].quantile(0.9)
        epoch_features_df['EEG_SD'] = epoch_features_df['EEG_SD'].apply(
            lambda x: x if x < EEG_SD_upper_quartile else EEG_SD_upper_quartile)
        epoch_features_df['EMG_SD'] = epoch_features_df['EMG_SD'].apply(
            lambda x: x if x < EMG_SD_upper_quartile else EMG_SD_upper_quartile)
        epoch_features_df['EEG_SD_norm'] = epoch_features_df[['EEG_SD']].apply(
            lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        epoch_features_df['EMG_SD_norm'] = epoch_features_df[['EMG_SD']].apply(
            lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

        # Calculate only for EEG
        epoch_features_df['EEG_skewness'] = [stats.skew(each) for each in self.epoch_EEG]
        epoch_features_df['EEG_kurtosis'] = [stats.kurtosis(each) for each in self.epoch_EEG]

        epoch_features_df['EMG_skewness'] = [stats.skew(each) for each in self.epoch_EMG]
        epoch_features_df['EMG_kurtosis'] = [stats.kurtosis(each) for each in self.epoch_EMG]

        # normalize skewness and kurtosis
        epoch_features_df[['EEG_skewness_norm', 'EEG_kurtosis_norm', 'EMG_skewness_norm', 'EMG_kurtosis_norm']] = \
            epoch_features_df[['EEG_skewness', 'EEG_kurtosis', 'EMG_skewness', 'EMG_kurtosis']].apply(
                lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

        # Rate of zero cross
        epoch_features_df['EEG_rate_zerocross'] = [antropy.num_zerocross(each) / (self.epoch_length * self.SR) for each
                                                   in self.epoch_EEG]
        epoch_features_df['EMG_rate_zerocross'] = [antropy.num_zerocross(each) / (self.epoch_length * self.SR) for each
                                                   in self.epoch_EMG]

        # Hjorth mobility and complexity
        epoch_features_df[['EEG_Hjorth_M', 'EEG_Hjorth_C']] = [antropy.hjorth_params(each) for each in self.epoch_EEG]
        epoch_features_df[['EMG_Hjorth_M', 'EMG_Hjorth_C']] = [antropy.hjorth_params(each) for each in self.epoch_EMG]

        # Permutation entropy
        epoch_features_df['EEG_perm_entropy'] = [antropy.perm_entropy(each) for each in self.epoch_EEG]
        epoch_features_df['EMG_perm_entropy'] = [antropy.perm_entropy(each) for each in self.epoch_EMG]

        # Frequency features for EEG
        epoch_eeg_spectrum = get_epoch_spectrum(data=self.epoch_EEG, SR=self.SR)
        epoch_band_power_ave_percentage = np.array([get_ave_bands(each[0], each[1]) for each in epoch_eeg_spectrum])

        epoch_features_df[
            ['EEG_delta', 'EEG_theta', 'EEG_alpha', 'EEG_beta', 'EEG_gamma']] = epoch_band_power_ave_percentage
        epoch_features_df['EEG_alpha_theta'] = epoch_features_df['EEG_alpha'] / epoch_features_df['EEG_theta']
        epoch_features_df['EEG_delta_beta'] = epoch_features_df['EEG_delta'] / epoch_features_df['EEG_beta']
        epoch_features_df['EEG_delta_theta'] = epoch_features_df['EEG_delta'] / epoch_features_df['EEG_theta']

        self.epoch_features_df = epoch_features_df[
            ['epoch', 'EEG_SD_norm', 'EMG_SD_norm', 'EEG_skewness_norm', 'EEG_kurtosis_norm',
             'EMG_skewness_norm', 'EMG_kurtosis_norm', 'EEG_rate_zerocross', 'EMG_rate_zerocross',
             'EEG_Hjorth_M', 'EEG_Hjorth_C', 'EMG_Hjorth_M', 'EMG_Hjorth_C', 'EEG_perm_entropy',
             'EMG_perm_entropy', 'EEG_delta', 'EEG_theta', 'EEG_alpha', 'EEG_beta', 'EEG_gamma',
             'EEG_alpha_theta', 'EEG_delta_beta', 'EEG_delta_theta']]

    def lightGBM_model(self):
        """
        Use lightGBM model for prediction
        :return:
        :rtype:
        """

        self.extract_features()

        X = self.epoch_features_df.drop('epoch', axis=1)
        del self.epoch_features_df
        # Do feature mining first
        chi_model = joblib.load('../models/chi_model.pkl')
        X_chi_5 = chi_model.transform(X)

        X = pd.DataFrame(X_chi_5)

        gbm_model = self.model

        y_pred = gbm_model.predict(X)
        y_prob = gbm_model.predict_proba(X)

        REM_trans_threshold = 0.05
        WAKE_trans_threshold = 0.82
        NREM_trans_threshold = 0.7
        REM_threshold = 0.29

        y_label_prob = constrains(y_prob, copy.deepcopy(y_pred), NREM_trans_threshold,
                                  REM_trans_threshold, WAKE_trans_threshold, REM_threshold)

        self.predicted_label = y_label_prob

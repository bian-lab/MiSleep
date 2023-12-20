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
from scipy.stats import skew, kurtosis

# import antropy
import numpy as np
from pandas import DataFrame
from misleeputils import get_artifacts, z_score_norm, get_frequency_features, num_zerocross, perm_entropy, hjorth_params
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
        self.epoch_features_df = DataFrame()

        self.predicted_label = []

        if self.model_type == "lightGBM_1EEG_1EMG":
            self.lightGBM_model()
        elif self.model_type == "lightGBM_1EEG_1EMG_19features":
            self.lightGBM_model_2()

    def construct_epoch_data(self):
        self.epoch_EEG = [self.EEG[i:i + self.epoch_length * self.SR] for i in
                          range(0, len(self.EEG), self.epoch_length * self.SR)][:-1]
        self.epoch_EMG = [self.EMG[i:i + self.epoch_length * self.SR] for i in
                          range(0, len(self.EMG), self.epoch_length * self.SR)][:-1]

    def extract_features(self):
        # Calculation different time feature of EEG and EMG based on epoch
        epoch_features_df = DataFrame({'epoch': range(len(self.epoch_EEG))})
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
        epoch_features_df['EEG_skewness'] = [skew(each) for each in self.epoch_EEG]
        epoch_features_df['EEG_kurtosis'] = [kurtosis(each) for each in self.epoch_EEG]

        epoch_features_df['EMG_skewness'] = [skew(each) for each in self.epoch_EMG]
        epoch_features_df['EMG_kurtosis'] = [kurtosis(each) for each in self.epoch_EMG]

        # normalize skewness and kurtosis
        epoch_features_df[['EEG_skewness_norm', 'EEG_kurtosis_norm', 'EMG_skewness_norm', 'EMG_kurtosis_norm']] = \
            epoch_features_df[['EEG_skewness', 'EEG_kurtosis', 'EMG_skewness', 'EMG_kurtosis']].apply(
                lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

        # Rate of zero cross
        epoch_features_df['EEG_rate_zerocross'] = [num_zerocross(each) / (self.epoch_length * self.SR) for each
                                                   in self.epoch_EEG]
        epoch_features_df['EMG_rate_zerocross'] = [num_zerocross(each) / (self.epoch_length * self.SR) for each
                                                   in self.epoch_EMG]

        # Hjorth mobility and complexity
        epoch_features_df[['EEG_Hjorth_M', 'EEG_Hjorth_C']] = [hjorth_params(each) for each in self.epoch_EEG]
        epoch_features_df[['EMG_Hjorth_M', 'EMG_Hjorth_C']] = [hjorth_params(each) for each in self.epoch_EMG]

        # Permutation entropy
        epoch_features_df['EEG_perm_entropy'] = [perm_entropy(each) for each in self.epoch_EEG]
        epoch_features_df['EMG_perm_entropy'] = [perm_entropy(each) for each in self.epoch_EMG]

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

    def lightGBM_model_2(self):
        """
        From analysis_ipynb feature_extraction_lightGBM
        :return:
        :rtype:
        """

        X = [self.epoch_EEG, self.epoch_EMG]

        # Get artifacts index
        EEG_artifacts_idx = get_artifacts(X[0], threshold=5)
        EMG_artifacts_idx = get_artifacts(X[1], threshold=5)
        artifacts_idx = np.array(list(set(EEG_artifacts_idx + EMG_artifacts_idx)))

        # Set the artifacts label to init
        # original_Y = []
        # original_Y = [each if each not in artifacts_idx else 4 for idx, each in enumerate(test_Y_1)]

        # Normalization data
        # Remove artifact data and labels
        X = np.array([[each for idx, each in enumerate(X[0]) if idx not in artifacts_idx],
                      [each for idx, each in enumerate(X[1]) if idx not in artifacts_idx]])

        # test_Y_1 = np.array([each for idx, each in enumerate(test_Y_1) if idx not in artifacts_idx])
        EEG_data_norm = z_score_norm(X[0].flatten())
        EMG_data_norm = z_score_norm(X[1].flatten())
        X = np.array(
            [[EEG_data_norm[i:i + self.epoch_length * self.SR] for i in
              range(0, len(EEG_data_norm), self.epoch_length * self.SR)],
             [EMG_data_norm[i:i + self.epoch_length * self.SR] for i in
              range(0, len(EMG_data_norm), self.epoch_length * self.SR)]])

        X = np.concatenate((X[0], X[1]), axis=0).reshape(
            (X.shape[1], X.shape[0], X.shape[2]), order="F")

        del EEG_data_norm
        del EMG_data_norm

        eeg_time_features = np.array([np.array(
            [np.std(each[0]), skew(each[0]), kurtosis(each[0]), num_zerocross(each[0]),
             perm_entropy(each[0])] + get_frequency_features(each[0])) for each in X])
        emg_time_features = np.array([np.array(
            [np.std(each[1]), skew(each[1]), kurtosis(each[1]), num_zerocross(each[1]),
             perm_entropy(each[1])]) for each in X])
        features = np.concatenate((eeg_time_features, emg_time_features), axis=1)
        del X
        del eeg_time_features
        del emg_time_features

        gbm_model = self.model

        y_pred = gbm_model.predict(features, num_iteration=gbm_model.best_iteration_)

        y_prob = gbm_model.predict_proba(features)

        REM_trans_threshold = 0.05
        WAKE_trans_threshold = 0.82
        NREM_trans_threshold = 0.7
        REM_threshold = 0.29

        y_label_prob = constrains(y_prob, copy.deepcopy(y_pred), NREM_trans_threshold,
                                  REM_trans_threshold, WAKE_trans_threshold, REM_threshold)

        for each in sorted(artifacts_idx):
            y_label_prob = np.insert(y_label_prob, each, 4)

        self.predicted_label = y_label_prob

        # self.predicted_label = y_pred

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
        # chi_model = joblib.load('MiSleep/models/chi_model.pkl')

        # Debug use this path
        chi_model = joblib.load(r'MiSleep\models\chi_model.pkl')

        X_chi_5 = chi_model.transform(X)

        X = DataFrame(X_chi_5)

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
        # self.predicted_label = y_pred

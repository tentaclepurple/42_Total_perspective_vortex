#!/usr/bin/env python
# coding: utf-8

import mne
import warnings
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from IPython.display import display, HTML
from typing import List
import pickle

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.exceptions import ConvergenceWarning

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP


def ica_filter(raw, picks):
    raw.filter(l_freq=1.0, h_freq=None)
    ica = mne.preprocessing.ICA(n_components=20, random_state=42)
    ica.fit(raw, picks=picks)
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='Fpz')
    ica.exclude.extend(eog_indices)
    ica.apply(raw, exclude=ica.exclude)

    return raw


def filter_alpha_beta(raw):
    raw.filter(l_freq=8.0, h_freq=30.0)


def preprocess(raw, picks):
    raw_clean = raw.copy()
    ica_filter(raw_clean, picks)
    filter_alpha_beta(raw_clean)


def save_interactive_plot(raw, filename):
    ask = input("Do you want to plot an interactive graphic? (y/n)")
    if ask == 'y':
        with open(f'data/{filename}.pkl', 'wb') as f:
            pickle.dump(raw, f)
        print("Now open a terminal and type: python3 utils/plot.py")
    else:
        print("See you soon!")


def rename_chan(raw) -> None:
    mapping = {
        'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz', 'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6',
        'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6',
        'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2', 'Cp4.': 'CP4', 'Cp6.': 'CP6',
        'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2', 'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8',
        'F7..': 'F7', 'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8',
        'Ft7.': 'FT7', 'Ft8.': 'FT8', 'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10', 'Tp7.': 'TP7', 'Tp8.': 'TP8',
        'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8',
        'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4', 'Po8.': 'PO8', 'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2', 'Iz..': 'Iz'
    }
    raw.rename_channels(mapping)


def load_data(subjects, runs):
    """
    Load and preprocess data for the given subjects and runs.
    """
    all_raws = []
    for subject in subjects:
        raw_fnames = eegbci.load_data(subject, runs)
        raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
        raw = concatenate_raws(raws)

        rename_chan(raw)

        all_raws.append(raw)

    # Concatenate all raw data from different subjects
    raw = concatenate_raws(all_raws)

    return raw


def load_and_preprocess_data(subjects: list[int], runs: list[int]) -> Epochs:
    """
    Parameters:
        subjects: List of subject IDs to load the data for.
        runs: List of run IDs corresponding to the experiments.

    Returns:
        Epochs: Concatenated epochs of the preprocessed EEG data.
    """
    all_epochs = []
    for subject in subjects:
        raw_fnames = eegbci.load_data(subject, runs)
        raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
        raw = concatenate_raws(raws)

        rename_chan(raw)

        raw.notch_filter(freqs=60)

        raw = ica_filter(raw, picks=pick_types(raw.info, eeg=True))

        montage = make_standard_montage('standard_1005')
        raw.set_montage(montage)
        #filter_alpha_beta(raw)
        #raw.filter(7., 32., fir_design='firwin', skip_by_annotation='edge')

        events, _ = events_from_annotations(raw)

        event_id = dict(T1=1, T2=2)  # Only keep T1 and T2

        tmin, tmax = -1, 2
        epochs = Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, proj=True, picks=pick_types(raw.info, eeg=True), baseline=None, preload=True)

        #epochs = epochs.crop(tmin= 1., tmax=2.)

        all_epochs.append(epochs)

    return mne.concatenate_epochs(all_epochs)

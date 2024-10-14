#!/usr/bin/env python
# coding: utf-8

import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.preprocessing import ICA

import pandas as pd
import matplotlib.pyplot as plt


""" def rename_chan(raw) -> None:
    '''
    Rename channels in an MNE raw object to match the standard 1005 montage.

    Parameters:
        raw: The MNE raw object containing EEG data. The channels in this object will be renamed according to the standard 1005 montage.

    Returns:
        None: The function modifies the `raw` object in place and does not return anything.
    '''
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
    raw.rename_channels(mapping) """


def load_data(subjects, runs):
    """
    Load and preprocess data for the given subjects and runs.
    """
    all_raws = []
    for subject in subjects:
        raw_fnames = eegbci.load_data(subject, runs)
        raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
        raw = concatenate_raws(raws)

        #rename_chan(raw)

        all_raws.append(raw)

    # Concatenate all raw data from different subjects
    raw = concatenate_raws(all_raws)

    return raw


def ica_filter(raw, picks):
    raw.filter(l_freq=1.0, h_freq=None)
    ica = mne.preprocessing.ICA(n_components=20, random_state=42)
    ica.fit(raw, picks=picks)
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='Fpz.')
    ica.excluda.extend(eog_indices)
    ica.apply(raw, exclude=ica.exclude)


def filter_alpha_beta(raw):
    raw.filter(l_freq=8.0, h_freq=30.0)


def preprocess(raw, picks):
    raw_clean = raw.copy()
    ica_filter(raw_clean, picks)
    filter_alpha_beta(raw_clean)

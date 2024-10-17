import mne
import warnings
import numpy as np
import pandas as pd
import time
import pywt
from tqdm import tqdm
from IPython.display import display, HTML
from typing import List

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
from sklearn.model_selection import cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP



class FourierTransform(BaseEstimator, TransformerMixin):
    """
    Applies Fourier Transform to EEG data.
    """

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'FourierTransform':
        """
        Fit the FourierTransform (no operation, included for compatibility).

        Parameters:
            X (np.ndarray): Input data of shape (n_epochs, n_channels, n_times).
            y (np.ndarray, optional): Target values (ignored).

        Returns:
            FourierTransform: The transformer instance.
        """
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Apply the Fourier Transform to the input data.

        Parameters:
            X (np.ndarray): Input data of shape (n_epochs, n_channels, n_times).
            y (np.ndarray, optional): Target values (ignored).

        Returns:
            np.ndarray: The magnitude of the Fourier coefficients.
        """
        X_fft = np.fft.rfft(X, axis=-1)
        return np.abs(X_fft)

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this transformer.

        Parameters:
            deep (bool): If True, will return the parameters for this transformer and contained subobjects that are estimators.

        Returns:
            dict: Parameters of the transformer.
        """
        return {}

    def set_params(self, **params) -> 'FourierTransform':
        """
        Set the parameters of the transformer.

        Parameters:
            params (dict): Dictionary of parameters to set.

        Returns:
            FourierTransform: The transformer instance with updated parameters.
        """
        return self


class Float64CSP(CSP):
    """
    CSP class that converts input data to float64 for precision.
    Another type of Dimensionality Reduction.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Float64CSP':
        """
        Override the fit method to convert input data to float64.
        """
        X = X.astype(np.float64)
        return super().fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using the fitted CSP model.

        Parameters:
            X (np.ndarray): The input data to transform, of shape (n_epochs, n_channels, n_times).

        Returns:
            np.ndarray: The transformed data.
        """
        X = X.astype(np.float64)
        return super().transform(X)


class EEGStandardScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler for 3D EEG data arrays.
    """

    def __init__(self) -> None:
        """
        Initializes the EEGStandardScaler.
        """
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'EEGStandardScaler':
        """
        Fits the scaler on the EEG data.

        Parameters:
            X (np.ndarray): Input data of shape (n_epochs, n_channels, n_times).
            y (np.ndarray, optional): Target values (ignored).

        Returns:
            EEGStandardScaler: Returns the fitted scaler instance.
        """
        n_epochs, n_channels, n_times = X.shape
        self.scaler.fit(X.reshape(n_epochs * n_channels, n_times))
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Transforms the EEG data using the fitted scaler.

        Parameters:
            X (np.ndarray): Input data of shape (n_epochs, n_channels, n_times).
            y (np.ndarray, optional): Target values (ignored).

        Returns:
            np.ndarray: Scaled data of shape (n_epochs, n_channels, n_times).
        """
        n_epochs, n_channels, n_times = X.shape
        X_scaled = self.scaler.transform(X.reshape(n_epochs * n_channels, n_times))
        return X_scaled.reshape(n_epochs, n_channels, n_times)

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Parameters:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            dict: Parameters of the estimator.
        """
        return {}

    def set_params(self, **kwargs) -> 'EEGStandardScaler':
        """
        Set the parameters of the scaler.

        Parameters:
            params (dict): Dictionary of parameters to set.

        Returns:
            EEGStandardScaler: The scaler instance with updated parameters.
        """
        self.scaler.set_params(**kwargs)
        return self


class WaveletTransform:
    def __init__(self, wavelet='db4', level=4):
        self.wavelet = wavelet
        self.level = level
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_wavelet = []
        for trial in X:
            trial_wavelet = [pywt.wavedec(trial_ch, self.wavelet, level=self.level) for trial_ch in trial]
            trial_wavelet = np.hstack([np.hstack(coeffs) for coeffs in trial_wavelet])
            X_wavelet.append(trial_wavelet)
        
        return np.array(X_wavelet)

import numpy as np
import pywt

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

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


class MyPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        covariance_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        eigenvalues = eigenvalues[sorted_idx]

        self.components_ = eigenvectors[:, :self.n_components]

        return self

    def transform(self, X):
        X_centered = X - self.mean_
        X_transformed = np.dot(X_centered, self.components_)
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

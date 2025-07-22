"""
Feature engineering and transformation utilities for time series data.
"""
from typing import List, Tuple, Optional, Union
import numpy as np
import pywt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class WaveletTransform:
    """
    Multi-resolution wavelet transform for denoising and feature extraction.
    """
    
    def __init__(self, wavelet: str = 'db4', levels: int = 4):
        """
        Initialize the WaveletTransform.
        
        Args:
            wavelet: Wavelet type (default: 'db4' - Daubechies 4)
            levels: Number of decomposition levels
        """
        self.wavelet = wavelet
        self.levels = levels
        self.coefficients = None
    
    def decompose(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Decompose signal using wavelet transform.
        
        Args:
            signal: 1D array of signal values
            
        Returns:
            List of coefficient arrays [cA_n, cD_n, cD_n-1, ..., cD_1]
        """
        self.coefficients = pywt.wavedec(signal, self.wavelet, level=self.levels)
        return self.coefficients
    
    def denoise(self, signal: np.ndarray, threshold_mode: str = 'soft', 
                threshold_factor: float = 0.1) -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.
        
        Args:
            signal: 1D array of signal values
            threshold_mode: 'soft' or 'hard' thresholding
            threshold_factor: Multiplier for threshold calculation
            
        Returns:
            Denoised signal
        """
        coeffs = self.decompose(signal)
        
        # Calculate adaptive threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Median Absolute Deviation
        threshold = threshold_factor * sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Apply thresholding to detail coefficients
        coeffs_thresh = coeffs.copy()
        coeffs_thresh[1:] = [
            pywt.threshold(c, threshold, mode=threshold_mode)
            for c in coeffs[1:]
        ]
        
        # Reconstruct signal
        denoised_signal = pywt.waverec(coeffs_thresh, self.wavelet)
        return denoised_signal
    
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract multi-resolution features from signal.
        
        Args:
            signal: 1D array of signal values
            
        Returns:
            Feature vector with statistical properties of wavelet coefficients
        """
        coeffs = self.decompose(signal)
        
        # Extract statistical features from each level
        features = []
        for i, coeff in enumerate(coeffs):
            if i == 0:  # Approximation coefficients
                prefix = f'A{self.levels}'
            else:  # Detail coefficients
                level = self.levels - i + 1
                prefix = f'D{level}'
            
            features.extend([
                np.mean(coeff, dtype=np.float32),
                np.std(coeff, dtype=np.float32),
                np.var(coeff, dtype=np.float32),
                np.max(coeff, dtype=np.float32),
                np.min(coeff, dtype=np.float32),
                np.median(coeff, dtype=np.float32),
                np.sum(coeff**2, dtype=np.float32)  # Energy
            ])
        
        return np.array(features, dtype=np.float32)


class FeatureEngineering:
    """
    Feature engineering and preprocessing for time series data.
    """
    
    def __init__(self, scaler_type: str = 'minmax', wavelet: str = 'db4', wavelet_levels: int = 4):
        """
        Initialize the FeatureEngineering.
        
        Args:
            scaler_type: Type of scaler to use ('minmax' or 'standard')
            wavelet: Type of wavelet to use (default: 'db4' - Daubechies 4)
            wavelet_levels: Number of wavelet decomposition levels
        """
        self.scaler_type = scaler_type
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        self.scalers = {}
        self.feature_names = None
    
    def create_sequences(self, data: np.ndarray, 
                        sequence_length: int,
                        target_col: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: 2D array of shape (n_samples, n_features)
            sequence_length: Length of input sequences
            target_col: Index of the target column in data
            
        Returns:
            Tuple of (X, y) where X has shape (n_sequences, sequence_length, n_features)
            and y has shape (n_sequences,)
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, target_col])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def fit_scaler(self, X: np.ndarray):
        """
        Fit scaler to the data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        """
        if self.scaler_type == 'minmax':
            self.scalers['feature'] = MinMaxScaler()
        else:
            self.scalers['feature'] = StandardScaler()
            
        self.scalers['feature'].fit(X.reshape(-1, X.shape[-1]))
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features using the fitted scaler.
        
        Args:
            X: Input data of shape (n_samples, sequence_length, n_features)
            
        Returns:
            Scaled features with the same shape as input
        """
        if 'feature' not in self.scalers:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scalers['feature'].transform(X_flat)
        return X_scaled.reshape(original_shape)
    
    def fit_target_scaler(self, y: np.ndarray):
        """
        Fit target scaler.
        
        Args:
            y: Target values of shape (n_samples,)
        """
        self.scalers['target'] = MinMaxScaler()
        self.scalers['target'].fit(y.reshape(-1, 1))
    
    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Scale target values.
        
        Args:
            y: Target values of shape (n_samples,)
            
        Returns:
            Scaled target values
        """
        if 'target' not in self.scalers:
            raise ValueError("Target scaler not fitted. Call fit_target_scaler first.")
        
        return self.scalers['target'].transform(y.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform target values.
        
        Args:
            y: Scaled target values of shape (n_samples,)
            
        Returns:
            Original scale target values
        """
        if 'target' not in self.scalers:
            return y
        return self.scalers['target'].inverse_transform(y.reshape(-1, 1)).flatten()
        
    def extract_wavelet_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract wavelet-based features from time series data.
        
        Args:
            data: 2D array of shape (n_samples, n_features)
            
        Returns:
            Array of wavelet-based features with shape (n_samples, n_features * (wavelet_levels + 1))
        """
        n_samples, n_features = data.shape
        wavelet = pywt.Wavelet(self.wavelet)
        
        # Initialize array for wavelet features
        n_wavelet_features = n_features * (self.wavelet_levels + 1)
        wavelet_features = np.zeros((n_samples, n_wavelet_features))
        
        for i in range(n_features):
            # Handle any NaN values by forward filling
            signal = data[:, i]
            valid_idx = ~np.isnan(signal)
            if not np.all(valid_idx):
                signal = pd.Series(signal).ffill().values
            
            # Decompose signal using wavelet transform
            coeffs = pywt.wavedec(signal, wavelet, level=self.wavelet_levels)
            
            # Reconstruct each level and store as features
            for j in range(len(coeffs)):
                # Reconstruct the signal using only coefficients up to this level
                rec_coeffs = [np.zeros_like(c) for c in coeffs]
                rec_coeffs[j] = coeffs[j]
                rec_signal = pywt.waverec(rec_coeffs, wavelet)
                
                # Ensure the reconstructed signal has the same length as input
                if len(rec_signal) > n_samples:
                    rec_signal = rec_signal[:n_samples]
                elif len(rec_signal) < n_samples:
                    rec_signal = np.pad(rec_signal, (0, n_samples - len(rec_signal)), 'edge')
                
                # Store the reconstructed signal as a feature
                feature_idx = i * (self.wavelet_levels + 1) + j
                wavelet_features[:, feature_idx] = rec_signal
        
        return wavelet_features
    
    def wavelet_denoise(self, data: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Apply wavelet denoising to the input data.
        
        Args:
            data: 2D array of shape (n_samples, n_features)
            threshold: Threshold for wavelet coefficient thresholding
            
        Returns:
            Denoised data with the same shape as input
        """
        n_samples, n_features = data.shape
        wavelet = pywt.Wavelet(self.wavelet)
        denoised_data = np.zeros_like(data)
        
        for i in range(n_features):
            signal = data[:, i]
            
            # Handle any NaN values
            valid_idx = ~np.isnan(signal)
            if not np.all(valid_idx):
                signal = pd.Series(signal).ffill().values
            
            # Decompose signal
            coeffs = pywt.wavedec(signal, wavelet, level=self.wavelet_levels)
            
            # Apply thresholding to detail coefficients
            thresholded_coeffs = [coeffs[0]]  # Keep approximation coefficients
            for j in range(1, len(coeffs)):
                # Soft thresholding
                thresholded_coeffs.append(
                    np.sign(coeffs[j]) * np.maximum(0, np.abs(coeffs[j]) - threshold * np.max(np.abs(coeffs[j])))
                )
            
            # Reconstruct signal
            denoised_signal = pywt.waverec(thresholded_coeffs, wavelet)
            
            # Ensure the reconstructed signal has the same length as input
            if len(denoised_signal) > n_samples:
                denoised_signal = denoised_signal[:n_samples]
            elif len(denoised_signal) < n_samples:
                denoised_signal = np.pad(denoised_signal, (0, n_samples - len(denoised_signal)), 'edge')
            
            denoised_data[:, i] = denoised_signal
        
        return denoised_data

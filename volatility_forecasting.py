# Financial Market Volatility Forecasting Using Multi-Resolution Wavelet Transform 
# and Hybrid Deep Learning Techniques

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import pywt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import yfinance as yf

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D,
    MultiHeadAttention, LayerNormalization, Add, GRU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber

# Statistical models for comparison
from statsmodels.tsa.arima.model import ARIMA
import optuna

# Technical indicators
import talib

import logging
import joblib
import os
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loading and preprocessing module for financial time series
    """
    
    def __init__(self, symbol: str = '^VIX', start_date: str = '2012-01-01', 
                 end_date: str = '2023-12-31'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load financial data from Yahoo Finance"""
        try:
            logger.info(f"Loading data for {self.symbol} from {self.start_date} to {self.end_date}")
            self.raw_data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            logger.info(f"Loaded {len(self.raw_data)} data points")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = data.copy()
        
        # Price-based indicators
        df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
        df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'].values)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'].values)
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values)
        
        # Volatility measures
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Realized_Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df.dropna()
    
    def preprocess_data(self) -> pd.DataFrame:
        """Complete data preprocessing pipeline"""
        if self.raw_data is None:
            self.load_data()
        
        # Calculate technical indicators
        self.processed_data = self.calculate_technical_indicators(self.raw_data)
        
        # Feature engineering
        self.processed_data['Price_Change'] = self.processed_data['Close'].pct_change()
        self.processed_data['High_Low_Ratio'] = (self.processed_data['High'] / 
                                               self.processed_data['Low'] - 1)
        self.processed_data['Open_Close_Ratio'] = (self.processed_data['Close'] / 
                                                 self.processed_data['Open'] - 1)
        
        return self.processed_data.dropna()

class WaveletTransform:
    """
    Multi-resolution wavelet transform for denoising and feature extraction
    """
    
    def __init__(self, wavelet: str = 'db4', levels: int = 4):
        self.wavelet = wavelet
        self.levels = levels
        self.coefficients = None
        
    def decompose(self, signal: np.ndarray) -> List[np.ndarray]:
        """Decompose signal using wavelet transform"""
        self.coefficients = pywt.wavedec(signal, self.wavelet, level=self.levels)
        return self.coefficients
    
    def denoise(self, signal: np.ndarray, threshold_mode: str = 'soft', 
                threshold_factor: float = 0.1) -> np.ndarray:
        """Denoise signal using wavelet thresholding"""
        coeffs = self.decompose(signal)
        
        # Calculate adaptive threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = threshold_factor * sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Apply thresholding to detail coefficients
        coeffs_thresh = coeffs.copy()
        coeffs_thresh[1:] = [pywt.threshold(c, threshold, mode=threshold_mode) 
                           for c in coeffs[1:]]
        
        # Reconstruct signal
        denoised_signal = pywt.waverec(coeffs_thresh, self.wavelet)
        return denoised_signal
    
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract multi-resolution features"""
        coeffs = self.decompose(signal)
        
        # Extract statistical features from each level
        features = []
        for coeff in coeffs:
            features.extend([
                np.mean(coeff),
                np.std(coeff),
                np.var(coeff),
                np.max(coeff),
                np.min(coeff),
                np.median(coeff)
            ])
        
        return np.array(features)

class FeatureEngineering:
    """
    Advanced feature engineering and selection
    """
    
    def __init__(self):
        self.scalers = {}
        self.selected_features = None
        
    def create_sequences(self, data: np.ndarray, seq_length: int, 
                        target_col: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length, target_col])
        return np.array(X), np.array(y)
    
    def scale_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                      X_test: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, ...]:
        """Scale features using specified method"""
        if method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        # Reshape for scaling
        original_shape_train = X_train.shape
        original_shape_val = X_val.shape
        original_shape_test = X_test.shape
        
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(original_shape_train)
        X_val_scaled = X_val_scaled.reshape(original_shape_val)
        X_test_scaled = X_test_scaled.reshape(original_shape_test)
        
        self.scalers['feature_scaler'] = scaler
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def scale_target(self, y_train: np.ndarray, y_val: np.ndarray, 
                    y_test: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Scale target variable"""
        scaler = MinMaxScaler()
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        self.scalers['target_scaler'] = scaler
        return y_train_scaled, y_val_scaled, y_test_scaled

class AttentionLSTM:
    """
    LSTM with Attention Mechanism for volatility forecasting with improved numerical stability
    """
    
    def __init__(self, input_shape: Tuple[int, int], lstm_units: int = 50,
                 attention_units: int = 64, dropout_rate: float = 0.2,
                 learning_rate: float = 0.001, clip_norm: float = 1.0):
        self.input_shape = input_shape  # (seq_length, n_features)
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.model = None
        self.history = None
        
    def _attention_layer(self, inputs):
        """Self-attention layer with residual connection"""
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=4, 
            key_dim=self.attention_units // 4,  # Split into 4 heads
            dropout=self.dropout_rate
        )(inputs, inputs)
        
        # Residual connection and layer norm
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        return attention_output
        
    def build_model(self) -> Model:
        """Build LSTM with attention mechanism and improved stability"""
        inputs = Input(shape=self.input_shape)
        
        # First LSTM layer with return sequences
        lstm1 = LSTM(
            self.lstm_units, 
            return_sequences=True, 
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate * 0.5,  # Lower recurrent dropout
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4)
        )(inputs)
        lstm1 = LayerNormalization(epsilon=1e-6)(lstm1)
        
        # Second LSTM layer
        lstm2 = LSTM(
            self.lstm_units, 
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate * 0.5,
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4)
        )(lstm1)
        lstm2 = LayerNormalization(epsilon=1e-6)(lstm2)
        
        # Attention mechanism
        attention_output = self._attention_layer(lstm2)
        
        # Global average pooling instead of flattening to reduce parameters
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
        
        # Dense layers with regularization
        dense1 = Dense(
            64, 
            activation='relu',
            kernel_regularizer=l2(1e-4),
            kernel_initializer='he_normal'
        )(pooled)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(self.dropout_rate)(dense1)
        
        dense2 = Dense(
            32, 
            activation='relu',
            kernel_regularizer=l2(1e-4),
            kernel_initializer='he_normal'
        )(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(self.dropout_rate)(dense2)
        
        # Output layer with linear activation for regression
        outputs = Dense(1, activation='linear', kernel_initializer='glorot_normal')(dense2)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile(self, learning_rate: float = None):
        """Compile the model with gradient clipping and learning rate scheduling"""
        if learning_rate is not None:
            self.learning_rate = learning_rate
            
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        # Optimizer with gradient clipping
        optimizer = Adam(
            learning_rate=lr_schedule,
            clipnorm=self.clip_norm
        )
        
        # Compile with Huber loss which is more robust to outliers
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(),
            metrics=['mae', 'mse']
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> dict:
        """Train the model with improved callbacks and validation"""
        if self.model is None:
            self.build_model()
            self.compile()
        
        # Early stopping and model checkpoint
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TerminateOnNaN()
        ]
        
        # Train with validation
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with input validation"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Ensure input has the right shape
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0)
            
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        # Handle potential NaN/Inf values
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            logger.warning("NaN or Inf values detected in predictions. Applying corrections.")
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
            
        return predictions.squeeze()
    
    def save(self, filepath: str):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a saved model"""
        model = tf.keras.models.load_model(filepath)
        # Create a new instance and attach the loaded model
        instance = cls(input_shape=model.input_shape[1:])
        instance.model = model
        return instance

class CNN_GRU_Attention:
    """
    3D-CNN-GRU with Attention for advanced pattern recognition
    """
    
    def __init__(self, seq_length: int, n_features: int, 
                 conv_filters: int = 64, gru_units: int = 50, dropout_rate: float = 0.2):
        self.seq_length = seq_length
        self.n_features = n_features
        self.conv_filters = conv_filters
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self) -> Model:
        """Build CNN-GRU with attention"""
        inputs = Input(shape=(self.seq_length, self.n_features))
        
        # CNN layers for local pattern extraction
        conv1 = Conv1D(filters=self.conv_filters, kernel_size=3, 
                       activation='relu', padding='same')(inputs)
        conv1 = MaxPooling1D(pool_size=2, padding='same')(conv1)
        conv1 = Dropout(self.dropout_rate)(conv1)
        
        conv2 = Conv1D(filters=self.conv_filters * 2, kernel_size=3, 
                       activation='relu', padding='same')(conv1)
        conv2 = MaxPooling1D(pool_size=2, padding='same')(conv2)
        conv2 = Dropout(self.dropout_rate)(conv2)
        
        # GRU layers for temporal dependencies
        gru_out = GRU(self.gru_units, return_sequences=True, 
                     dropout=self.dropout_rate)(conv2)
        gru_out = LayerNormalization()(gru_out)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=64)(gru_out, gru_out)
        attention = Dropout(self.dropout_rate)(attention)
        attention = Add()([gru_out, attention])
        attention = LayerNormalization()(attention)
        
        # Global pooling and dense layers
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
        
        dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(pooled)
        dense1 = Dropout(self.dropout_rate)(dense1)
        
        outputs = Dense(1, activation='linear')(dense1)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile_and_train(self, X_train, y_train, X_val, y_val, 
                         learning_rate=0.001, epochs=100, batch_size=32):
        """Compile and train the model"""
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

class BenchmarkModels:
    """
    Traditional benchmark models for comparison
    """
    
    def __init__(self):
        self.models = {}
        
    def fit_arima(self, data: np.ndarray, order: Tuple[int, int, int] = (1, 1, 1)):
        """Fit ARIMA model"""
        try:
            model = ARIMA(data, order=order)
            fitted_model = model.fit()
            self.models['ARIMA'] = fitted_model
            return fitted_model
        except Exception as e:
            logger.error(f"Error fitting ARIMA: {e}")
            return None
    

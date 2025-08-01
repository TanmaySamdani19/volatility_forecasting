"""
Deep learning models for volatility forecasting.
"""
import json
import os
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D,
    MultiHeadAttention, LayerNormalization, Add, GRU, GlobalAveragePooling1D,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all forecasting models."""
    
    def __init__(self, input_shape: Tuple[int, int]):
        """
        Initialize the base model.
        
        Args:
            input_shape: Tuple of (sequence_length, n_features)
        """
        self.input_shape = input_shape
        self.model = None
    
    def build_model(self) -> Model:
        """Build the model architecture."""
        raise NotImplementedError("Subclasses must implement build_model")
    
    def compile(self, learning_rate=0.001, gradient_clip=1.0):
        """
        Compile the model with improved settings.
        
        Args:
            learning_rate: Learning rate for the optimizer (float or LearningRateSchedule)
            gradient_clip: Maximum value for gradient clipping (default: 1.0)
        """
        if self.model is None:
            self.model = self.build_model()
        
        # Configure optimizer with gradient clipping
        # Only use one type of gradient clipping at a time
        optimizer_kwargs = {
            'learning_rate': learning_rate if hasattr(learning_rate, '__call__') else float(learning_rate),
            'epsilon': 1e-7,
            'global_clipnorm': gradient_clip  # Using global_clipnorm for better stability
        }
        
        optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)
        
        # Use Huber loss which is more robust to outliers than MSE
        # Also add gradient clipping in the loss function
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=[
                'mae',
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsolutePercentageError()
            ],
            # Add global gradient clipping
            run_eagerly=False  # Disable eager execution for better performance
        )
        
        # Enable mixed precision training if available
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info('Mixed precision enabled')
        except Exception as e:
            logger.warning(f'Could not enable mixed precision: {e}')
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 200, batch_size: int = 64,
              patience: int = 20, min_delta: float = 1e-4) -> dict:
        """
        Train the model with enhanced callbacks and data handling.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Patience for early stopping
            min_delta: Minimum change to qualify as an improvement
            
        Returns:
            Training history
        """
        if self.model is None:
            self.compile()
            
        # Ensure inputs are numpy arrays
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        
        # Check for NaNs or Infs
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            logger.warning("Training data contains NaN or Inf values")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            logger.warning("Training targets contain NaN or Inf values")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create output directory for callbacks
        import os
        import datetime
        from tensorflow.keras.callbacks import (
            EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
        )
        
        log_dir = os.path.join('logs', 'fit', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(log_dir, exist_ok=True)
        
        # Model checkpoint path
        checkpoint_path = os.path.join(log_dir, 'best_model.keras')
        
        # Define callbacks with improved settings
        callbacks = [
            # Early stopping with improved settings
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1,
                mode='min',
                baseline=None,
                start_from_epoch=10  # Start checking after 10 epochs
            ),
            
            # Learning rate scheduler with more aggressive reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(3, patience // 3),  # More frequent reduction
                min_lr=1e-6,
                min_delta=min_delta,
                verbose=1,
                mode='min'
            ),
            
            # Model checkpoint to save the best model
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                save_freq='epoch',
                verbose=1
            ),
            # TensorBoard for visualization
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch',
                profile_batch=0  # Disable profiling for better performance
            ),
            # CSV logger
            tf.keras.callbacks.CSVLogger(
                os.path.join(log_dir, 'training.log')
            )
        ]
        
        # Calculate sample weights if needed (e.g., for imbalanced data)
        sample_weight = self._calculate_sample_weights(y_train)
        
        try:
            # Prepare training arguments
            fit_kwargs = {
                'x': X_train,
                'y': y_train,
                'validation_data': (X_val, y_val),
                'epochs': epochs,
                'batch_size': batch_size,
                'callbacks': callbacks,
                'shuffle': True,  # Important for time series with proper validation split
                'verbose': 1
            }
            
            # Add sample_weight only if it's not None
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            
            # Train the model with validation
            history = self.model.fit(**fit_kwargs)
            
            # Load the best model weights from checkpoint
            if os.path.exists(checkpoint_path):
                try:
                    self.model = tf.keras.models.load_model(checkpoint_path, compile=False)
                    logger.info("Loaded best model weights from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load best model weights: {e}")
            
            # Save the final model
            final_model_path = os.path.join(log_dir, 'final_model')
            self.save(final_model_path)
            logger.info(f"Saved final model to {final_model_path}")
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            # Try to load the best model if available
            if os.path.exists(checkpoint_path):
                try:
                    self.model = tf.keras.models.load_model(checkpoint_path, compile=False)
                    logger.info("Loaded best available model after error")
                except Exception as load_error:
                    logger.error(f"Could not load best model after error: {load_error}")
            raise
    
    def _calculate_sample_weights(self, y):
        """
        Calculate sample weights to handle imbalanced data.
        For regression, we can create bins and assign weights based on bin frequency.
        """
        try:
            # Create bins for the target values
            y_flat = y.flatten()
            hist, bin_edges = np.histogram(y_flat, bins=10)
            
            # Calculate class weights (inverse of frequency)
            bin_indices = np.digitize(y_flat, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, len(hist) - 1)  # Ensure all indices are valid
            
            # Calculate weights
            class_weights = 1.0 / (hist[bin_indices] + 1e-6)  # Add small epsilon to avoid division by zero
            
            # Normalize weights
            class_weights = class_weights / np.sum(class_weights) * len(class_weights)
            
            return class_weights
            
        except Exception as e:
            logger.warning(f"Could not calculate sample weights: {str(e)}. Using uniform weights.")
            return None
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions with enhanced error handling and input validation.
        
        Args:
            X: Input features array of shape (n_samples, sequence_length, n_features)
            batch_size: Batch size for prediction (smaller batches use less memory)
            
        Returns:
            Array of predictions with shape (n_samples,)
            
        Raises:
            ValueError: If model is not built or input is invalid
            RuntimeError: If prediction fails
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        if not isinstance(X, np.ndarray) or len(X.shape) != 3:
            raise ValueError("Input must be a 3D numpy array of shape (samples, timesteps, features)")
            
        # Convert to float32 if needed
        X = X.astype(np.float32, copy=False)
        
        # Check for NaNs or Infs
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("Input contains NaN or Inf values. Replacing with zeros.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            # Make predictions in batches to avoid memory issues
            predictions = []
            n_samples = X.shape[0]
            
            for i in range(0, n_samples, batch_size):
                batch = X[i:i + batch_size]
                batch_pred = self.model.predict(batch, verbose=0)
                predictions.append(batch_pred)
                
            predictions = np.concatenate(predictions, axis=0)
            
            # Ensure predictions are 1D
            predictions = predictions.reshape(-1)
            
            # Check for NaN/Inf in predictions
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                logger.warning("Predictions contain NaN or Inf values. Replacing with zeros.")
                predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
                
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Failed to make predictions: {str(e)}")
    
    def save(self, filepath: str):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model not built. Nothing to save.")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save only the weights, not the entire model to avoid issues with custom objects
        self.model.save_weights(filepath + '.weights.h5')
        # Save the model config separately
        config = {
            'input_shape': self.input_shape,
            'model_params': {k: v for k, v in self.__dict__.items() 
                           if k not in ['model', 'input_shape']}
        }
        with open(filepath + '.config.json', 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, filepath: str, **kwargs):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model (without extension)
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Loaded model instance
        """
        # Load the config
        with open(filepath + '.config.json', 'r') as f:
            config = json.load(f)
        
        # Create a new instance of the model
        model = cls(input_shape=tuple(config['input_shape']), 
                   **config['model_params'],
                   **kwargs)
        
        # Build and load weights
        model.build_model()
        model.model.load_weights(filepath + '.weights.h5')
        
        return model


class AttentionLSTM(BaseModel):
    """
    LSTM with Attention Mechanism for volatility forecasting.
    """
    
    def __init__(self, input_shape: Tuple[int, int], 
                 lstm_units: int = 256,  # Increased capacity
                 attention_units: int = 128,
                 dropout_rate: float = 0.3,  # Slightly higher dropout
                 l2_reg: float = 0.001,  # Reduced L2 regularization
                 learning_rate: Union[float, Dict[str, Any]] = 0.001,
                 use_batch_norm: bool = True):
        """
        Initialize the enhanced AttentionLSTM model.
        
        Args:
            input_shape: Tuple of (sequence_length, n_features)
            lstm_units: Number of LSTM units in the first layer
            attention_units: Dimensionality of the attention layer
            dropout_rate: Dropout rate
            l2_reg: L2 regularization factor
            learning_rate: Learning rate for the optimizer (float or dict with schedule config)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__(input_shape)
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate  # Store the learning rate config
        self.use_batch_norm = use_batch_norm
    
    def build_model(self) -> Model:
        """Build the enhanced LSTM with attention architecture."""
        inputs = Input(shape=self.input_shape)
        x = inputs
        
        # Initial batch normalization
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        
        # First LSTM layer with skip connection
        lstm1 = LSTM(
            units=self.lstm_units,
            return_sequences=True,
            kernel_regularizer=l2(self.l2_reg),
            recurrent_regularizer=l2(self.l2_reg),
            recurrent_dropout=self.dropout_rate * 0.5
        )(x)
        lstm1 = LayerNormalization()(lstm1)
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        
        # Second LSTM layer with skip connection
        lstm2 = LSTM(
            units=self.lstm_units // 2,
            return_sequences=True,
            kernel_regularizer=l2(self.l2_reg),
            recurrent_regularizer=l2(self.l2_reg),
            recurrent_dropout=self.dropout_rate * 0.5
        )(lstm1)
        lstm2 = LayerNormalization()(lstm2)
        
        # Project to attention dimension
        attention_input = Dense(self.attention_units)(lstm2)
        
        # Multi-head attention with residual connection
        attention_output = MultiHeadAttention(
            num_heads=8,
            key_dim=self.attention_units // 8,
            value_dim=self.attention_units // 8,
            dropout=self.dropout_rate * 0.5
        )(attention_input, attention_input)
        
        # Add & Norm with residual connection
        attention = LayerNormalization(epsilon=1e-6)(attention_input + attention_output)
        attention = Dropout(self.dropout_rate)(attention)
        
        # Project back to LSTM dimension if needed
        if attention.shape[-1] != lstm2.shape[-1]:
            attention = Dense(lstm2.shape[-1])(attention)
        
        # Final residual connection from LSTM2
        attention = Add()([lstm2, attention])
        attention = LayerNormalization(epsilon=1e-6)(attention)
        
        # Global average pooling
        pooled = GlobalAveragePooling1D()(attention)
        
        # Enhanced dense layers with skip connections
        dense1 = self._dense_block(pooled, 256, dropout_rate=self.dropout_rate)
        dense2 = self._dense_block(dense1, 128, dropout_rate=self.dropout_rate)
        dense3 = self._dense_block(dense2, 64, dropout_rate=self.dropout_rate)
        
        # Output layer
        outputs = Dense(1, activation='linear')(dense3)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
        
    def _dense_block(self, x, units, dropout_rate=0.2):
        """Helper function to create a dense block with batch norm and dropout."""
        # Skip connection if dimensions match
        if x.shape[-1] == units:
            x_shortcut = x
        else:
            x_shortcut = Dense(units)(x)
        
        # Dense layer with activation
        x = Dense(units, activation='relu', 
                 kernel_regularizer=l2(self.l2_reg))(x)
        
        # Batch normalization if enabled
        if self.use_batch_norm:
            x = BatchNormalization()(x)
            
        x = Dropout(dropout_rate)(x)
        
        # Add skip connection
        x = Add()([x, x_shortcut])
        
        return x


class CNN_GRU_Attention(BaseModel):
    """
    CNN-GRU with Attention for volatility forecasting.
    Combines CNN for local pattern extraction with GRU for sequence modeling
    and attention for focusing on important time steps.
    """
    
    def __init__(self, input_shape: Tuple[int, int],
                 conv_filters: int = 64,
                 gru_units: int = 64,
                 attention_units: int = 32,
                 dropout_rate: float = 0.2,
                 l2_reg: float = 0.01,
                 learning_rate: Union[float, Dict[str, Any]] = 0.001):
        """
        Initialize the CNN_GRU_Attention model.
        
        Args:
            input_shape: Tuple of (sequence_length, n_features)
            conv_filters: Number of filters in CNN layers
            gru_units: Number of GRU units
            attention_units: Dimensionality of the attention layer
            dropout_rate: Dropout rate
            l2_reg: L2 regularization factor
            learning_rate: Learning rate for the optimizer (float or dict with schedule config)
            gru_units: Number of GRU units
            attention_units: Dimensionality of the attention layer
            dropout_rate: Dropout rate
            l2_reg: L2 regularization factor
        """
        super().__init__(input_shape)
        self.conv_filters = conv_filters
        self.gru_units = gru_units
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate  # Store the learning rate config
    
    def build_model(self) -> Model:
        """Build the CNN-GRU with attention architecture."""
        inputs = Input(shape=self.input_shape)
        
        # CNN layers for local pattern extraction
        conv1 = Conv1D(filters=self.conv_filters, 
                      kernel_size=3, 
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(self.l2_reg))(inputs)
        conv1 = MaxPooling1D(pool_size=2, padding='same')(conv1)
        conv1 = Dropout(self.dropout_rate)(conv1)
        
        conv2 = Conv1D(filters=self.conv_filters * 2, 
                      kernel_size=3, 
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(self.l2_reg))(conv1)
        conv2 = MaxPooling1D(pool_size=2, padding='same')(conv2)
        conv2 = Dropout(self.dropout_rate)(conv2)
        
        # GRU layer for temporal dependencies
        gru_out = GRU(self.gru_units, 
                     return_sequences=True,
                     kernel_regularizer=l2(self.l2_reg),
                     recurrent_regularizer=l2(self.l2_reg))(conv2)
        gru_out = LayerNormalization()(gru_out)
        
        # Attention mechanism
        attention = MultiHeadAttention(
            num_heads=4,
            key_dim=self.attention_units,
            value_dim=self.attention_units
        )([gru_out, gru_out])
        attention = Dropout(self.dropout_rate)(attention)
        
        # Residual connection
        attention = Add()([gru_out, attention])
        attention = LayerNormalization()(attention)
        
        # Global average pooling
        pooled = GlobalAveragePooling1D()(attention)
        
        # Dense layers
        dense1 = Dense(64, activation='relu',
                      kernel_regularizer=l2(self.l2_reg))(pooled)
        dense1 = Dropout(self.dropout_rate)(dense1)
        
        # Output layer
        outputs = Dense(1, activation='linear')(dense1)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


class BenchmarkModels:
    """
    Traditional statistical models for benchmarking.
    """
    
    def __init__(self):
        self.models = {}
    
    def fit_arima(self, data: np.ndarray, order: tuple = (1, 1, 1)) -> Any:
        """
        Fit an ARIMA model.
        
        Args:
            data: 1D array of time series data
            order: (p, d, q) order of the ARIMA model
            
        Returns:
            Fitted ARIMA model
        """
        from statsmodels.tsa.arima.model import ARIMA
        
        try:
            model = ARIMA(data, order=order)
            fitted_model = model.fit()
            self.models['ARIMA'] = fitted_model
            return fitted_model
        except Exception as e:
            logger.error(f"Error fitting ARIMA: {e}")
            return None
    
    def fit_garch(self, returns: np.ndarray, p: int = 1, q: int = 1) -> Any:
        """
        Fit a GARCH model.
        
        Args:
            returns: 1D array of returns (not prices)
            p: GARCH order
            q: ARCH order
            
        Returns:
            Fitted GARCH model
        """
        from arch import arch_model
        
        try:
            # Scale returns to avoid numerical issues
            scaled_returns = returns * 100
            model = arch_model(scaled_returns, vol='Garch', p=p, q=q)
            fitted_model = model.fit(disp='off')
            self.models['GARCH'] = fitted_model
            return fitted_model
        except Exception as e:
            logger.error(f"Error fitting GARCH: {e}")
            return None
    
    def predict_arima(self, steps: int = 1) -> np.ndarray:
        """Make predictions with ARIMA model."""
        if 'ARIMA' not in self.models:
            raise ValueError("ARIMA model not fitted. Call fit_arima first.")
        
        forecast = self.models['ARIMA'].get_forecast(steps=steps)
        return forecast.predicted_mean
    
    def predict_garch(self, steps: int = 1) -> np.ndarray:
        """Make volatility forecasts with GARCH model."""
        if 'GARCH' not in self.models:
            raise ValueError("GARCH model not fitted. Call fit_garch first.")
        
        forecasts = self.models['GARCH'].forecast(horizon=steps)
        # Return annualized volatility
        return np.sqrt(forecasts.variance.values[-1] * 252) / 100

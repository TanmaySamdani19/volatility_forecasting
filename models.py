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
    BatchNormalization, Embedding, Flatten, Reshape, Multiply
)
from tensorflow.keras import backend as K
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
    
    def compile(self, learning_rate=0.001):
        """
        Compile the model with improved settings.
        
        Args:
            learning_rate: Learning rate for the optimizer (float or LearningRateSchedule)
        """
        if self.model is None:
            self.model = self.build_model()
        
        # If learning_rate is a schedule, use it directly
        # Otherwise, create a fixed learning rate
        if hasattr(learning_rate, '__call__'):
            # It's a learning rate schedule
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=1.0,
                epsilon=1e-7
            )
        else:
            # It's a float learning rate
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=float(learning_rate),
                clipnorm=1.0,
                epsilon=1e-7
            )
        
        # Use Huber loss which is more robust to outliers than MSE
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=[
                'mae',
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsolutePercentageError()
            ]
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 200, batch_size: int = 64,
              patience: int = 20) -> dict:
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
            
        Returns:
            Training history
        """
        if self.model is None:
            self.compile()
        
        # Create output directory for callbacks
        import os
        import datetime
        log_dir = os.path.join('logs', 'fit', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(log_dir, exist_ok=True)
        
        # Enhanced callbacks
        callbacks = [
            # Early stopping with restoration of best weights
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            # Learning rate scheduler
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(5, patience // 4),  # More frequent LR reduction
                min_lr=1e-6,
                verbose=1,
                mode='min'
            ),
            # Model checkpointing
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'best_model.keras'),
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=0
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
            # Train the model with validation
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                sample_weight=sample_weight,
                shuffle=True,  # Important for time series with proper validation split
                verbose=1
            )
            
            # Save the best model
            best_model_path = os.path.join(log_dir, 'best_model')
            self.save(best_model_path)
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        return self.model.predict(X, verbose=0).flatten()
    
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
    Enhanced LSTM with Attention Mechanism for volatility forecasting.
    Includes improvements for numerical stability and training reliability.
    """
    
    def __init__(self, input_shape: Tuple[int, int], 
                 lstm_units: int = 128,
                 attention_units: int = 64,
                 dropout_rate: float = 0.2,
                 l2_reg: float = 1e-4,
                 learning_rate: Union[float, Dict[str, Any]] = 1e-3,
                 clip_norm: float = 1.0,
                 use_batch_norm: bool = True,
                 activation: str = 'tanh',
                 recurrent_activation: str = 'sigmoid'):
        """
        Initialize the enhanced AttentionLSTM model.
        
        Args:
            input_shape: Tuple of (sequence_length, n_features)
            lstm_units: Number of LSTM units in layers
            attention_units: Dimensionality of the attention layer
            dropout_rate: Dropout rate (0-1)
            l2_reg: L2 regularization factor
            learning_rate: Learning rate (float or dict with schedule config)
            clip_norm: Maximum gradient norm for clipping
            use_batch_norm: Whether to use batch normalization
            activation: Activation function for LSTM cells
            recurrent_activation: Recurrent activation function for LSTM cells
        """
        super().__init__(input_shape)
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self._validate_initialization()
    
    def _validate_initialization(self):
        """Validate initialization parameters."""
        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        if self.l2_reg < 0:
            raise ValueError(f"l2_reg must be >= 0, got {self.l2_reg}")
        if self.clip_norm <= 0:
            raise ValueError(f"clip_norm must be > 0, got {self.clip_norm}")
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and clean input data."""
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        
        # Check for NaN/Inf values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("Input contains NaN or Inf values. Replacing with nearest valid values.")
            X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, 
                            neginf=np.finfo(np.float32).min)
        
        # Ensure correct shape
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0)
        elif len(X.shape) != 3:
            raise ValueError(f"Input must be 2D or 3D array, got shape {X.shape}")
            
        return X
    
    def _attention_layer(self, inputs):
        """Self-attention layer with residual connection and layer norm."""
        # Multi-head attention with scaled dot-product attention
        attention_output = MultiHeadAttention(
            num_heads=4,
            key_dim=max(1, self.attention_units // 4),  # Ensure at least 1
            dropout=self.dropout_rate * 0.5,
            kernel_regularizer=l2(self.l2_reg),
            bias_regularizer=l2(self.l2_reg)
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        return attention_output
    
    def _dense_block(self, x, units, activation='relu', dropout_rate=0.2):
        """Helper function to create a dense block with batch norm and dropout."""
        # Weight initialization
        initializer = tf.keras.initializers.HeNormal()
        
        # Dense layer with kernel initialization and regularization
        x = Dense(
            units,
            activation=activation,
            kernel_initializer=initializer,
            kernel_regularizer=l2(self.l2_reg),
            bias_regularizer=l2(self.l2_reg)
        )(x)
        
        # Batch normalization if enabled
        if self.use_batch_norm:
            x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        
        # Dropout with rate scaling
        if dropout_rate > 0:
            x = Dropout(min(dropout_rate, 0.5))(x)  # Cap dropout at 0.5
            
        return x
    
    def _create_lstm_layer(self, units, return_sequences=False, return_state=False):
        """Create an LSTM layer with consistent configuration."""
        return LSTM(
            units=units,
            activation=self.activation,
            recurrent_activation=self.recurrent_activation,
            return_sequences=return_sequences,
            return_state=return_state,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            kernel_regularizer=l2(self.l2_reg),
            recurrent_regularizer=l2(self.l2_reg),
            bias_regularizer=l2(self.l2_reg),
            dropout=self.dropout_rate * 0.5,
            recurrent_dropout=self.dropout_rate * 0.3
        )
    
    def build_model(self) -> Model:
        """Build the enhanced LSTM with attention architecture."""
        # Input layer
        inputs = Input(shape=self.input_shape, name='input_layer')
        x = inputs
        
        # Input normalization
        if self.use_batch_norm:
            x = BatchNormalization(name='input_bn')(x)
        
        # First LSTM layer
        lstm1 = self._create_lstm_layer(
            units=self.lstm_units,
            return_sequences=True
        )(x)
        lstm1 = LayerNormalization(epsilon=1e-6, name='lstm1_ln')(lstm1)
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        
        # Second LSTM layer with residual connection
        lstm2 = self._create_lstm_layer(
            units=self.lstm_units // 2,
            return_sequences=True
        )(lstm1)
        lstm2 = LayerNormalization(epsilon=1e-6, name='lstm2_ln')(lstm2)
        
        # Attention mechanism with residual connection
        attention_output = self._attention_layer(lstm2)
        attention_output = Dropout(self.dropout_rate)(attention_output)
        
        # Residual connection from LSTM2 to attention output
        if lstm2.shape[-1] == attention_output.shape[-1]:
            attention_output = Add()([lstm2, attention_output])
            attention_output = LayerNormalization(epsilon=1e-6, name='attention_res_ln')(attention_output)
        
        # Global average pooling
        pooled = GlobalAveragePooling1D(name='global_avg_pool')(attention_output)
        
        # Dense layers with skip connections
        dense1 = self._dense_block(pooled, 64, dropout_rate=self.dropout_rate)
        dense2 = self._dense_block(dense1, 32, dropout_rate=self.dropout_rate)
        
        # Output layer with linear activation
        outputs = Dense(
            1,
            activation='linear',
            kernel_initializer='glorot_normal',
            kernel_regularizer=l2(self.l2_reg),
            name='output_layer'
        )(dense2)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs, name='AttentionLSTM')
        return model
    
    def compile(self, learning_rate: Optional[Union[float, Dict[str, Any]]] = None) -> None:
        """
        Compile the model with improved settings and stability.
        
        Args:
            learning_rate: Optional learning rate to override the one set in __init__
                Can be a float or a dict with schedule configuration.
        """
        if self.model is None:
            self.model = self.build_model()
        
        # Use provided learning rate or the one from init
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        # Handle learning rate schedule if provided as dict
        if isinstance(lr, dict):
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr.get('initial_learning_rate', 1e-3),
                decay_steps=lr.get('decay_steps', 1000),
                decay_rate=lr.get('decay_rate', 0.9),
                staircase=lr.get('staircase', True)
            )
            lr = lr_schedule
        
        # Optimizer with gradient clipping and weight decay
        optimizer = Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,  # Larger epsilon for numerical stability
            clipnorm=self.clip_norm,
            clipvalue=0.5  # Additional gradient clipping
        )
        
        # Huber loss is more robust to outliers than MSE
        loss = tf.keras.losses.Huber(
            delta=1.0,  # Controls the point where the loss changes from L2 to L1
            reduction='auto',
            name='huber_loss'
        )
        
        # Compile model with metrics
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'mae',
                'mse',
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanAbsolutePercentageError(name='mape')
            ]
        )
        
        logger.info("Model compiled successfully with learning rate: %s", 
                   lr.initial_learning_rate if hasattr(lr, 'initial_learning_rate') else lr)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with input validation and error handling.
        
        Args:
            X: Input data of shape (n_samples, seq_len, n_features) or (seq_len, n_features)
            
        Returns:
            Array of predictions with shape (n_samples,) or (n_samples, 1)
            
        Raises:
            ValueError: If model is not trained or input is invalid
            RuntimeError: If prediction fails
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call fit() first.")
                
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype=np.float32)
            
            # Store original shape for later
            original_shape = X.shape
            
            # Ensure input has the right shape
            if len(X.shape) == 2:
                X = np.expand_dims(X, axis=0)
            elif len(X.shape) != 3:
                raise ValueError(f"Input must be 2D or 3D array, got shape {original_shape}")
            
            # Validate input dimensions match model expectations
            if X.shape[1:] != self.input_shape:
                raise ValueError(
                    f"Input shape {X.shape[1:]} does not match model's expected input shape {self.input_shape}"
                )
            
            # Clean input data
            X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, 
                             neginf=np.finfo(np.float32).min)
            
            # Make predictions
            predictions = self.model.predict(X, verbose=0)
            
            # Handle potential NaN/Inf values in predictions
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                logger.warning("NaN or Inf values detected in predictions. Applying corrections.")
                predictions = np.nan_to_num(
                    predictions, 
                    nan=0.0, 
                    posinf=np.finfo(np.float32).max,
                    neginf=np.finfo(np.float32).min
                )
            
            # Return with appropriate shape
            return predictions.squeeze()
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Failed to make predictions: {str(e)}") from e


class CNN_GRU_Attention(BaseModel):
    """CNN-GRU with Attention for volatility forecasting.
    Combines CNN for local pattern extraction with GRU for sequence modeling
    and attention for focusing on important time steps."""
    
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

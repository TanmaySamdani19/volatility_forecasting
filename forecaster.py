"""
Main forecasting pipeline and utilities.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineering, WaveletTransform
from .models import AttentionLSTM, CNN_GRU_Attention, BenchmarkModels
from .training import HyperparameterOptimizer, ModelEvaluator

logger = logging.getLogger(__name__)

class VolatilityForecaster:
    """
    End-to-end volatility forecasting pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the forecaster with configuration.
        
        Args:
            config: Configuration dictionary (see default_config for structure)
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.data_loader = None
        self.feature_engineer = None
        self.wavelet = WaveletTransform()
        self.models = {}
        self.evaluator = ModelEvaluator()
        self.results = {}
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'symbol': '^VIX',
                'start_date': '2012-01-01',
                'end_date': '2023-12-31',
                'test_size': 0.2,
                'target_col': 'Close',
                'sequence_length': 10
            },
            'feature_engineering': {
                'scaler_type': 'standard',
                'use_wavelet': True,
                'wavelet_params': {
                    'wavelet': 'db4',
                    'levels': 4,
                    'threshold_mode': 'soft',
                    'threshold_factor': 0.1
                }
            },
            'models': {
                'AttentionLSTM': {
                    'lstm_units': 128,
                    'dropout_rate': 0.2,
                    'learning_rate': {
                        'schedule': 'exponential_decay',
                        'initial_learning_rate': 0.001,
                        'decay_steps': 1000,
                        'decay_rate': 0.96,
                        'staircase': True
                    },
                    'batch_size': 32,
                    'epochs': 100,
                    'patience': 10
                },
                'CNN_GRU_Attention': {
                    'conv_filters': 64,
                    'gru_units': 64,
                    'dropout_rate': 0.2,
                    'learning_rate': {
                        'schedule': 'exponential_decay',
                        'initial_learning_rate': 0.002,
                        'decay_steps': 1000,
                        'decay_rate': 0.96,
                        'staircase': True
                    },
                    'batch_size': 32,
                    'epochs': 100,
                    'patience': 10
                },
                'ARIMA': {
                    'order': (1, 1, 1)
                },
                'GARCH': {
                    'p': 1,
                    'q': 1
                }
            },
            'optimization': {
                'enabled': False,
                'n_trials': 50,
                'direction': 'minimize',
                'study_name': 'volatility_forecasting_study',
                'storage': None
            },
            'evaluation': {
                'metrics': ['RMSE', 'MAE', 'MAPE', 'R2'],
                'save_plots': True,
                'output_dir': 'results'
            }
        }
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the data.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Initialize data loader
        self.data_loader = DataLoader(
            symbol=self.config['data']['symbol'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date']
        )
        
        # Load and preprocess data
        self.data_loader.preprocess_data()
        
        # Get train/test split
        X_train, X_test, y_train, y_test = self.data_loader.get_train_test_split(
            test_size=self.config['data']['test_size'],
            target_col=self.config['data']['target_col'],
            sequence_length=self.config['data']['sequence_length']
        )
        
        # Initialize feature engineering
        self.feature_engineer = FeatureEngineering(
            scaler_type=self.config['feature_engineering']['scaler_type']
        )
        
        # Ensure all inputs are numpy arrays
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train).reshape(-1, 1)  # Ensure y is 2D (samples, 1)
        y_test = np.asarray(y_test).reshape(-1, 1)    # Ensure y is 2D (samples, 1)
        
        # Ensure shapes match by trimming the longer array
        min_len = min(len(X_train), len(y_train))
        X_train = X_train[:min_len]
        y_train = y_train[:min_len]
        
        min_len_test = min(len(X_test), len(y_test))
        X_test = X_test[:min_len_test]
        y_test = y_test[:min_len_test]
        
        # Debug prints
        print("\nAfter ensuring shapes match:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        # Ensure X has the right shape (samples, timesteps, features)
        if len(X_train.shape) == 2:
            # If 2D, add timestep dimension
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        elif len(X_train.shape) == 1:
            # If 1D, reshape to 3D (samples, 1, 1)
            X_train = X_train.reshape((X_train.shape[0], 1, 1))
            
        if len(X_test.shape) == 2:
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        elif len(X_test.shape) == 1:
            X_test = X_test.reshape((X_test.shape[0], 1, 1))
            
        # Debug prints after reshaping
        print("\nAfter reshaping:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
            
        # Get dimensions
        n_samples, n_timesteps, n_features = X_train.shape
        
        # Reshape to 2D for scaling
        X_train_2d = X_train.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)
        
        # Scale features
        self.feature_engineer.fit_scaler(X_train_2d)
        X_train_scaled = self.feature_engineer.transform_features(X_train_2d)
        X_test_scaled = self.feature_engineer.transform_features(X_test_2d)
        
        # Reshape back to 3D for LSTM
        X_train_scaled = X_train_scaled.reshape(-1, n_timesteps, n_features)
        X_test_scaled = X_test_scaled.reshape(-1, n_timesteps, n_features)
        
        # Scale target if needed
        self.feature_engineer.fit_target_scaler(y_train)
        y_train_scaled = self.feature_engineer.transform_target(y_train)
        y_test_scaled = self.feature_engineer.transform_target(y_test)
        
        # Apply wavelet denoising if enabled
        if self.config['feature_engineering']['use_wavelet']:
            y_train_scaled = self.wavelet.denoise(
                y_train_scaled,
                threshold_mode=self.config['feature_engineering']['wavelet_params']['threshold_mode'],
                threshold_factor=self.config['feature_engineering']['wavelet_params']['threshold_factor']
            )
        
        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray):
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        # Ensure shapes match and y is 1D
        min_len = min(len(X_train), len(y_train))
        X_train = X_train[:min_len]
        y_train = y_train[:min_len].squeeze()  # Ensure y is 1D
        
        min_len_val = min(len(X_val), len(y_val))
        X_val = X_val[:min_len_val]
        y_val = y_val[:min_len_val].squeeze()  # Ensure y is 1D
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        print(f"\nFinal shapes before training:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        # Train deep learning models
        for model_name in ['AttentionLSTM', 'CNN_GRU_Attention']:
            if model_name in self.config['models']:
                logger.info(f"Training {model_name}...")
                
                # Get model class and config
                model_class = globals()[model_name]
                model_config = self.config['models'][model_name].copy()
                
                # Remove training-specific parameters
                batch_size = model_config.pop('batch_size')
                epochs = model_config.pop('epochs')
                patience = model_config.pop('patience')
                
                # Hyperparameter optimization if enabled
                if self.config['optimization']['enabled']:
                    logger.info(f"Optimizing hyperparameters for {model_name}...")
                    optimizer = HyperparameterOptimizer(
                        model_class=model_class,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        fixed_params=model_config
                    )
                    
                    best_params = optimizer.optimize(
                        n_trials=self.config['optimization']['n_trials'],
                        direction=self.config['optimization']['direction'],
                        study_name=f"{model_name}_{self.config['optimization']['study_name']}",
                        storage=self.config['optimization']['storage']
                    )
                    
                    # Update model config with optimized parameters
                    model_config.update(best_params)
                
                # Get learning rate config (default to 0.001 if not specified)
                lr_config = model_config.pop('learning_rate', 0.001)
                
                # Initialize model with remaining config
                model = model_class(input_shape=input_shape, **model_config)
                
                # If learning rate is a dictionary, it contains schedule parameters
                if isinstance(lr_config, dict):
                    if lr_config.get('schedule') == 'exponential_decay':
                        initial_learning_rate = lr_config.get('initial_learning_rate', 0.001)
                        decay_steps = lr_config.get('decay_steps', 1000)
                        decay_rate = lr_config.get('decay_rate', 0.96)
                        staircase = lr_config.get('staircase', True)
                        
                        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=initial_learning_rate,
                            decay_steps=decay_steps,
                            decay_rate=decay_rate,
                            staircase=staircase
                        )
                        learning_rate = lr_schedule
                    else:
                        # Default to fixed learning rate if schedule type is unknown
                        learning_rate = lr_config.get('initial_learning_rate', 0.001)
                else:
                    # It's a fixed learning rate
                    learning_rate = float(lr_config)
                
                # Compile the model with the learning rate (can be float or schedule)
                model.compile(learning_rate=learning_rate)
                
                history = model.train(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience
                )
                
                self.models[model_name] = model
                logger.info(f"{model_name} training completed.")
        
        # Train benchmark models
        if 'ARIMA' in self.config['models'] or 'GARCH' in self.config['models']:
            logger.info("Training benchmark models...")
            
            # Get returns for GARCH model
            returns = self.data_loader.processed_data['Returns'].dropna().values
            
            benchmark = BenchmarkModels()
            
            if 'ARIMA' in self.config['models']:
                order = self.config['models']['ARIMA'].get('order', (1, 1, 1))
                benchmark.fit_arima(self.data_loader.processed_data[self.config['data']['target_col']], order=order)
            
            if 'GARCH' in self.config['models']:
                p = self.config['models']['GARCH'].get('p', 1)
                q = self.config['models']['GARCH'].get('q', 1)
                benchmark.fit_garch(returns, p=p, q=q)
            
            self.models['Benchmark'] = benchmark
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        results = {}
        
        # Ensure output directory exists
        os.makedirs(self.config['evaluation']['output_dir'], exist_ok=True)
        
        # Evaluate deep learning models
        for model_name, model in self.models.items():
            if model_name == 'Benchmark':
                continue
                
            logger.info(f"Evaluating {model_name}...")
            
            # Make predictions
            y_pred_scaled = model.predict(X_test)
            
            # Inverse transform predictions
            y_pred = self.feature_engineer.inverse_transform_target(y_pred_scaled)
            y_true = self.feature_engineer.inverse_transform_target(y_test)
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(y_true, y_pred, model_name)
            results[model_name] = metrics
            
            # Plot predictions
            if self.config['evaluation']['save_plots']:
                plot_path = os.path.join(
                    self.config['evaluation']['output_dir'],
                    f"{model_name.lower()}_predictions.png"
                )
                self.evaluator.plot_predictions(
                    y_true, y_pred, 
                    model_name=model_name,
                    save_path=plot_path
                )
        
        # Evaluate benchmark models
        if 'Benchmark' in self.models:
            benchmark = self.models['Benchmark']
            
            # ARIMA evaluation
            if hasattr(benchmark, 'models') and 'ARIMA' in benchmark.models:
                logger.info("Evaluating ARIMA...")
                
                # ARIMA needs to be evaluated differently as it's univariate
                # This is a simplified evaluation
                try:
                    y_pred_arima = benchmark.predict_arima(steps=len(y_test))
                    y_true_arima = self.data_loader.processed_data[self.config['data']['target_col']].values[-len(y_test):]
                    
                    metrics = self.evaluator.calculate_metrics(
                        y_true_arima, y_pred_arima, 'ARIMA'
                    )
                    results['ARIMA'] = metrics
                except Exception as e:
                    logger.error(f"Error evaluating ARIMA: {e}")
            
            # GARCH evaluation
            if hasattr(benchmark, 'models') and 'GARCH' in benchmark.models:
                logger.info("Evaluating GARCH...")
                
                try:
                    # GARCH predicts volatility, not prices
                    # For comparison, we'll use the last price * (1 + predicted_vol)
                    last_price = self.data_loader.processed_data[self.config['data']['target_col']].iloc[-len(y_test)-1]
                    y_pred_garch = benchmark.predict_garch(steps=len(y_test))
                    y_pred_prices = last_price * (1 + y_pred_garch)
                    
                    y_true_garch = self.data_loader.processed_data[self.config['data']['target_col']].values[-len(y_test):]
                    
                    metrics = self.evaluator.calculate_metrics(
                        y_true_garch, y_pred_prices, 'GARCH'
                    )
                    results['GARCH'] = metrics
                except Exception as e:
                    logger.error(f"Error evaluating GARCH: {e}")
        
        # Save results
        self.results = results
        results_path = os.path.join(
            self.config['evaluation']['output_dir'],
            'evaluation_results.csv'
        )
        
        results_df = pd.DataFrame(results).T
        results_df.to_csv(results_path)
        logger.info(f"Results saved to {results_path}")
        
        return results
    
    def save_models(self, output_dir: str = 'saved_models'):
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        import joblib  # Import joblib here to avoid dependency if not using benchmark models
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                if model_name in ['ARIMA', 'GARCH']:
                    # Save benchmark models using joblib
                    model_path = os.path.join(output_dir, f'{model_name.lower()}.pkl')
                    joblib.dump(model, model_path)
                else:
                    # Save Keras models using the modern .keras format
                    model_path = os.path.join(output_dir, f'{model_name.lower()}.keras')
                    model.save(model_path)
                
                logger.info(f"Saved {model_name} model to {model_path}")
            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")
        
        # Save feature engineering objects
        fe_path = os.path.join(output_dir, 'feature_engineering.joblib')
        joblib.dump(self.feature_engineer, fe_path)
        logger.info(f"Saved feature engineering objects to {fe_path}")
        
        # Save config
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved config to {config_path}")
    
    @classmethod
    def load_models(cls, model_dir: str) -> 'VolatilityForecaster':
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            VolatilityForecaster instance with loaded models
        """
        # Load config
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize forecaster with loaded config
        forecaster = cls(config)
        
        # Load feature engineering objects
        fe_path = os.path.join(model_dir, 'feature_engineering.joblib')
        if os.path.exists(fe_path):
            forecaster.feature_engineer = joblib.load(fe_path)
        
        # Load models
        for model_file in os.listdir(model_dir):
            if model_file.endswith('.h5'):
                model_name = os.path.splitext(model_file)[0].capitalize()
                model_path = os.path.join(model_dir, model_file)
                
                # Determine model class from name
                if 'lstm' in model_name.lower():
                    model = AttentionLSTM(input_shape=(0, 0))  # Dummy shape, will be loaded
                elif 'cnn' in model_name.lower() or 'gru' in model_name.lower():
                    model = CNN_GRU_Attention(input_shape=(0, 0))  # Dummy shape
                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                # Load model weights
                model.model = tf.keras.models.load_model(model_path)
                forecaster.models[model_name] = model
                logger.info(f"Loaded {model_name} model from {model_path}")
            
            elif model_file.endswith('.joblib') and 'feature_engineering' not in model_file:
                # Load benchmark models
                model_name = os.path.splitext(model_file)[0].upper()
                model_path = os.path.join(model_dir, model_file)
                
                if 'Benchmark' not in forecaster.models:
                    forecaster.models['Benchmark'] = BenchmarkModels()
                
                if model_name in ['ARIMA', 'GARCH']:
                    model = joblib.load(model_path)
                    forecaster.models['Benchmark'].models[model_name] = model
                    logger.info(f"Loaded {model_name} model from {model_path}")
        
        return forecaster
    
    def forecast(self, steps: int = 1) -> Dict[str, np.ndarray]:
        """
        Generate forecasts using all trained models.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Dictionary of model forecasts
        """
        if not self.models:
            raise ValueError("No models available for forecasting. Train models first.")
        
        forecasts = {}
        
        # Get the most recent sequence for prediction
        if self.data_loader is None or self.feature_engineer is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get the most recent sequence
        data = self.data_loader.processed_data
        target_col = self.config['data']['target_col']
        sequence_length = self.config['data']['sequence_length']
        
        # Prepare features for the last sequence
        X = data.drop(columns=[target_col])
        
        # Scale features
        X_scaled = self.feature_engineer.transform_features(
            X.values[-sequence_length:].reshape(1, sequence_length, -1)
        )
        
        # Generate forecasts for each model
        for model_name, model in self.models.items():
            if model_name == 'Benchmark':
                continue
                
            # For deep learning models
            y_pred_scaled = model.predict(X_scaled)
            y_pred = self.feature_engineer.inverse_transform_target(y_pred_scaled)
            forecasts[model_name] = y_pred
        
        # Handle benchmark models
        if 'Benchmark' in self.models:
            benchmark = self.models['Benchmark']
            
            # ARIMA forecast
            if hasattr(benchmark, 'models') and 'ARIMA' in benchmark.models:
                try:
                    arima_forecast = benchmark.predict_arima(steps=steps)
                    forecasts['ARIMA'] = arima_forecast
                except Exception as e:
                    logger.error(f"Error generating ARIMA forecast: {e}")
            
            # GARCH forecast
            if hasattr(benchmark, 'models') and 'GARCH' in benchmark.models:
                try:
                    garch_forecast = benchmark.predict_garch(steps=steps)
                    # Convert to price forecast
                    last_price = data[target_col].iloc[-1]
                    garch_price_forecast = last_price * (1 + garch_forecast)
                    forecasts['GARCH'] = garch_price_forecast
                except Exception as e:
                    logger.error(f"Error generating GARCH forecast: {e}")
        
        return forecasts

def run_analysis(config: Optional[Dict[str, Any]] = None):
    """
    Run the complete volatility forecasting pipeline.
    
    Args:
        config: Optional configuration dictionary
    """
    # Initialize forecaster
    forecaster = VolatilityForecaster(config)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test = forecaster.load_data()
    
    # Train models
    logger.info("Training models...")
    forecaster.train_models(X_train, y_train, X_test, y_test)
    
    # Evaluate models
    logger.info("Evaluating models...")
    results = forecaster.evaluate_models(X_test, y_test)
    
    # Save models
    logger.info("Saving models...")
    output_dir = forecaster.config['evaluation']['output_dir']
    forecaster.save_models(os.path.join(output_dir, 'saved_models'))
    
    # Generate forecasts
    logger.info("Generating forecasts...")
    forecasts = forecaster.forecast(steps=forecaster.config['data']['sequence_length'])
    
    logger.info("Analysis complete!")
    return {
        'forecaster': forecaster,
        'results': results,
        'forecasts': forecasts
    }

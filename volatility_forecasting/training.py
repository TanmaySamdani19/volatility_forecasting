"""
Model training and evaluation utilities.
"""
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
import logging
import joblib
import os

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna.
    """
    
    def __init__(self, model_class, X_train, y_train, X_val, y_val, 
                 fixed_params: Optional[Dict] = None):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            model_class: Model class to optimize (e.g., AttentionLSTM)
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            fixed_params: Fixed parameters to pass to the model
        """
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.fixed_params = fixed_params or {}
    
    def objective(self, trial):
        """Objective function for Optuna optimization."""
        # Define hyperparameter search space
        params = {
            'lstm_units': trial.suggest_int('lstm_units', 32, 256, step=32),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        }
        
        # Add model-specific parameters
        if hasattr(self.model_class, '__name__') and 'CNN' in self.model_class.__name__:
            params.update({
                'conv_filters': trial.suggest_int('conv_filters', 32, 128, step=32),
                'gru_units': trial.suggest_int('gru_units', 32, 128, step=32)
            })
        
        # Merge with fixed parameters
        params.update(self.fixed_params)
        
        try:
            # Initialize and train model
            input_shape = (self.X_train.shape[1], self.X_train.shape[2])
            model = self.model_class(input_shape=input_shape, **params)
            
            # Compile and train
            model.compile(learning_rate=params['learning_rate'])
            
            history = model.model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=100,  # Will use early stopping
                batch_size=params['batch_size'],
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )
            
            # Return validation loss
            return min(history.history['val_loss'])
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            return float('inf')
    
    def optimize(self, n_trials: int = 50, direction: str = 'minimize', 
                study_name: Optional[str] = None, storage: Optional[str] = None) -> dict:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of optimization trials
            direction: 'minimize' or 'maximize'
            study_name: Name for the study (for storage)
            storage: Database URL for storing study results
            
        Returns:
            Dictionary of best parameters
        """
        study = optuna.create_study(
            direction=direction,
            study_name=study_name or f"{self.model_class.__name__}_study",
            storage=storage,
            load_if_exists=True
        )
        
        study.optimize(self.objective, n_trials=n_trials)
        
        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study.best_params


class ModelEvaluator:
    """
    Model evaluation and comparison utilities.
    """
    
    def __init__(self):
        self.results = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'R2': r2_score(y_true, y_pred)
        }
        
        # Additional financial metrics
        residuals = y_true - y_pred
        metrics.update({
            'Mean_Residual': np.mean(residuals),
            'Std_Residual': np.std(residuals),
            'Sharpe_Ratio': np.mean(residuals) / (np.std(residuals) + 1e-8),
            'Max_Drawdown': self._calculate_max_drawdown(y_true, y_pred)
        })
        
        self.results[model_name] = metrics
        return metrics
    
    def _calculate_max_drawdown(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate maximum drawdown of the prediction error."""
        error = y_true - y_pred
        cumulative = np.cumsum(error)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1e-8)
        return np.max(drawdown)
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model_name: str, save_path: Optional[str] = None):
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model (for title)
            save_path: Path to save the plot (optional)
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 6))
        
        # Time series plot
        plt.subplot(1, 2, 1)
        plt.plot(y_true, label='Actual', alpha=0.7, linewidth=2)
        plt.plot(y_pred, label='Predicted', alpha=0.7, linewidth=1.5, linestyle='--')
        plt.title(f'{model_name} - Predictions vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2)
        plt.title(f'{model_name} - Scatter Plot')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def comparison_table(self) -> pd.DataFrame:
        """
        Create a comparison table of all evaluated models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            logger.warning("No results available for comparison")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results).T
        
        # Reorder columns for better readability
        columns_order = [
            'RMSE', 'MAE', 'MAPE', 'R2', 
            'Mean_Residual', 'Std_Residual',
            'Sharpe_Ratio', 'Max_Drawdown'
        ]
        
        # Only keep columns that exist in the results
        columns_order = [col for col in columns_order if col in df.columns]
        
        return df[columns_order].round(4)
    
    def save_results(self, filepath: str):
        """
        Save evaluation results to a file.
        
        Args:
            filepath: Path to save the results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if filepath.endswith('.pkl') or filepath.endswith('.joblib'):
            joblib.dump(self.results, filepath)
        else:
            # Default to CSV
            df = self.comparison_table()
            df.to_csv(filepath)
        
        logger.info(f"Results saved to {filepath}")
    
    @classmethod
    def load_results(cls, filepath: str) -> 'ModelEvaluator':
        """
        Load evaluation results from a file.
        
        Args:
            filepath: Path to the saved results
            
        Returns:
            ModelEvaluator instance with loaded results
        """
        evaluator = cls()
        
        if filepath.endswith('.pkl') or filepath.endswith('.joblib'):
            evaluator.results = joblib.load(filepath)
        else:
            # Try to load as CSV
            df = pd.read_csv(filepath, index_col=0)
            evaluator.results = df.to_dict('index')
        
        return evaluator

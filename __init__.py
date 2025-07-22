"""
Financial Market Volatility Forecasting System
-------------------------------------------
A hybrid deep learning approach for volatility forecasting using wavelet transforms
and attention mechanisms.
"""

__version__ = "0.1.0"

# Import core components
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineering, WaveletTransform
from .models import AttentionLSTM, CNN_GRU_Attention, BenchmarkModels
from .training import HyperparameterOptimizer, ModelEvaluator
from .forecaster import VolatilityForecaster, run_analysis

__all__ = [
    'DataLoader',
    'FeatureEngineering',
    'WaveletTransform',
    'AttentionLSTM',
    'CNN_GRU_Attention',
    'BenchmarkModels',
    'HyperparameterOptimizer',
    'ModelEvaluator',
    'VolatilityForecaster',
    'run_analysis'
]

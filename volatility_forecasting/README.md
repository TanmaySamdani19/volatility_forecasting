# Financial Market Volatility Forecasting

A comprehensive framework for forecasting financial market volatility using deep learning and traditional statistical models.

## Features

- Multi-resolution wavelet analysis for feature extraction
- Hybrid deep learning models (LSTM + Attention, CNN-GRU + Attention)
- Traditional statistical models (ARIMA, GARCH) for benchmarking
- Hyperparameter optimization with Optuna
- Comprehensive model evaluation and visualization
- End-to-end pipeline for data loading, preprocessing, training, and evaluation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/volatility-forecasting.git
   cd volatility-forecasting
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Note: For TA-Lib, you might need to install system dependencies first. On Ubuntu/Debian:
   ```bash
   sudo apt-get install python3-tk
   wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   tar -xzf ta-lib-0.4.0-src.tar.gz
   cd ta-lib/
   ./configure --prefix=/usr
   make
   sudo make install
   pip install TA-Lib
   ```

## Usage

### Basic Usage

```python
from volatility_forecasting import VolatilityForecaster, run_analysis

# Run with default configuration
results = run_analysis()

# Or customize the configuration
config = {
    'data': {
        'symbol': '^GSPC',  # S&P 500
        'start_date': '2010-01-01',
        'end_date': '2023-12-31',
        'sequence_length': 20
    },
    'models': {
        'AttentionLSTM': {
            'lstm_units': 128,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        },
        'ARIMA': {
            'order': (1, 1, 1)
        }
    }
}

results = run_analysis(config)
```

### Jupyter Notebook Example

See the `examples/volatility_forecasting_demo.ipynb` notebook for a complete walkthrough of the library's capabilities.

## Project Structure

```
volatility_forecasting/
├── __init__.py              # Package initialization
├── data_loader.py           # Data loading and preprocessing
├── feature_engineering.py   # Feature engineering and wavelet transforms
├── models.py                # Model architectures
├── training.py              # Training and evaluation utilities
├── forecaster.py            # Main forecasting pipeline
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Model Architectures

### AttentionLSTM
- Multi-head self-attention mechanism
- Stacked LSTM layers with residual connections
- Layer normalization and dropout for regularization

### CNN-GRU with Attention
- 1D CNN for local pattern extraction
- GRU for sequential modeling
- Multi-head attention for capturing long-range dependencies

### Benchmark Models
- ARIMA: AutoRegressive Integrated Moving Average
- GARCH: Generalized Autoregressive Conditional Heteroskedasticity

## Evaluation Metrics

- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error
- R²: Coefficient of Determination
- Sharpe Ratio: Risk-adjusted returns
- Maximum Drawdown: Maximum loss from peak to trough

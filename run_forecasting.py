"""
Example script to run the volatility forecasting pipeline.
"""
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from volatility_forecasting import VolatilityForecaster, run_analysis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the volatility forecasting pipeline."""
    # Configuration
    config = {
        'data': {
            'symbol': '^VIX',  # CBOE Volatility Index
            'start_date': '2010-01-01',  # 5 years of training data
            'end_date': '2023-12-31',
            'test_size': 0.3,  # 20% test set
            'target_col': 'Close',
            'sequence_length': 20  # Number of time steps to use for prediction
        },
        'models': {
            'AttentionLSTM': {
                'lstm_units': 128,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,  # Reduced for faster execution
                'patience': 10
            },
            'ARIMA': {
                'order': (1, 1, 1)
            }
        },
        'evaluation': {
            'output_dir': 'results',
            'save_plots': True
        }
    }

    # Create output directory if it doesn't exist
    os.makedirs(config['evaluation']['output_dir'], exist_ok=True)

    # Run the complete analysis pipeline
    logger.info("Starting volatility forecasting pipeline...")
    results = run_analysis(config)
    
    # Print model comparison
    if hasattr(results['forecaster'].evaluator, 'comparison_table'):
        comparison = results['forecaster'].evaluator.comparison_table()
        print("\nModel Comparison:")
        print(comparison.to_string())
        
        # Save comparison to file
        comparison_file = os.path.join(
            config['evaluation']['output_dir'], 
            'model_comparison.csv'
        )
        comparison.to_csv(comparison_file)
        logger.info(f"Model comparison saved to {comparison_file}")
    
    # Generate and display forecasts
    logger.info("Generating forecasts...")
    forecasts = results['forecaster'].forecast(steps=5)  # 5-day forecast
    
    print("\nForecasts:")
    for model, values in forecasts.items():
        print(f"{model}: {values}")
    
    logger.info("Analysis complete! Check the 'results' directory for outputs.")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create results directory if it doesn't exist
os.makedirs('results/figures', exist_ok=True)

# Model comparison data
model_data = {
    'Model': ['AttentionLSTM', 'ARIMA'],
    'RMSE': [8.3023, 13.1412],
    'MAE': [5.7080, 10.0069],
    'MAPE': [28.4897, 38.4557],
    'R2': [0.0526, -1.3734],
    'Sharpe_Ratio': [-0.0354, 1.1719],
    'Max_Drawdown': [1.2369, 0.4125]
}

df = pd.DataFrame(model_data)

# Set up the figure and axes
plt.figure(figsize=(15, 10))

# Plot 1: RMSE and MAE comparison
plt.subplot(2, 2, 1)
x = np.arange(len(df['Model']))
width = 0.35

plt.bar(x - width/2, df['RMSE'], width, label='RMSE')
plt.bar(x + width/2, df['MAE'], width, label='MAE')
plt.xlabel('Model')
plt.title('RMSE and MAE Comparison')
plt.xticks(x, df['Model'])
plt.legend()

# Plot 2: MAPE comparison
plt.subplot(2, 2, 2)
plt.bar(df['Model'], df['MAPE'], color='green')
plt.xlabel('Model')
plt.title('Mean Absolute Percentage Error (MAPE)')

# Plot 3: R2 Score
plt.subplot(2, 2, 3)
plt.bar(df['Model'], df['R2'], color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.xlabel('Model')
plt.title('R² Score (Higher is better)')

# Plot 4: Sharpe Ratio and Max Drawdown
plt.subplot(2, 2, 4)
width = 0.35
plt.bar(x - width/2, df['Sharpe_Ratio'], width, label='Sharpe Ratio')
plt.bar(x + width/2, df['Max_Drawdown'], width, label='Max Drawdown')
plt.xlabel('Model')
plt.title('Risk Metrics')
plt.xticks(x, df['Model'])
plt.legend()

plt.tight_layout()
plt.savefig('results/figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a radar chart for better comparison
metrics = ['RMSE', 'MAE', 'MAPE', 'R2', 'Sharpe_Ratio', 'Max_Drawdown']
labels = metrics.copy()
num_vars = len(metrics)

# Normalize the data for radar chart (lower is better for most metrics except R2 and Sharpe Ratio)
df_radar = df.copy()
for col in metrics:
    if col in ['R2', 'Sharpe_Ratio']:
        # For these, higher is better
        df_radar[col] = (df_radar[col] - df_radar[col].min()) / (df_radar[col].max() - df_radar[col].min() + 1e-10)
    else:
        # For these, lower is better, so we invert the scale
        df_radar[col] = 1 - ((df_radar[col] - df_radar[col].min()) / (df_radar[col].max() - df_radar[col].min() + 1e-10))

# Compute angle for each axis
angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
angles += angles[:1]  # Close the plot

# Initialize the radar plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Add one line per model
for idx, model in enumerate(df_radar['Model']):
    values = df_radar.loc[df_radar['Model'] == model, metrics].values[0].tolist()
    values += values[:1]  # Close the plot
    
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=model)
    ax.fill(angles, values, alpha=0.1)

# Add labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticklabels([])  # Remove radial labels
ax.set_title('Model Comparison (Normalized Metrics)', size=15, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig('results/figures/radar_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations saved to 'results/figures/' directory")

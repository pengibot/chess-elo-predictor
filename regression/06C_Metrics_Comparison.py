import matplotlib.pyplot as plt
import numpy as np

# Example data
metrics = ['MAE', 'RMSE']
elastic_net = [257, 318]
random_forest = [265, 327]
svr = [258, 321]

# Combine data
data = np.array([elastic_net, random_forest, svr]).T  # Transpose for grouping
models = ['Elastic Net CV', 'Random Forest', 'Support Vector']

x = np.arange(len(metrics))  # Metric indices
width = 0.25  # Width of each bar

# Plot grouped bars
fig, ax = plt.subplots(figsize=(10, 6))
for i, model in enumerate(models):
    bars = ax.bar(x + i * width, data[:, i], width-0.02, label=model)
    # Add value labels on bars
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
            bar.get_height() + 5,  # Y position (slightly above the bar)
            f'{bar.get_height()}',  # The value to display
            ha='center', va='bottom', fontsize=10  # Text alignment
        )

# Add labels, title, and legend
ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Values', fontsize=12)
ax.set_title('Comparison of Testing Metrics Across Models', fontsize=14)
ax.set_xticks(x + width)  # Center labels
ax.set_xticklabels(metrics, fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Show plot
plt.tight_layout()
plt.show()

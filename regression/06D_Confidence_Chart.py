import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example data: Coefficients, p-values, and confidence intervals
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

pickls_directory = Path("Data/Pickls")
testing_data_filename = r"test_data_df"

# Load data
with open(pickls_directory / '{}.pkl'.format(testing_data_filename), 'rb') as file:
    testing_data_frame = pickle.load(file)

# Separate features and target variable for each dataset
X_test = testing_data_frame
y_test = testing_data_frame['ELO']

# List of fields to keep
fields_to_keep = ['black_blunder_count',
                  'black_inaccuracy_count',
                  'black_mistake_count',
                  'black_double_pawns',
                  'black_first_knight_on_edge',
                  'black_isolated_pawns',
                  'black_king_castle',
                  'black_moves_before_castling',
                  'black_piece_activity',
                  'black_queen_castle',
                  'black_queen_moved_at',
                  'black_total_knights_on_edge',
                  'black_tripled_pawns']  # Replace with your actual column names

# Keep only these fields in X_test
X_test = X_test[fields_to_keep]

# Scale features using training set
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Assuming X_train is your feature matrix and y_train is your target variable
X_test_scaled = sm.add_constant(X_test_scaled)  # Add intercept if needed
model = sm.OLS(y_test, X_test_scaled).fit()

# Model setup and hyperparameter tuning
elcv_model = ElasticNetCV(alphas=[0.1, 1.0, 10.0], max_iter=1000, cv=5, l1_ratio=0.1)
elcv_model.fit(X_test_scaled, y_test)

# Extract features, coefficients, p-values, and confidence intervals
features = X_test.columns.tolist()  # Feature names
features = ['Intercept'] + features  # Add 'Intercept' to the list of features
coefficients = model.params.tolist()  # Coefficients
p_values = model.pvalues.tolist()  # P-values
conf_intervals = model.conf_int().values.tolist()  # Confidence intervals

features = features[1:]
coefficients = coefficients[1:]
p_values = p_values[1:]
conf_intervals = conf_intervals[1:]

# Recalculate positions and error bars
x_positions = np.arange(len(features))
lower_bounds = [coef - conf[0] for coef, conf in zip(coefficients, conf_intervals)]
upper_bounds = [conf[1] - coef for coef, conf in zip(coefficients, conf_intervals)]

print(f"Number of x_positions: {len(x_positions)}")
print(f"Number of coefficients: {len(coefficients)}")

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(x_positions, coefficients, yerr=np.array([lower_bounds, upper_bounds]), fmt='o', capsize=5, label='Coefficient (95% CI)')
plt.axhline(0, color='red', linestyle='--', linewidth=1, label='No Effect (Zero Coefficient)')


# Annotate p-values
for i, p in enumerate(p_values):
    plt.text(i, coefficients[i], f"p={p:.3f}", fontsize=10, ha='center', va='bottom')

# Add labels and legend
plt.xticks(x_positions, features, fontsize=10, rotation=90)  # Rotate feature names
plt.xlabel('Features', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Coefficient Significance with 95% Confidence Intervals', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# Show plot
plt.tight_layout()
plt.show()

# Add 'Intercept' to fields_to_keep if a constant column was added
if X_test_scaled.shape[1] == len(fields_to_keep) + 1:
    fields_to_keep = ["Intercept"] + fields_to_keep

# Assuming X_test_scaled is your scaled feature matrix (excluding target variable)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=fields_to_keep)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_test_scaled_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_test_scaled_df.values, i) for i in range(X_test_scaled_df.shape[1])]

# Display VIF
print(vif_data)

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Define the bucket categories
def categorize_elo(elo):
    if elo < 1350:
        return 'Low'
    elif 1350 <= elo < 1650:
        return 'Medium'
    else:
        return 'High'


# Load the data (replace 'file_path' with your actual file path)
# Assuming the file is a text file where each line is formatted like "Actual: 1674, Predicted: 1349.07"
pickls_directory = Path('Data/Pickls')
ratings_file = r"ratings.txt"
data = []

with open(pickls_directory / ratings_file, 'r') as f:
    for line in f:
        parts = line.strip().split(", ")
        actual = float(parts[0].split(": ")[1])
        predicted = float(parts[1].split(": ")[1])
        data.append({"Actual": actual, "Predicted": predicted})

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Plot 1: Scatter plot with a reference line
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Actual", y="Predicted", data=df, alpha=0.6)
plt.plot([df["Actual"].min(), df["Actual"].max()], [df["Actual"].min(), df["Actual"].max()],
         color="red", linestyle="--", label="Perfect Prediction")
plt.title("Actual vs Predicted Elo Ratings")
plt.xlabel("Actual Elo Ratings")
plt.ylabel("Predicted Elo Ratings")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Residual plot
df['Residual'] = df['Actual'] - df['Predicted']

plt.figure(figsize=(10, 6))
sns.histplot(df['Residual'], kde=True, bins=30, color="blue", alpha=0.7)
plt.axvline(0, color="red", linestyle="--", label="Zero Error")
plt.title("Residual Distribution (Actual - Predicted)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# Categorize the actual and predicted Elo values
df['Actual Category'] = df['Actual'].apply(categorize_elo)
df['Predicted Category'] = df['Predicted'].apply(categorize_elo)

# Create the confusion matrix
categories = ['Low', 'Medium', 'High']
conf_matrix = confusion_matrix(df['Actual Category'], df['Predicted Category'], labels=categories)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix for Elo Predictions')
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.show()
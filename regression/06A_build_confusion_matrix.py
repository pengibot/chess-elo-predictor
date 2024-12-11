# Import required libraries
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Function to categorize Elo ratings
def categorize_elo(elo):
    if elo < 1350:
        return 'Low'
    elif 1350 <= elo < 1650:
        return 'Mid'
    else:
        return 'High'


def main():

    print("Started Generating Confusion Matrix for Models...")

    input_directory = Path("Data/Pickls")
    model_directory = Path("Data/Models")
    input_filename = r"test_data_df"

    # Load the combined dataframe
    with open(input_directory / f'{input_filename}.pkl', 'rb') as file:
        data_frame = pickle.load(file)

    # Split the data into features (X) and target (y)
    y_test = data_frame['ELO']
    X_test = data_frame.drop(columns=['ELO', 'evals'])  # Drop the ELO and Stockfish columns for the features

    print("Scaling...")

    # Scale the features (ensure the scaler used during training is used again)
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    model_names = ["elastic_net_model", "random_forest_model", "support_vector_model"]

    for model_name in model_names:

        # Load the model
        with open(model_directory / f'{model_name}.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        print(f"{model_name} loaded successfully.")

        print("Predicting...")

        # Predict on the test set
        y_pred = model.predict(X_test_scaled)

        # Calculate MAE, MSE, RMSE, and R2
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Display the metrics
        print(f"{model_name} Mean Absolute Error (MAE): {mae}")
        print(f"{model_name} Mean Squared Error (MSE): {mse}")
        print(f"{model_name} Root Mean Squared Error (RMSE): {rmse}")
        print(f"{model_name} R-squared (R2): {r2}")

        # Convert continuous predictions and actual values into categories
        y_pred_categories = [categorize_elo(pred) for pred in y_pred]
        y_test_categories = [categorize_elo(actual) for actual in y_test]

        print(f"Building Confusion Matrix for {model_name}...")

        # Create confusion matrix
        conf_matrix = confusion_matrix(y_test_categories, y_pred_categories, labels=['Low', 'Mid', 'High'])

        # Normalize confusion matrix values by column
        col_normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=0)[np.newaxis, :]

        # Define custom labels
        class_labels = ['Low', 'Mid', 'High']

        # Plot confusion matrix with column-normalized colors and custom labels
        plt.figure(figsize=(8, 6))
        sns.heatmap(col_normalized_conf_matrix, annot=conf_matrix, fmt='d', cmap='Blues', cbar=True,
                    linewidths=0.5, linecolor='black',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        print(f"Finished Generating Confusion Matrix for {model_name}.")


if __name__ == "__main__":
    main()

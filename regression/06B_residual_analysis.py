import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import numpy as np


def main():
    print("Started Generating Graphs...")

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
    X_test_scaled = scaler.fit_transform(X_test)  # You might want to load the scaler if it was saved during training

    model_names = ["elastic_net_model", "random_forest_model", "support_vector_model"]

    for model_name in model_names:

        # Load the trained Random Forest model
        with open(model_directory / '{}.pkl'.format(model_name), 'rb') as model_file:
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

        residuals = y_test - y_pred  # Calculate residuals

        # Plot the actual vs predicted ELO ratings
        # Plot residuals vs. fitted values
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, edgecolor=None)
        sns.kdeplot(x=y_pred, y=residuals, levels=10, color='blue', linewidths=1)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
        plt.xlabel('Fitted Values (Predicted)')
        plt.ylabel('Residuals')
        plt.title('Residuals vs. Fitted Values for {}'.format(model_name))
        plt.show()

        # Create a histogram of residuals
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.title('Histogram of Residuals for {}'.format(model_name), fontsize=14)
        plt.xlabel('Residuals', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(0, color='red', linestyle='dashed', linewidth=1, label='Zero Residual')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        print("Finished Generating Graphs.")


if __name__ == "__main__":
    main()

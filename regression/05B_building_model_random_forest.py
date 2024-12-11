import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def main():
    print("Started Building Model for Random Forest Regressor...")

    # Define file paths
    pickls_directory = Path("Data/Pickls")
    models_directory = Path("Data/Models")
    training_data_filename = r"training_data_df"
    validation_data_filename = "validation_data_df.pkl"
    testing_data_filename = r"test_data_df"
    model_filename = r"random_forest_model.pkl"

    # Load data
    with open(pickls_directory / '{}.pkl'.format(training_data_filename), 'rb') as file:
        training_data_frame = pickle.load(file)

    with open(pickls_directory / validation_data_filename, 'rb') as f:
        validation_data_frame = pickle.load(f)

    with open(pickls_directory / '{}.pkl'.format(testing_data_filename), 'rb') as file:
        testing_data_frame = pickle.load(file)

    # Separate features and target variable for each dataset
    X_train = training_data_frame.drop(columns=['ELO', 'evals'])
    y_train = training_data_frame['ELO']

    X_val = validation_data_frame.drop(columns=['ELO', 'evals'])
    y_val = validation_data_frame['ELO']

    X_test = testing_data_frame.drop(columns=['ELO', 'evals'])
    y_test = testing_data_frame['ELO']

    # Scale features using training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Model setup and hyperparameter tuning
    rfr_model = RandomForestRegressor(n_estimators=500,
                                      min_samples_split=10,
                                      min_samples_leaf=1,
                                      max_features='sqrt',
                                      max_depth=100)
    rfr_model.fit(X_train_scaled, y_train)

    # Validate the model on the validation set
    y_val_pred = rfr_model.predict(X_val_scaled)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"Validation Results - MAE: {val_mae}, MSE: {val_mse}, RMSE: {val_rmse}, R2: {val_r2}")

    # Final evaluation on the test set
    y_test_pred = rfr_model.predict(X_test_scaled)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Test Results - MAE: {test_mae}, MSE: {test_mse}, RMSE: {test_rmse}, R2: {test_r2}")

    # Save the final model
    models_directory.mkdir(parents=True, exist_ok=True)
    with open(models_directory / model_filename, 'wb') as model_file:
        pickle.dump(rfr_model, model_file)
    print(f"Model saved to {(models_directory / model_filename).resolve()}")

    print("Finished Building Model for Random Forest Regressor...")


if __name__ == "__main__":
    main()

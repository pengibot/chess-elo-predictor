import os
import re
from pathlib import Path
import numpy as np
from keras.utils import pad_sequences, load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def extract_move_number(filename):
    move_number = re.search(r'chessboard_move_(\d+)_', filename)
    return int(move_number.group(1)) if move_number else None


def extract_ratings(filename):
    parts = filename.split('_')
    white_rating = int(parts[3][1:])
    black_rating = int(parts[4][1:])
    return white_rating, black_rating


def load_images_and_labels_from_game(game_path):
    filenames = os.listdir(game_path)
    filenames = sorted(filenames, key=extract_move_number)

    images = []
    white_ratings = []
    black_ratings = []

    for filename in filenames:
        img = load_img(os.path.join(game_path, filename), target_size=(8, 8))
        img = img_to_array(img) / 255.0
        white_rating, black_rating = extract_ratings(filename)
        images.append(img)
        white_ratings.append(white_rating)
        black_ratings.append(black_rating)

    return np.array(images), np.array(white_ratings), np.array(black_ratings)


def load_images_and_labels(directory):
    all_images = []
    all_white_ratings = []
    all_black_ratings = []
    max_moves = 0

    for game_folder in os.listdir(directory):
        game_path = os.path.join(directory, game_folder)
        if os.path.isdir(game_path):
            images, white_ratings, black_ratings = load_images_and_labels_from_game(game_path)

            all_images.append(images)
            all_white_ratings.append(white_ratings[0])
            all_black_ratings.append(black_ratings[0])
            max_moves = max(len(images), max_moves)

    all_images_padded = pad_sequences(all_images, maxlen=max_moves, padding='post', dtype='float32')

    return np.array(all_images_padded), np.array(all_white_ratings), np.array(all_black_ratings), max_moves


# Function to categorize Elo ratings
def categorize_elo(elo):
    if elo < 1350:
        return 'Low'
    elif 1350 <= elo < 1650:
        return 'Mid'
    else:
        return 'High'


def main():

    print("Started Analyzing Best Colour Model")

    models_directory = Path('Data/Models')
    pickls_directory = Path('Data/Pickls')
    input_file = r'best_model_combined_mae.keras'
    output_file = r"ratings.txt"

    # Load the saved model
    model = load_model(models_directory / input_file)

    games_directory = Path('Data/Games')
    image_files, white_ratings, black_ratings, max_moves = load_images_and_labels(games_directory)

    X_train_val, X_test, y_train_val_white, y_test_white, y_train_val_black, y_test_black = train_test_split(
        image_files, white_ratings, black_ratings, test_size=0.3
    )

    plt.hist(white_ratings, bins=20, alpha=0.5, label='White Ratings')
    plt.hist(black_ratings, bins=20, alpha=0.5, label='Black Ratings')
    plt.xlabel('Elo Rating')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

    # Generate predictions using the loaded model
    pred_white, pred_black = model.predict(X_test)

    # Flatten the predictions (since predict() might return them as arrays)
    pred_white = pred_white.flatten()
    pred_black = pred_black.flatten()

    # Print both actual and predicted Elo ratings for white and black players
    with open(pickls_directory / output_file, "w") as file:
        for actual, predicted in zip(y_test_white, pred_white):
            file.write(f"Actual: {actual}, Predicted: {predicted:.2f}\n")

        for actual, predicted in zip(y_test_black, pred_black):
            file.write(f"Actual: {actual}, Predicted: {predicted:.2f}\n")

    # Convert continuous predictions to categories
    pred_white_categories = [categorize_elo(pred) for pred in pred_white.flatten()]
    pred_black_categories = [categorize_elo(pred) for pred in pred_black.flatten()]

    # Convert actual Elo ratings to categories
    actual_white_categories = [categorize_elo(elo) for elo in y_test_white]
    actual_black_categories = [categorize_elo(elo) for elo in y_test_black]

    # Create confusion matrices for both white and black Elo predictions
    conf_matrix_white = confusion_matrix(actual_white_categories, pred_white_categories,
                                         labels=['Low', 'Mid', 'High'])
    conf_matrix_black = confusion_matrix(actual_black_categories, pred_black_categories,
                                         labels=['Low', 'Mid', 'High'])

    # Convert confusion matrices to DataFrame for better visualization
    conf_matrix_white_df = pd.DataFrame(conf_matrix_white, index=['Low', 'Mid', 'High'],
                                        columns=['Low', 'Mid', 'High'])
    conf_matrix_black_df = pd.DataFrame(conf_matrix_black, index=['Low', 'Mid', 'High'],
                                        columns=['Low', 'Mid', 'High'])

    # Plot confusion matrix for white ratings
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_white_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for White Elo Ratings')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Plot confusion matrix for black ratings
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_black_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Black Elo Ratings')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Calculate the difference between actual and predicted
    white_differences = np.array(pred_white) - np.array(y_test_white)

    plt.figure(figsize=(12, 8))
    plt.scatter(y_test_white, pred_white, c=white_differences, cmap='coolwarm', edgecolor='k')
    plt.colorbar(label='Prediction Error')
    plt.plot([min(y_test_white), max(y_test_white)], [min(y_test_white), max(y_test_white)], 'k--', lw=2)
    plt.xlabel('Actual ELO')
    plt.ylabel('Predicted ELO')
    plt.title('Actual vs. Predicted White ELO Ratings')
    plt.show()

    # Calculate the difference between actual and predicted
    black_differences = np.array(pred_black) - np.array(y_test_black)

    plt.figure(figsize=(12, 8))
    plt.scatter(y_test_black, pred_black, c=black_differences, cmap='coolwarm', edgecolor='k')
    plt.colorbar(label='Prediction Error')
    plt.plot([min(y_test_black), max(y_test_black)], [min(y_test_black), max(y_test_black)], 'k--', lw=2)
    plt.xlabel('Actual ELO')
    plt.ylabel('Predicted ELO')
    plt.title('Actual vs. Predicted Black ELO Ratings')
    plt.show()

    print("Finished Analyzing Best Colour Model")

    # Calculate metrics for White player Elo predictions
    mae_white = mean_absolute_error(y_test_white, pred_white)
    mse_white = mean_squared_error(y_test_white, pred_white)
    rmse_white = np.sqrt(mse_white)
    r2_white = r2_score(y_test_white, pred_white)

    # Calculate metrics for Black player Elo predictions
    mae_black = mean_absolute_error(y_test_black, pred_black)
    mse_black = mean_squared_error(y_test_black, pred_black)
    rmse_black = np.sqrt(mse_black)
    r2_black = r2_score(y_test_black, pred_black)

    # Calculate combined metrics by averaging white and black metrics
    mae_combined = (mae_white + mae_black) / 2
    mse_combined = (mse_white + mse_black) / 2
    rmse_combined = (rmse_white + rmse_black) / 2
    r2_combined = (r2_white + r2_black) / 2

    # Print combined metrics
    print("\nCombined Performance Metrics for Elo Predictions:")
    print(f"Mean Absolute Error (MAE): {mae_combined:.2f}")
    print(f"Mean Squared Error (MSE): {mse_combined:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_combined:.2f}")
    print(f"R² Score: {r2_combined:.2f}")

    # Metrics for White and Black predictions
    metrics_white = [mae_white, mse_white, rmse_white, r2_white]
    metrics_black = [mae_black, mse_black, rmse_black, r2_black]

    # Combined metrics
    metrics_combined = [mae_combined, mse_combined, rmse_combined, r2_combined]

    # Metric names
    metric_names = ['MAE', 'MSE', 'RMSE', 'R²']

    # Plotting
    x = range(len(metric_names))

    plt.figure(figsize=(10, 6))

    # Bar width
    bar_width = 0.25

    # Plot individual metrics for white and black
    plt.bar(x, metrics_white, width=bar_width, label='White', alpha=0.7)
    plt.bar([p + bar_width for p in x], metrics_black, width=bar_width, label='Black', alpha=0.7)
    plt.bar([p + 2 * bar_width for p in x], metrics_combined, width=bar_width, label='Combined', alpha=0.7)

    # Adding labels and title
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.title('Performance Metrics for Elo Predictions', fontsize=16)
    plt.xticks([p + bar_width for p in x], metric_names)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

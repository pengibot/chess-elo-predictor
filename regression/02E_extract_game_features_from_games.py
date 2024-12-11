from pathlib import Path
from typing import Dict, Callable, Any

import numpy as np
import pandas as pd
import pickle


def calculate_game_features(evaluations):
    """
    Extract statistical features from a series of game evaluations.
    """
    if evaluations.size == 0:
        return {}

    evaluations_with_initial = add_initial_score(evaluations)
    total_diffs, white_turn_diffs, black_turn_diffs = compute_differences(evaluations_with_initial)

    if total_diffs.size == 0 or white_turn_diffs.size == 0 or black_turn_diffs.size == 0:
        return {}

    return aggregate_statistical_features(total_diffs, white_turn_diffs, black_turn_diffs)


def add_initial_score(scores):
    """Prepend an initial score of 0 to the score sequence."""
    return np.r_[0, scores]


def compute_differences(scores):
    """Calculate differences between consecutive scores."""
    all_differences = np.diff(scores)
    white_turn_differences = all_differences[::2]
    black_turn_differences = all_differences[1::2]
    return all_differences, white_turn_differences, black_turn_differences


def aggregate_statistical_features(all_diffs, white_diffs, black_diffs):
    """
    Compute statistical features for the differences.
    """
    feature_stats: Dict[str, Callable[..., Any]] = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'median_abs': lambda x: np.median(np.abs(x))
    }
    subsets = {
        'diffs': all_diffs,
        'white_diffs': white_diffs,
        'black_diffs': black_diffs
    }

    features = {}
    for subset_name, subset in subsets.items():
        for stat_name, stat_func in feature_stats.items():
            feature_key = f"{stat_name}_{subset_name}"
            features[feature_key] = stat_func(subset) if subset.size > 0 else np.nan

    return features


def process_game_evaluations(input_path, output_path, input_file, output_file):
    """
    Process game evaluations from a pickle file and save statistical features.
    """
    input_file_path = input_path / input_file
    output_pickle_path = output_path / f"{output_file}.pkl"
    output_csv_path = output_path / f"{output_file}.csv"

    with open(input_file_path, 'rb') as file:
        game_evaluations_df = pickle.load(file)

    if game_evaluations_df.empty:
        print("Input DataFrame is empty. Exiting...")
        return

    # Extract features for each game
    feature_records = []
    for _, row in game_evaluations_df.iterrows():
        evaluations = np.array(row['evals'])
        game_features = calculate_game_features(evaluations)
        feature_records.append(game_features)

    # Create a DataFrame from extracted features
    features_df = pd.DataFrame(feature_records)

    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to pickle and CSV
    features_df.to_pickle(output_pickle_path)
    features_df.to_csv(output_csv_path, index=False)

    print(f"Processed {len(features_df)} games.")
    print(features_df.head())


def main():
    """
    Main function to process game evaluations and extract features.
    """
    input_dir = Path("Data/Pickls")
    output_dir = Path("Data/Pickls")
    input_filename = "evals_df.pkl"
    output_filename = "game_features_df"

    print("Starting feature extraction...")
    process_game_evaluations(input_dir, output_dir, input_filename, output_filename)
    print("Feature extraction completed.")


if __name__ == "__main__":
    main()

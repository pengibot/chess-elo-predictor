import pickle
from pathlib import Path


def main():

    print("Started Combining Columns to Pickle File...")

    pickls_directory = Path("Data/Pickls")
    output_filename = r"combined_df"

    # Load the dataframes from the pickle files
    with open(pickls_directory / r"elos_df.pkl", 'rb') as file:
        elos_df = pickle.load(file)
    with open(pickls_directory / r"ecos_df.pkl", 'rb') as file:
        ecos_df = pickle.load(file)
    with open(pickls_directory / r"results_df.pkl", 'rb') as file:
        results_df = pickle.load(file)
    with open(pickls_directory / r"evals_df.pkl", 'rb') as file:
        evals_df = pickle.load(file)
    with open(pickls_directory / r"score_features_df.pkl", 'rb') as file:
        score_features_df = pickle.load(file)
    with open(pickls_directory / r"game_features_df.pkl", 'rb') as file:
        game_features_df = pickle.load(file)

    print(f"The size of the elos_df is {elos_df.shape}")
    print(f"The size of the ecos_df is {ecos_df.shape}")
    print(f"The size of the results_df is {results_df.shape}")
    print(f"The size of the evals_df is {evals_df.shape}")
    print(f"The size of the score_features_df is {score_features_df.shape}")
    print(f"The size of the game_features_df is {game_features_df.shape}")

    data_frame = elos_df.join(ecos_df) \
        .join(results_df) \
        .join(score_features_df) \
        .join(game_features_df) \
        .join(evals_df) \
        .ffill()  # It fills the missing values with the last observed value going forward along the index axis.

    data_frame["ECO"] = data_frame["ECO"].astype('category')

    # Save the DataFrame to a pickle file
    pickls_directory.mkdir(parents=True, exist_ok=True)
    data_frame.to_pickle(pickls_directory / '{}.pkl'.format(output_filename))

    # Export the DataFrame to a CSV file
    data_frame.to_csv(pickls_directory / '{}.csv'.format(output_filename), index=False)

    # Print the length of the DataFrame and the first 5 entries for verification
    print(len(data_frame))
    print(data_frame.head())

    print("Finished Combining Columns to Pickle File...")


if __name__ == "__main__":
    main()

import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    print("Started Splitting Data into 60-20-20 and Saving to Pickle Files...")

    pickls_directory = Path("Data/Pickls")
    input_filename = "combined_df"
    train_filename = "training_data_df"
    val_filename = "validation_data_df"
    test_filename = "test_data_df"

    # Load the combined dataframe
    with open(pickls_directory / f"{input_filename}.pkl", 'rb') as file:
        data_frame = pickle.load(file)

    # Using 100000 entries or fewer
    subset_df = data_frame.head(50000)

    # Split data into 60% training and 40% remaining
    train_df, temp_df = train_test_split(subset_df, test_size=0.4, random_state=42)

    # Split the remaining 40% into 20% validation and 20% test (50% of temp_df)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Save each subset to a new pickle file
    train_df.to_pickle(pickls_directory / f"{train_filename}.pkl")
    val_df.to_pickle(pickls_directory / f"{val_filename}.pkl")
    test_df.to_pickle(pickls_directory / f"{test_filename}.pkl")



    # Export the DataFrame to a CSV file
    train_df.to_csv(pickls_directory / '{}.csv'.format(train_filename), index=False)
    val_df.to_csv(pickls_directory / '{}.csv'.format(val_filename), index=False)
    test_df.to_csv(pickls_directory / '{}.csv'.format(test_filename), index=False)

    print("Finished Splitting Data into 60-20-20 and Saving to Pickle Files...")


if __name__ == "__main__":
    main()

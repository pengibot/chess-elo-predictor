import pickle
from pathlib import Path
import pandas as pd


def main():

    print("Started Cleaning/Processing Data Frame to Pickle File...")

    pickls_directory = Path("Data/Pickls")
    filename = r"combined_df"

    # Load the combined dataframe
    with open(pickls_directory / '{}.pkl'.format(filename), 'rb') as file:
        data_frame = pickle.load(file)

    print("Replacing Result values 1 --> Win, 0 --> Loss, 1/2 --> Draw")
    data_frame['Result'] = data_frame['Result'].replace({'1': 'Win', '0': 'Loss', '1/2': 'Draw'})

    # Apply one-hot encoding to the 'Result' column
    print("Applying One-Hot Encoding to Result Column")
    data_frame = pd.get_dummies(data_frame, columns=['Result'])

    # Apply one-hot encoding to the 'ECO' column with a prefix
    print("Applying One-Hot Encoding to ECO Column")
    data_frame = pd.get_dummies(data_frame, columns=['ECO'], prefix='eco')

    # Remove rows where the Stockfish column is empty or has less than 20 moves
    print("Removing rows where stockfish evaluation is fewer than 20 moves")
    data_frame = data_frame[data_frame['evals'].apply(lambda x: len(x) >= 20)]

    # Reset the index to have a clean index after dropping rows
    print("Resetting index after possible rows being dropped")
    data_frame.reset_index(drop=True, inplace=True)

    # Save the DataFrame to a pickle file
    pickls_directory.mkdir(parents=True, exist_ok=True)
    data_frame.to_pickle(pickls_directory / '{}.pkl'.format(filename))

    # Export the DataFrame to a CSV file
    data_frame.to_csv(pickls_directory / '{}.csv'.format(filename), index=False)

    # Print the length of the DataFrame and the first 5 entries for verification
    print(len(data_frame))
    print(data_frame.head())

    print("Finished Cleaning/Processing Data Frame to Pickle File...")


if __name__ == "__main__":
    main()

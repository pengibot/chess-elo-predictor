import pickle
from pathlib import Path


def describe_pickle_files(_pickle_directory, _file_names):
    """
    Loads and describes each pickle file in the given directory.

    Parameters:
    - pickle_directory: Path to the directory containing pickle files.
    - file_names: List of pickle file names (without extensions).
    """
    for file_name in _file_names:
        # Load the DataFrame from the pickle file
        with open(_pickle_directory / f"{file_name}.pkl", 'rb') as file:
            df = pickle.load(file)

        # Display file name
        print(f"\nDescription of {file_name} DataFrame:")

        # Describe the DataFrame
        print(df.describe())
        print("\n" + "=" * 50 + "\n")  # Divider for readability


pickle_directory = Path("Data/Pickls")
file_names = ["training_data_df", "validation_data_df", "test_data_df"]
describe_pickle_files(pickle_directory, file_names)

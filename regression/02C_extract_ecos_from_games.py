from pathlib import Path
import pandas as pd
import re


def extract_ecos(input_directory, output_directory, input_filename, output_filename):
    # Update the pattern to capture the ECO
    pattern = r'^\[ECO\s+"'

    ecos = []

    with open(input_directory / input_filename, 'r') as file:
        for line in file:
            if re.match(pattern, line):
                split_line = line.split("\"")
                ecos.append(split_line[1])

    data_frame = pd.DataFrame(ecos, index=range(0, 2*len(ecos), 2), columns=['ECO'])

    # Save the DataFrame to a pickle file
    output_directory.mkdir(parents=True, exist_ok=True)
    data_frame.to_pickle(output_directory / '{}.pkl'.format(output_filename))

    # Export the DataFrame to a CSV file
    data_frame.to_csv(output_directory / '{}.csv'.format(output_filename), index=False)

    # Print the length of the DataFrame and the first 5 entries for verification
    print(len(data_frame))
    print(data_frame.head())


def main():

    input_directory = Path("../filter-games/Data/TrainingData")
    output_directory = Path("Data/Pickls")
    input_filename = r"combined_games.pgn"
    output_filename = r"ecos_df"

    print("Started Extracting Ecos to Pickle File...")
    extract_ecos(input_directory, output_directory, input_filename, output_filename)
    print("Finished Extracting Ecos to Pickle File...")


if __name__ == "__main__":
    main()

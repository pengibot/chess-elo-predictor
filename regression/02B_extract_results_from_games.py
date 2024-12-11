from pathlib import Path
import pandas as pd
import re


def extract_results(input_directory, output_directory, input_filename, output_filename):
    # Update the pattern to capture Win, Loss or Draw
    pattern = r'\[Result\s*"(1\/2-1\/2|1-0|0-1)"\]'

    results = []

    with open(input_directory / input_filename, 'r') as file:
        for line in file:
            if re.search(pattern, line):
                split_line = line.split("\"")
                split_result = split_line[1].split("-")
                results.append(split_result[0])
                results.append(split_result[1])

    data_frame = pd.DataFrame(results, columns=['Result'])

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
    output_filename = r"results_df"

    print("Started Extracting Results to Pickle File...")
    extract_results(input_directory, output_directory, input_filename, output_filename)
    print("Finished Extracting Results to Pickle File...")


if __name__ == "__main__":
    main()

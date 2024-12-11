from pathlib import Path
import pandas as pd
import re


def extract_elos(input_directory, output_directory, input_filename, output_filename):
    # Update the pattern to capture both WhiteElo and BlackElo
    pattern = r'(WhiteElo|BlackElo)\s*"\d+"'

    elos = []
    is_white_player = []
    is_black_player = []

    with open(input_directory / input_filename, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Extract the ELO value and whether the player is White or Black
                split_line = line.split("\"")
                elo = int(split_line[1])
                elos.append(elo)

                if 'WhiteElo' in line:
                    is_white_player.append(1)
                    is_black_player.append(0)
                elif 'BlackElo' in line:
                    is_white_player.append(0)
                    is_black_player.append(1)

    # Create a DataFrame with the ELO and player color information
    data_frame = pd.DataFrame({
        'ELO': elos,
        'is_white_player': is_white_player,
        'is_black_player': is_black_player
    })

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
    output_filename = r"elos_df"

    # Elo extraction
    print("Started Extracting Elos to Pickle File...")
    extract_elos(input_directory, output_directory, input_filename, output_filename)
    print("Finished Extracting Elos to Pickle File...")


if __name__ == "__main__":
    main()

import re
from pathlib import Path
import pandas as pd


def extract_and_aggregate_evals(pgn_content):
    # Updated regex to match both numerical evals and checkmate evals
    eval_pattern = re.compile(r'\[%eval (-?\d+\.\d+|#-?\d+)]')
    games_evaluations = []
    current_game_evals = []

    for line in pgn_content:
        if line.startswith('[Event '):  # New game starts
            if current_game_evals:  # If there are evaluations from the previous game, save them
                games_evaluations.append(current_game_evals)
            current_game_evals = []  # Start a new list for the next game

        for eval_match in eval_pattern.finditer(line):
            eval_value = eval_match.group(1)
            if eval_value.startswith('#'):
                # Checkmate evaluation
                eval_value = int(eval_value.replace('#', '')) * 10000  # Large number to represent checkmate
            else:
                # Numerical evaluation
                eval_value = round(float(eval_value) * 100)  # Convert to centipawns
            current_game_evals.append(eval_value)

    # Add the last game's evaluations if they exist
    if current_game_evals:
        games_evaluations.append(current_game_evals)

    return games_evaluations


def main():
    input_directory = Path("../filter-games/Data/TrainingData")
    output_directory = Path("Data/Pickls")
    input_filename = r"combined_games.pgn"
    output_filename = r"evals_df"

    # Eval extraction
    print("Started Extracting Evals to Pickle File...")

    with open(input_directory / input_filename, 'r') as file:
        pgn_content = file.readlines()

    games_evaluations = extract_and_aggregate_evals(pgn_content)

    data_frame = pd.DataFrame({
        "evals": games_evaluations
    })
    data_frame.index = list(range(0, 2 * len(games_evaluations), 2))

    # Save the DataFrame to a pickle file
    output_directory.mkdir(parents=True, exist_ok=True)
    data_frame.to_pickle(output_directory / '{}.pkl'.format(output_filename))

    # Export the DataFrame to a CSV file
    data_frame.to_csv(output_directory / '{}.csv'.format(output_filename), index=False)

    # Print the length of the DataFrame and the first 5 entries for verification
    print(len(data_frame))
    print(data_frame["evals"][0])
    print(data_frame["evals"][2])
    print(data_frame["evals"][4])
    print(data_frame["evals"][6])
    print(data_frame["evals"][8])

    print("Finished Extracting Evals to Pickle File...")


if __name__ == "__main__":
    main()

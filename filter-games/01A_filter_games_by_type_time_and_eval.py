import os
import re
import sys
from pathlib import Path


def filter_games_with_eval(pgn_content):
    """
    Filters games from PGN content based on specific criteria:
    - Contains evaluation data ('%eval')
    - Is a Rated Rapid game
    - Matches the specified time control pattern

    Parameters:
        pgn_content (list): List of lines from the PGN file.

    Returns:
        list: Filtered games as a list of concatenated strings.
    """
    filtered_games = []  # List to store games that meet criteria
    current_game = []  # Temporary storage for the current game being processed
    game_has_eval = False  # Flag to check if game has evaluation data
    game_type_rated_rapid_game = False  # Flag for Rated Rapid game type
    pattern = r'\[TimeControl "([9]\d{2}|[1-9]\d{3,})\+\d+"\]'  # Matches specific time control format
    game_length = False  # Flag to check if the game length condition is satisfied

    # Process each line in the PGN content
    for line in pgn_content:
        if line.startswith("[Event"):  # New game starts
            # If the previous game meets all criteria, add it to the filtered list
            if current_game and game_has_eval and game_type_rated_rapid_game and game_length:
                filtered_games.append(''.join(current_game))

            # Reset flags and store the current game start line
            current_game = [line]
            game_has_eval = False
            game_length = False
            game_type_rated_rapid_game = line.startswith("[Event \"Rated Rapid game\"]")
        else:
            # Continue processing the current game
            current_game.append(line)
            if '%eval' in line:  # Check for evaluation data
                game_has_eval = True
            if re.search(pattern, line):  # Check for valid time control
                game_length = True

    # Add the last game if it meets all criteria
    if current_game and game_has_eval and game_type_rated_rapid_game and game_length:
        filtered_games.append(''.join(current_game))

    return filtered_games


def main():
    """
    Main function to filter games from PGN files and save the filtered games
    to a specified output directory.
    """
    print("Started Filtering Games")  # Indicate the start of the process

    # Define the input and output directories
    input_directory = Path("Data/Lichess")  # Directory with unfiltered PGN files
    output_directory = Path("Data/FilteredGames")  # Directory for filtered PGN files

    # Prefix for the output files
    output_file = 'output_'
    counter = 1  # Counter for output file naming
    game_count = 0  # Total count of filtered games

    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        # Check if the file has a .pgn extension
        if filename.endswith(".pgn"):
            input_file = os.path.join(input_directory, filename)

            # Read the content of the PGN file
            with open(input_file, 'r') as file:
                pgn_content = file.readlines()

            # Filter the games using the criteria defined in the helper function
            filtered_games = filter_games_with_eval(pgn_content)
            game_count += len(filtered_games)  # Update the total count of filtered games

            # Define the name of the output file for the filtered games
            output_file_name = output_directory / f"{output_file}{counter}.pgn"

            # Ensure the output directory exists; create it if it doesn't
            output_directory.mkdir(parents=True, exist_ok=True)

            # Write the filtered games to the output file
            with open(output_file_name, 'w') as file:
                file.writelines(filtered_games)

            counter += 1  # Increment the counter for the next output file

            # Display progress in the terminal
            sys.stdout.write(f"\r{''}")  # Clears the terminal line
            sys.stdout.write(f"Filtered {game_count} Games".ljust(50))  # Display total filtered games
            sys.stdout.flush()  # Ensure the message is displayed immediately

    print("\nFinished Filtering Games")  # Indicate the completion of the process


if __name__ == "__main__":
    # Entry point of the script
    main()

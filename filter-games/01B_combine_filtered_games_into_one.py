import os
from pathlib import Path


def main():
    """
    Combines multiple PGN files from a specified input directory into a single output file.

    The combined games are saved in the specified output directory.
    """
    print("Started Combining Games")  # Indicate the start of the process

    # Define the input directory containing filtered PGN files
    input_directory = Path("Data/FilteredGames")

    # Define the output directory where the combined PGN file will be saved
    output_directory = Path("Data/TrainingData")

    # Name of the output file where all combined games will be stored
    output_file = 'combined_games.pgn'

    # Ensure the output directory exists; create it if it does not
    output_directory.mkdir(parents=True, exist_ok=True)

    # Open the output file for writing the combined content
    with open(output_directory / output_file, 'w') as outfile:
        # Loop through all files in the input directory
        for pgn_file in os.listdir(input_directory):
            file_path = input_directory / pgn_file  # Full path of the current file

            # Process only files with a .pgn extension
            if file_path.suffix.lower() == ".pgn":
                # Open the current PGN file for reading
                with open(file_path, 'r') as infile:
                    contents = infile.read()  # Read the entire content of the file
                    outfile.write(contents)  # Append the content to the output file

    # Print a confirmation message indicating where the combined file is saved
    print(f"Finished Combining Games, saved file to {output_directory / output_file}")


# Entry point of the script
if __name__ == "__main__":
    main()

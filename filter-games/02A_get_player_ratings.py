from pathlib import Path
import chess.pgn


def get_player_ratings(_pgn_file, _player_name):
    """
    Parse the PGN file to find the ratings of the specified player
    in all games, along with the highest and lowest ratings.

    Parameters:
        _pgn_file (Path): Path to the PGN file.
        _player_name (str): Name of the player whose ratings are being tracked.

    Returns:
        dict: A dictionary containing the player's ratings history,
              highest rating, and lowest rating.
    """
    ratings = []  # List to store the player's Elo ratings from all games

    # Open the PGN file for reading
    with open(_pgn_file, 'r') as file:
        while True:
            # Read the next game in the PGN file
            game = chess.pgn.read_game(file)
            if game is None:  # Stop if no more games are found
                break

            # Extract player names and ratings from the game headers
            white_player = game.headers.get("White")  # White player's name
            black_player = game.headers.get("Black")  # Black player's name
            white_elo = game.headers.get("WhiteElo")  # White player's Elo rating
            black_elo = game.headers.get("BlackElo")  # Black player's Elo rating

            # Check if the specified player participated in this game
            if white_player == _player_name and white_elo is not None:
                ratings.append(int(white_elo))  # Add the player's White Elo rating
            elif black_player == _player_name and black_elo is not None:
                ratings.append(int(black_elo))  # Add the player's Black Elo rating

    # Determine the highest and lowest ratings from the collected data
    if ratings:
        highest_rating = max(ratings)  # Maximum rating in the list
        lowest_rating = min(ratings)  # Minimum rating in the list
    else:
        highest_rating = lowest_rating = None  # Handle case with no ratings found

    # Return a dictionary containing the player's rating history and extremes
    return {
        "ratings_history": ratings,
        "highest_rating": highest_rating,
        "lowest_rating": lowest_rating
    }


def save_ratings_to_file(ratings_data, filename):
    """
    Save the player's ratings data to a text file.

    Parameters:
        ratings_data (dict): The dictionary containing ratings history, highest, and lowest ratings.
        filename (Path): Path to the file where the ratings will be saved.
    """
    # Open the specified file for writing
    with open(filename, 'w') as file:
        # Write the ratings history to the file
        file.write(f"{ratings_data['ratings_history']}")


# Example usage
input_directory = Path("Data/TrainingData")  # Path to the PGN directory containing chess games
input_file_name = "combined_games.pgn"  # Name of the input file
player_name = "MatsumotoKiyoshi"  # Name of the player to analyze

# Get the player's ratings data from the PGN file
result = get_player_ratings(input_directory / input_file_name, player_name)

# Print the results to the console
print(f"Player: {player_name}")
print(f"Ratings History: {result['ratings_history']}")
print(f"Highest Rating: {result['highest_rating']}")
print(f"Lowest Rating: {result['lowest_rating']}")

# Save the results to a text file
# Define the output directory and filename for the player's ratings
output_directory = Path("Data")
file_name = "player_ratings.txt"  # Name of the output file
output_file = output_directory / file_name  # Full path to the output file

# Save the ratings history to the output file
save_ratings_to_file(result, output_file)

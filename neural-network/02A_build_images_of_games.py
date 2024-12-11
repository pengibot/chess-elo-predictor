import sys
from pathlib import Path
import chess
import chess.pgn
import matplotlib.pyplot as plt
import numpy as np

# Define a color map for chess pieces in RGB format
color_map_rgb = {
    # Black Pieces
    chess.PAWN: (0, 0.5, 0),  # dark green
    chess.KNIGHT: (0, 0, 0.5),  # dark blue
    chess.BISHOP: (0.25, 0, 0.25),  # dark purple
    chess.ROOK: (0.5, 0.25, 0),  # dark orange
    chess.QUEEN: (0.5, 0, 0),  # dark red
    chess.KING: (0.5, 0.42, 0),  # dark gold

    # White Pieces
    chess.PAWN + 6: (0, 1, 0),  # green
    chess.KNIGHT + 6: (0, 0, 1),  # blue
    chess.BISHOP + 6: (0.5, 0, 0.5),  # purple
    chess.ROOK + 6: (1, 0.5, 0),  # orange
    chess.QUEEN + 6: (1, 0, 0),  # red
    chess.KING + 6: (1, 0.84, 0),  # gold
}


# Helper function to retrieve the color for a chess piece
def get_piece_color(piece):
    if piece.color == chess.WHITE:
        return color_map_rgb[piece.piece_type]
    else:
        return color_map_rgb[piece.piece_type + 6]


# Function to generate color-coded images for each move in the game
def generate_colour_images_for_each_move(pgn, output_directory, total_number_of_games):
    game_number = 1

    # Process each game in the PGN file
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break  # Exit if no more games are available

        # Print progress
        sys.stdout.write(f"\r{''}")  # Clear the line
        sys.stdout.write(f"Processing Game {game_number}/{total_number_of_games}".ljust(50))
        sys.stdout.flush()

        # Create a directory for the game's images
        directory = Path(output_directory) / f'Game_{game_number}'
        directory.mkdir(parents=True, exist_ok=True)

        board = game.board()
        move_number = 1

        # Get metadata from PGN headers
        white_elo = game.headers.get('WhiteElo', 'NA')
        black_elo = game.headers.get('BlackElo', 'NA')
        result = game.headers.get('Result', 'NA')

        # Determine the winner for naming purposes
        if result == '1-0':
            winner = 'white'
        elif result == '0-1':
            winner = 'black'
        elif result == '1/2-1/2':
            winner = 'draw'
        else:
            winner = 'undetermined'

        # Process moves in the game
        for move in game.mainline_moves():
            board.push(move)

            # Create a 8x8x3 grid with white background
            grid = np.ones((8, 8, 3))

            # Update the grid based on the board state
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    grid[chess.square_rank(square)][chess.square_file(square)] = get_piece_color(piece)

            # Generate filename for the current move
            filename = f'chessboard_move_{move_number}_W{white_elo}_B{black_elo}_{winner}.png'

            # Save the grid as an image
            plot_color_grid(grid, filename=filename, directory=directory)
            move_number += 1

        game_number += 1

    print("\nProcessing complete!")


# Function to save the grid as an image
def plot_color_grid(grid, filename='chessboard.png', directory=Path('.')):
    """Plots the color grid and saves it as an image with no grid lines and 8x8 pixels."""
    fig, ax = plt.subplots()
    ax.imshow(grid, aspect='equal')  # Ensure the image fills the plot area
    ax.axis('off')  # Turn off the axis

    # Set the size of the figure to match the desired 8x8 pixel resolution
    dpi = 100  # Define the resolution in dots per inch
    pixel_size = 8  # We want the image to be 8x8 pixels
    inches = pixel_size / dpi  # Convert pixel dimensions to inches

    fig.set_size_inches(inches, inches)  # Set the dimensions of the figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Save the image with the exact dimensions and DPI
    plt.savefig(f"{directory}/{filename}", dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)


# Function to count the number of games in a PGN file
def count_games_in_pgn(file_path):
    with open(file_path, 'r') as pgn_file:
        game_count = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # End of file
            game_count += 1
    return game_count


# Main function to drive the process
def main():
    print("Started Building Images of Games")

    # Define input and output paths
    input_directory = Path("../filter-games/Data/TrainingData")
    input_file = r'combined_games.pgn'
    output_directory = Path("Data/Games")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Count the total number of games in the PGN file
    total_number_of_games = count_games_in_pgn(input_directory / input_file)

    # Process the PGN file and generate images
    with open(input_directory / input_file) as pgn_file:
        generate_colour_images_for_each_move(pgn_file, output_directory, total_number_of_games)

    print("Finished Building Images of Games")


# Execute the script
if __name__ == "__main__":
    main()

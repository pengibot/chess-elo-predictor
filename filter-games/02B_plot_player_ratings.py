from pathlib import Path
import matplotlib.pyplot as plt


def plot_ratings(_ratings, player_name):
    """
    Plot the Elo rating history of a player and annotate the highest and lowest ratings.

    Parameters:
        _ratings (list): List of player's Elo ratings over time.
        player_name (str): Name of the player.
    """
    # Determine the highest and lowest ratings
    highest_rating = max(_ratings)  # Highest Elo rating in the list
    lowest_rating = min(_ratings)  # Lowest Elo rating in the list
    highest_index = _ratings.index(highest_rating)  # Index of the highest rating
    lowest_index = _ratings.index(lowest_rating)  # Index of the lowest rating

    # Create a plot with a specified figure size
    plt.figure(figsize=(12, 8))

    # Plot the rating history with markers and a line
    plt.plot(_ratings, marker='o', linestyle='-', color='b', label=f'{player_name} Rating')

    # Add horizontal lines for the highest and lowest ratings
    plt.axhline(highest_rating, color='red', linestyle='--', label=f'Highest Rating ({highest_rating})')
    plt.axhline(lowest_rating, color='red', linestyle='--', label=f'Lowest Rating ({lowest_rating})')

    # Add a vertical line to show the range between highest and lowest ratings
    plt.vlines(
        x=highest_index, ymin=lowest_rating, ymax=highest_rating, colors='purple', linestyles=':',
        label='Rating Range'
    )

    # Annotate the highest rating point
    plt.text(highest_index, highest_rating, f'{highest_rating}', color='red', verticalalignment='bottom')

    # Annotate the lowest rating point
    plt.text(lowest_index, lowest_rating, f'{lowest_rating}', color='red', verticalalignment='top')

    # Add a label to show the difference between highest and lowest ratings
    difference = highest_rating - lowest_rating  # Calculate the rating difference
    plt.text(
        highest_index, (highest_rating + lowest_rating) / 2, f'Difference: {difference}', color='purple',
        horizontalalignment='right', verticalalignment='center', rotation=90
    )

    # Add labels and title to the plot
    plt.xlabel("Game Number")  # Label for the x-axis
    plt.ylabel("Elo Rating")  # Label for the y-axis
    plt.title(f"Rating History of {player_name}")  # Plot title

    # Add a legend and grid for better readability
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()


# Define the directory and file containing the player's ratings
input_directory = Path("Data")  # Directory containing the input file
file_name = "player_ratings.txt"  # Name of the file with ratings data
output_file = input_directory / file_name  # Path to the input file

# Read the ratings data from the file
with open(output_file, 'r') as file:
    # Convert the string of ratings into a list of integers
    ratings = [int(num.strip()) for num in file.readline().strip("[]").split(",")]

# Call the function to plot the ratings
plot_ratings(ratings, 'player')

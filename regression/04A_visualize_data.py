import pickle
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt


def main():
    print("Started Visualizing Data Frame...")

    pickls_directory = Path("Data/Pickls")
    filename = r"training_data_df"

    with open(pickls_directory / '{}.pkl'.format(filename), 'rb') as file:
        data_frame = pickle.load(file)

    # Identify non-numeric columns (including lists)
    non_numeric_columns = data_frame.columns[data_frame.map(type).isin([list, object]).any()]

    # Exclude columns that start with 'eco' and non-numeric columns for the correlation calculation
    columns_to_exclude = non_numeric_columns.union(data_frame.columns[data_frame.columns.str.startswith('eco')])
    df_numeric = data_frame.drop(columns=columns_to_exclude)

    # 1. Distribution of ELO Ratings
    # Purpose: To see the distribution of ELO ratings across the dataset.
    plt.figure(figsize=(10, 6))
    sns.histplot(data_frame['ELO'], bins=30, kde=True)
    plt.title('Distribution of ELO Ratings')
    plt.xlabel('ELO')
    plt.ylabel('Frequency')
    plt.show()

    # 2. ELO Ratings by Player Color
    # Purpose: To compare the ELO ratings of players based on whether they are playing as White or Black.
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_white_player', y='ELO', data=data_frame)
    plt.title('ELO Ratings by Player Color')
    plt.xlabel('Is White Player')
    plt.ylabel('ELO')
    plt.xticks([0, 1], ['Black', 'White'])
    plt.show()

    # 3. Correlation Heatmap
    # Purpose: To identify the correlations between different features and the ELO rating.
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_numeric.corr(), annot=False, cmap='coolwarm')  # Set annot=False to remove the values
    plt.title('Correlation Heatmap (Excluding ECO Columns)')
    plt.tight_layout()
    plt.show()

    # 4. ELO vs. Number of Blunders (Black and White)
    # Purpose: To examine the relationship between the number of blunders made and the ELO rating.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='black_blunder_count', y='ELO', data=data_frame, label='Black')
    sns.scatterplot(x='white_blunder_count', y='ELO', data=data_frame, label='White')
    plt.title('ELO vs. Number of Blunders')
    plt.xlabel('Number of Blunders')
    plt.ylabel('ELO')
    plt.legend()
    plt.show()

    # 5. ELO vs. Mean Difference in Scores
    # Purpose: To observe how the mean difference in scores correlates with the ELO rating.
    plt.figure(figsize=(10, 6))
    sns.regplot(x='mean_diffs', y='ELO', data=data_frame, scatter_kws={'s': 10})
    plt.title('ELO vs. Mean Difference in Scores')
    plt.xlabel('Mean Difference in Scores')
    plt.ylabel('ELO')
    plt.show()

    # 6. Material Advantage vs. ELO
    # Purpose: To explore the relationship between material advantage and ELO.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='material_advantage', y='ELO', data=data_frame)
    plt.title('Material Advantage vs. ELO')
    plt.xlabel('Material Advantage')
    plt.ylabel('ELO')
    plt.show()

    # 7. ELO vs. Total Moves
    # Purpose: To see if there is any correlation between the length of the game (total moves) and the player's ELO.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='total_moves', y='ELO', data=data_frame)
    plt.title('ELO vs. Total Moves')
    plt.xlabel('Total Moves')
    plt.ylabel('ELO')
    plt.show()

    # 8. ELO vs. Number of Inaccuracies (Black and White)
    # Purpose: To determine if there is any relationship between inaccuracies made and ELO.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='black_inaccuracy_count', y='ELO', data=data_frame, label='Black')
    sns.scatterplot(x='white_inaccuracy_count', y='ELO', data=data_frame, label='White')
    plt.title('ELO vs. Number of Inaccuracies')
    plt.xlabel('Number of Inaccuracies')
    plt.ylabel('ELO')
    plt.legend()
    plt.show()

    # 9. Piece Activity vs. ELO
    # Purpose: To compare the piece activity (both Black and White) across different ELO ratings.
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ELO', y='black_piece_activity', data=data_frame)
    sns.boxplot(x='ELO', y='white_piece_activity', data=data_frame)
    plt.title('Piece Activity vs. ELO')
    plt.xlabel('ELO')
    plt.ylabel('Piece Activity')
    plt.show()

    # 10. ELO vs. Check Counts (Black and White)
    # Purpose: To examine how the number of checks given during a game relates to the player's ELO.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='black_total_checks', y='ELO', data=data_frame, label='Black')
    sns.scatterplot(x='white_total_checks', y='ELO', data=data_frame, label='White')
    plt.title('ELO vs. Check Counts')
    plt.xlabel('Total Checks')
    plt.ylabel('ELO')
    plt.legend()
    plt.show()

    print("Finished Visualizing Data Frame...")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Provided list of Stockfish evaluations
values_list = [
    31, 32, 17, 32, 24, 108, 40, 48, 48, 55, 57, 57, 53, 56, 37, 319, 307, 546, 531, 566, 521, 561, 560, 586, 553, 575, 557, 580, 561, 595, 549, 593, 588, 592, 548, 662, 646, 630, 641, 667, 660, 664, 665, 658, 667, 659, 642, 661, 628, 642, 644, 764, 777, 759, 707, 717, 636, 654, 496, 522, 512, 505, 496, 500, 498, 494, 493, 497, 500, 512, 512, 518, 540, 561, 539, 571, 580, 592, 596, 599, 598, 623, 609, 618, 619, 54
]

# Calculate differences between consecutive values
differences = np.diff(values_list)

# Create a DataFrame to store results
df = pd.DataFrame({
    'Index': range(1, len(values_list)),
    'Previous Value': values_list[:-1],
    'Current Value': values_list[1:],
    'Difference': differences
})

# Filter rows where the absolute difference is greater than or equal to 200
significant_changes = df[abs(df['Difference']) >= 200].reset_index(drop=True)

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(significant_changes['Index'], significant_changes['Difference'], color='skyblue')
plt.xlabel('Move Index')
plt.ylabel('Change in Evaluation')
plt.title('Significant Changes in Stockfish Evaluation (±200)')
plt.axhline(200, color='red', linestyle='--', linewidth=1)
plt.axhline(-200, color='red', linestyle='--', linewidth=1)
plt.show()

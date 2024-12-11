import subprocess
import sys
from pathlib import Path


def run_script(script_name):
    """
    Runs the provided Python script using the current Python interpreter.

    Parameters:
        script_name (str or Path): Name or path of the Python script to execute.

    Returns:
        bool: True if the script executed successfully, False otherwise.
    """
    # Use subprocess to run the script with the current Python interpreter
    result = subprocess.run([sys.executable, script_name], text=True)

    # Check the return code to determine if the script ran successfully
    if result.returncode == 0:
        print(f"\u2705 {script_name} executed successfully.")  # Success message with a checkmark emoji
        return True
    else:
        print(f"\u2757 Error executing {script_name}: {result.stderr}")  # Error message with details
        return False


def main():
    """
    Main function to sequentially execute a list of Python scripts.
    Stops execution if any script fails.
    """
    # List of scripts to run in sequence
    scripts = [
        Path("01A_filter_games_by_type_time_and_eval.py"),  # Script to filter games based on criteria
        Path("01B_combine_filtered_games_into_one.py"),     # Script to combine filtered games into one file
        Path("02A_get_player_ratings.py"),                  # Script to extract player ratings
        Path("02B_plot_player_ratings.py")                  # Script to plot the player ratings
    ]

    # Iterate over each script in the list
    for script in scripts:
        # Resolve the script path to an absolute path and execute it
        result = run_script(script.resolve())
        if not result:  # Stop execution if a script fails
            break


if __name__ == "__main__":
    # Entry point of the program
    main()

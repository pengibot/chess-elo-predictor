import subprocess
import sys
from pathlib import Path


def run_script(script_name):
    """Runs the provided Python script using the current Python interpreter."""
    result = subprocess.run([sys.executable, script_name], text=True)

    # Check for errors
    if result.returncode == 0:
        print(f"\u2705 {script_name} executed successfully.")
        return True
    else:
        print(f"\u2757 Error executing {script_name}: {result.stderr}")
        return False


def main():
    scripts = [Path("02A_build_images_of_games.py"),
               Path("03A_create_model.py"),
               Path("03B_analyse_history.py"),
               Path("03C_analyse_best_model.py"),
               Path("03D_actual_vs_predicted.py"),
               Path("03E_visualize_model.py")]

    for script in scripts:
        result = run_script(script.resolve())
        if not result:
            break


if __name__ == "__main__":
    main()

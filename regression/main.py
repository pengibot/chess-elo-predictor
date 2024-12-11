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
    scripts = [Path("02A_extract_elos_from_games.py"),
               Path("02B_extract_results_from_games.py"),
               Path("02C_extract_ecos_from_games.py"),
               Path("02D_extract_evals_from_games.py"),
               Path("02E_extract_game_features_from_games.py"),
               Path("02F_extract_score_features_from_games.py"),
               Path("03A_combine_all_columns.py"),
               Path("03B_clean_and_process.py"),
               Path("03C_split_data.py"),
               Path("03D_describe_data.py"),
               Path("04A_visualize_data.py"),
               Path("05A_building_model_elastic_net.py"),
               Path("05B_building_model_random_forest.py"),
               Path("05C_building_model_support_vector.py"),
               Path("06A_build_confusion_matrix.py"),
               Path("06B_residual_analysis.py"),
               Path("06C_Metrics_Comparison.py"),
               Path("06D_Confidence_Chart.py"),
               Path("07A_analysing_stockfish_evaluations.py"),
               Path("07B_convert_datframe_to_arff.py")]

    for script in scripts:
        result = run_script(script)
        if not result:
            break


if __name__ == "__main__":
    main()

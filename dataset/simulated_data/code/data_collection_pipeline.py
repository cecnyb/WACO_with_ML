import subprocess
import os

"""
Pipeline Script Documentation

Description:
------------
This script is designed to execute a sequence of scripts (Python files, shell scripts, or executables) in a predefined order. 
Each script is executed sequentially, and the pipeline stops if any script fails.

Functions:
----------
run_pipeline(scripts: list)
    Executes the scripts in the specified order.
    Automatically detects and runs Python scripts, shell scripts, or executables.
    Stops execution if any script fails and prints an error message.

Usage Example:
--------------
scripts = [
    "$WACOHOME/dataset/simulated_data/code/simulate_matrices",
    "$WACOHOME/WACO/training_data_generator/SpMM_SuperSchedule_Generator_Par.py",
    "$WACOHOME/WACO/SDDMM/TrainingData/runtime_collection_script.sh"
]

run_pipeline(scripts)

Notes:
------
- If a script fails, the pipeline stops execution and prints an error message.
- Requires Python 3.6 or later.
- Scripts must have appropriate permissions and dependencies.
"""

WACO_HOME = os.environ.get("WACO_HOME")

def run_pipeline():
    # Define the scripts to run
    scripts = [
        f"{WACO_HOME}/dataset/simulated_data/code/simulate_matrices.py",
        f"{WACO_HOME}/WACO/training_data_generator/SpMM_SuperSchedule_Generator_Par.py",
        f"{WACO_HOME}/WACO/SDDMM/TrainingData/runtime_collection_script.sh"
    ]

    # Execute each script in order
    for script in scripts:
        try:
            if script.endswith(".py"):
                # Run Python scripts
                subprocess.run(["python", script], check=True)
            elif script.endswith(".sh"):
                # Run shell scripts
                subprocess.run(["bash", script], check=True)
            else:
                # Run executable files
                subprocess.run([script], check=True)
            print(f"Successfully executed: {script}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing: {script}")
            print(e)
            break

if __name__ == "__main__":
    run_pipeline()

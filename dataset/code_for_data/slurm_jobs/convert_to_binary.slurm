#!/bin/bash
#SBATCH --job-name=web_brows        # Job name
#SBATCH --output=/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/code_for_data/matrix_converter.out      # Standard output log
#SBATCH --error=/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/code_for_data/matrix_converter.err       # Standard error log
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --mem=32G                            # Memory limit
#SBATCH --partition=short                   # Partition to submit to
#SBATCH --time=12:00:00                           # Expected time for the job (adjust as needed)
#SBATCH --partition=general
#SBATCH --cpus-per-task=12   


# Navigate to the directory containing your script
cd $WACO_HOME/dataset/code_for_data

# Run the Python script
python convert_to_binary.py



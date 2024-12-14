#!/bin/bash
#SBATCH --job-name=SpMM_SuperSchedule_Generator_adjusted        # Job name
#SBATCH --output=SpMM_SuperSchedule_Generator_adjusted.out      # Standard output log
#SBATCH --error=SpMM_SuperSchedule_Generator_adjusted.err       # Standard error log
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --mem=16G                            # Memory limit
#SBATCH --partition=short                   # Partition to submit to
#SBATCH --time=12:00:00                           # Expected time for the job (adjust as needed)
#SBATCH --partition=general
#SBATCH --cpus-per-task=20  


# Navigate to the directory containing your script
cd $WACO_HOME/WACO/training_data_generator
INPUT_DIR="/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/all_csr_files"

# Run the Python script
python SpMM_SuperSchedule_Generator_Par.py --input_dir $INPUT_DIR

#try running this code. 
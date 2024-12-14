#!/bin/bash
#SBATCH --job-name=web_brows        # Job name
#SBATCH --output=file_conv.out      # Standard output log
#SBATCH --error=file_conv.err       # Standard error log
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --mem=4G                            # Memory limit
#SBATCH --partition=short                   # Partition to submit to
#SBATCH --time=12:00:00                           # Expected time for the job (adjust as needed)
#SBATCH --partition=general
#SBATCH --cpus-per-task=8    


# Navigate to the directory containing your script
cd $WACO_HOME/WACO/sparce_matrix_collector

# Run the Python script
python file_conv_par.py

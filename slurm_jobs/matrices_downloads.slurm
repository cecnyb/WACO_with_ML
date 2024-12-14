#!/bin/bash
#SBATCH --job-name=web_brows        # Job name
#SBATCH --output=web_brows.out      # Standard output log
#SBATCH --error=web_brows.err       # Standard error log
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --mem=4G                            # Memory limit
#SBATCH --partition=short                   # Partition to submit to
#SBATCH --time=12:00:00                           # Expected time for the job (adjust as needed)
#SBATCH --partition=general
#SBATCH --cpus-per-task=4    


# Navigate to the directory containing your script
cd $WACO_HOME/WACO/sparce_matrix_collector/web_brows.py

# Run the Python script
python web_brows.py

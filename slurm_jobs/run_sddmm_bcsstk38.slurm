#!/bin/bash
#SBATCH --job-name=run_sddmm_bcsstk38     # Job name for SDDMM operation on bcsstk38
#SBATCH --output=run_sddmm_bcsstk38.out   # Output file
#SBATCH --error=run_sddmm_bcsstk38.err    # Error file
#SBATCH --time=12:00:00                           # Expected time for the job (adjust as needed)
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                         # CPUs per task

cd $WACO_HOME/WACO/training_data_generator
python SpMM_SuperSchedule_Generator.py $WACO_HOME/WACO/SpMM/TrainingData/total.txt
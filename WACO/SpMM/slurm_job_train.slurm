#!/bin/bash
#SBATCH --partition=class4
#SBATCH --time=12:00:00
#SBATCH --job-name=my_gpu_job
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB 


python /home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SpMM/only_meta_train.py
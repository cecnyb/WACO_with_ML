#!/bin/bash

#SBATCH --job-name=runtime_gen          # Job name
#SBATCH --output=runtime_gen_%j.out     # Standard output log file
#SBATCH --error=runtime_gen_%j.err      # Standard error log file
#SBATCH --time=12:00:00                 # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                      # Single task
#SBATCH --cpus-per-task=1               # CPU cores per task
#SBATCH --mem=8G                       # Memory allocation
#SBATCH --partition=standard            # Partition/queue to run the job

# Set environment variables and paths
WACO_HOME=$WACO_HOME  
CODE_GENERATOR="$WACO_HOME/code_generator/spmm"
DATASET_DIR="$WACO_HOME/dataset/ss_matrix_collection/all_csr_files_binary"
CONFIG_DIR="$WACO_HOME/WACO/training_data_generator/config"
OUTPUT_DIR="$WACO_HOME/WACO/SpMM/TrainingData/CollectedData"

# Input file containing the list of sparse matrices
INPUT_FILE=$1

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find empty files and filter INPUT_FILE to process only matching entries
filtered_file=$(mktemp) # Create a temporary file to store filtered filenames
grep -Fx -f <(find "$OUTPUT_DIR" -type f -empty -exec basename {} .txt \;) "$INPUT_FILE" > "$filtered_file"

# Loop through each matrix name in the filtered input file
while read -r matrix_name; do
    if [ -z "$matrix_name" ]; then
        continue
    fi

    # Paths for the sparse matrix and its superschedule config
    csr_path="$DATASET_DIR/${matrix_name}.csr"
    config_path="$CONFIG_DIR/${matrix_name}.txt"
    output_path="$OUTPUT_DIR/${matrix_name}.txt"

    # Check if necessary files exist
    if [[ ! -f "$csr_path" || ! -f "$config_path" ]]; then
        echo "Missing file(s) for matrix: $matrix_name. Skipping." >> error.log
        continue
    fi

    # Run the spmm code generator and capture runtime
    echo "Processing $matrix_name..."
    $CODE_GENERATOR $csr_path $config_path > "$output_path"

    # Verify if runtimes were successfully collected
    if [[ -s "$output_path" ]]; then
        echo "Runtimes for $matrix_name saved to $output_path"
    else
        echo "Failed to collect runtimes for $matrix_name. Check the spmm execution." >> error.log
    fi

done < "$filtered_file"

# Cleanup temporary file
rm -f "$filtered_file"

echo "Runtime generation completed. Check $OUTPUT_DIR for results and error.log for issues."

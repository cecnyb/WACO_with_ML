#!/bin/bash

# Set environment variables and paths
WACO_HOME=$WACO_HOME  
CODE_GENERATOR="$WACO_HOME/code_generator/spmm"
DATASET_DIR="$WACO_HOME/dataset/simulated_data/simulated_matrices"
CONFIG_DIR="$WACO_HOME/WACO/training_data_generator/config/simulated_matrices"
OUTPUT_DIR="$WACO_HOME/WACO/SpMM/TrainingData/CollectedData/simulated_matrices"


# Input file containing the list of sparse matrices
INPUT_FILE="$WACO_HOME/WACO/SpMM/TrainingData/train_similated.txt"

# Clean up input file to ensure proper formatting
sed -i 's/\r$//' "$INPUT_FILE"  # Remove Windows carriage returns
sed -i '/^$/d' "$INPUT_FILE"    # Remove blank lines
echo >> "$INPUT_FILE"           # Ensure trailing newline

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through each matrix name in the input file
while IFS= read -r matrix_name; do
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

done < "$INPUT_FILE"

echo "Runtime generation completed. Check $OUTPUT_DIR for results and error.log for issues."

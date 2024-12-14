#!/bin/bash

# Paths
MATRIX_LIST="/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/test_part1.txt"
COLLECTED_DATA_DIR="/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SpMM/TrainingData/CollectedData"

# Counter for matches
count=0

# Loop through each matrix name in the list
while read -r matrix_name; do
    if [[ -z "$matrix_name" ]]; then
        continue
    fi

    # Construct the file path
    file_path="$COLLECTED_DATA_DIR/${matrix_name}.txt"
    echo $file_path

    # Check if the file contains the string
    if [[ -f "$file_path" && $(grep -q "SuperSchedule found by WACO" "$file_path") ]]; then
        count=$((count + 1))
    fi
done < "$MATRIX_LIST"

# Print the result
echo "Number of matrices with 'SuperSchedule found by WACO': $count"

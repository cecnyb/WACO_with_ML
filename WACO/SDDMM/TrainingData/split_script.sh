#!/bin/bash

total_file="/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/total.txt"
output_dir="/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData"

# Count total lines
total_lines=$(wc -l < $total_file)
train_lines=$((total_lines * 70 / 100))
validation_lines=$((total_lines * 10 / 100))
test_lines=$((total_lines * 10 / 100))
generalization_lines=$((total_lines - train_lines - validation_lines - test_lines))

echo "Total: $total_lines, Train: $train_lines, Validation: $validation_lines, Test: $test_lines, Generalization: $generalization_lines"


# Split the data
head -n $train_lines $total_file > "$output_dir/train.txt"
tail -n +$((train_lines + 1)) $total_file | head -n $validation_lines > "$output_dir/validation.txt"
tail -n +$((train_lines + validation_lines + 1)) $total_file | head -n $test_lines > "$output_dir/test.txt"
tail -n $generalization_lines $total_file > "$output_dir/generalization.txt"

# Verify the splits
wc -l $output_dir/*.txt

#!/bin/bash

test_file="/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/train.txt"
output_dir="/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData"

total_lines=$(wc -l < $test_file)
subset_size=$((total_lines / 3))
remainder=$((total_lines % 3))

echo "Total lines: $total_lines, Subset size: $subset_size, Remainder: $remainder"

head -n $subset_size $test_file > "$output_dir/test_part1.txt"
tail -n +$((subset_size + 1)) $test_file | head -n $subset_size > "$output_dir/test_part2.txt"
tail -n $((subset_size + remainder)) $test_file > "$output_dir/test_part3.txt"

echo "Splits created:"
wc -l "$output_dir"/test_part*.txt

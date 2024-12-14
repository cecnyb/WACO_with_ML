import os

# Paths
collected_data_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SpMM/TrainingData/CollectedData"
train_file = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/test.txt"
validation_file = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SpMM/TrainingData/validation.txt"

# Output files
filtered_train_file = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/filtered_test.txt"
filtered_validation_file = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SpMM/TrainingData/filtered_validation.txt"

# Get the list of matrix filenames (without extensions) from the directory
matrix_files = {os.path.splitext(f)[0] for f in os.listdir(collected_data_dir)}

def filter_file(input_file, output_file):
    """Filter rows in input_file based on matrix names and write to output_file."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            matrix_name = line.split()[0]  # Assuming the first column is the matrix name
            if matrix_name in matrix_files:
                outfile.write(line)

# Filter train.txt
filter_file(train_file, filtered_train_file)

# Filter validation.txt
filter_file(validation_file, filtered_validation_file)

print(f"Filtered train file written to: {filtered_train_file}")
print(f"Filtered validation file written to: {filtered_validation_file}")

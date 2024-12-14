import os
import numpy as np


input_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/filtered_test.txt"
out_put_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/filtered_test_small.txt"
DATA_DIRECTORY = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SpMM/TrainingData/CollectedData"
TARGET_LINE = "SuperSchedule found by WACO"

waco_prefix = os.getenv("WACO_HOME")
if waco_prefix == None : 
    print("Err : environment variable WACO_HOME is not defined")
    quit() 

SIZE_THRESHOLD = 10000
small_matrices = []

with open(input_dir) as f :

    names = f.read().splitlines() 

    for matrix_name in names:
        filepath = f"{waco_prefix}/dataset/ss_matrix_collection/all_csr_files_binary/{matrix_name}.csr"
        try:
            # Read the matrix header to get dimensions
            with open(filepath, 'rb') as f:
                header = np.fromfile(f, dtype=np.int32, count=3)
                num_row, num_col, nnz = header

                # Check size constraints
                if num_row <= SIZE_THRESHOLD and num_col <= SIZE_THRESHOLD and nnz <= SIZE_THRESHOLD and nnz>0:
                    matrix_file_path = os.path.join(DATA_DIRECTORY, f"{matrix_name}.txt")
                    if os.path.exists(matrix_file_path):
                        with open(matrix_file_path, 'r') as file:
                            for line in file:
                                if TARGET_LINE in line:
                                    small_matrices.append(matrix_name)
                                    break
                        print("skipping, no superschedule found")
                        
                    else: 
                        print("not found in collected: ", matrix_name)

                

                else:   
                    if nnz==0:
                        print(f"Skipping matrix {matrix_name}: nnz is 0")
                    print(f"Skipping matrix {matrix_name}: Exceeds size threshold")

        except FileNotFoundError:
            print(f"Skipping matrix {matrix_name}: File not found")


with open(out_put_dir, 'a') as f:
    for matrix in small_matrices:
        
        f.write(f"{matrix}\n")


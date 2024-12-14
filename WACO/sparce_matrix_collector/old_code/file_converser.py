
import os
import tarfile
import shutil

# Input directory where .tar.gz files are stored
input_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/raw_data"
extracted_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/extracted_matrices"
os.makedirs(extracted_dir, exist_ok=True)  # Ensure the output directory exists

# Extract all .tar.gz files
for filename in os.listdir(input_dir):
    if filename.endswith(".tar.gz"):
        file_path = os.path.join(input_dir, filename)
        print(f"Extracting {filename}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extracted_dir)
        print(f"Extracted {filename} to {extracted_dir}")

# Directory to delete
extracted_dir1 = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/extracted_matrices"
extracted_dir2 = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/raw_data"

# Delete the directory and all its contents
shutil.rmtree(extracted_dir1, ignore_errors=True)  # Removes the directory and everything inside it

os.makedirs(extracted_dir1, exist_ok=True)  # Recreates the directory as empty
print(f"Recreated empty directory: {extracted_dir}")

# Delete the directory and all its contents
shutil.rmtree(extracted_dir2, ignore_errors=True)  # Removes the directory and everything inside it

os.makedirs(extracted_dir2, exist_ok=True)  # Recreates the directory as empty
print(f"Recreated empty directory: {extracted_dir}")


print(f"Deleted directory: {extracted_dir}")
from scipy import sparse, io

import os
from scipy.io import mmread, mmwrite

# Directories
input_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/extracted_matrices"  # Where .mtx files are located
output_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/ss_matrix_csr" 
# Where .csr files will be saved
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Walk through the input directory to process all .mtx files
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".mtx"):
            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_dir, filename.replace(".mtx", ".csr"))
            print(f"Processing {input_path}...")

            # Read the .mtx file
            m = mmread(input_path)

            # Convert to CSR format
            m_csr = m.tocsr()

            # Write the CSR matrix back to a Matrix Market file with .csr extension
            mmwrite(output_path, m_csr)
            print(f"Saved CSR matrix to {output_path}")

        
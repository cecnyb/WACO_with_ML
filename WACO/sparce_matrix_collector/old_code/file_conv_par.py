import os
import tarfile
import shutil
from concurrent.futures import ProcessPoolExecutor
from scipy.io import mmread, mmwrite

# Input and output directories
raw_data_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/raw_data"
extracted_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/extracted_matrices"
csr_output_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/ss_matrix_csr"

# Ensure directories exist
os.makedirs(extracted_dir, exist_ok=True)
os.makedirs(csr_output_dir, exist_ok=True)

# Function to extract a .tar.gz file
def extract_tar_gz(filename):
    file_path = os.path.join(raw_data_dir, filename)
    print(f"Extracting {filename}...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extracted_dir)
    print(f"Extracted {filename} to {extracted_dir}")

# Function to convert .mtx to .csr
def convert_to_csr(filename):
    if filename.endswith(".mtx"):
        input_path = os.path.join(extracted_dir, filename)
        output_path = os.path.join(csr_output_dir, filename.replace(".mtx", ".csr"))
        print(f"Processing {input_path}...")

        # Read the .mtx file
        m = mmread(input_path)

        # Convert to CSR format
        m_csr = m.tocsr()

        # Write the CSR matrix back to a Matrix Market file with .csr extension
        mmwrite(output_path, m_csr)
        print(f"Saved CSR matrix to {output_path}")

# Step 1: Parallelize extraction of .tar.gz files
def extract_all_tar_gz():
    tar_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".tar.gz")]
    with ProcessPoolExecutor() as executor:
        executor.map(extract_tar_gz, tar_files)

# Step 2: Parallelize conversion of .mtx files to .csr
def convert_all_to_csr():
    mtx_files = [f for f in os.listdir(extracted_dir) if f.endswith(".mtx")]
    with ProcessPoolExecutor() as executor:
        executor.map(convert_to_csr, mtx_files)

# Step 3: Cleanup and recreate directories
def cleanup_and_recreate_dirs():
    dirs_to_clean = [extracted_dir]
    for dir_path in dirs_to_clean:
        shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Recreated empty directory: {dir_path}")

# Main function
if __name__ == "__main__":
    print("Starting extraction process...")
    extract_all_tar_gz()

    print("Starting conversion process...")
    convert_all_to_csr()

    print("Cleaning up directories...")
    #cleanup_and_recreate_dirs()

    print("Process completed.")

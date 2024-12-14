import os
from scipy.io import mmread, mmwrite

# Directories
input_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/extracted_matrices"  # Root directory containing subdirectories with .mtx files
output_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/ss_matrix_csr"  # Output directory for .csr files
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Walk through the input directory to locate .mtx files in subdirectories
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".mtx"):
            input_path = os.path.join(root, filename)

            # Preserve subdirectory structure in the output directory
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path.replace(".mtx", ".csr"))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create subdirectories if needed

            print(f"Processing {input_path}...")

            try:
                # Read the .mtx file
                m = mmread(input_path)

                # Convert to CSR format
                m_csr = m.tocsr()

                # Write the CSR matrix to the output directory
                mmwrite(output_path, m_csr)
                print(f"Saved CSR matrix to {output_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

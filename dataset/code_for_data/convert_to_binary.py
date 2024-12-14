import os
from scipy.io import mmread
from scipy.sparse import csr_matrix
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Input and output directories
INPUT_DIR = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/all_csr_files"
OUTPUT_DIR = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/all_csr_files_binary"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_to_binary(input_file, output_dir):
    """
    Converts a MatrixMarket file to binary CSR format and stores it in the output directory.
    """
    try:
        # Extract the filename without the path and extension
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, file_name)
        print("trying to process: ", output_file)

        # Read the MatrixMarket file
        matrix = mmread(input_file)

        # Convert to CSR format (if not already in CSR)
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)

        # Write the CSR matrix to a binary file
        with open(output_file, "wb") as f:
            # Write metadata (rows, cols, nnz)
            np.array([matrix.shape[0], matrix.shape[1], matrix.nnz], dtype=np.int32).tofile(f)
            # Write row pointer, column indices, and values
            matrix.indptr.astype(np.int32).tofile(f)
            matrix.indices.astype(np.int32).tofile(f)
            matrix.data.astype(np.float64).tofile(f)
        f.close()

        print(f"Binary CSR matrix saved to {output_file}")

    except Exception as e:
        print(f"Error converting {input_file}: {e}")


if __name__ == "__main__":
    # Get a list of all .csr files in the input directory
    csr_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".csr")]
    print(f"Number of files to process: {len(csr_files)}")

    # Sequential processing for debugging (uncomment this for initial testing)
    counter = 0
    for file in csr_files:
        try:
            convert_to_binary(file, OUTPUT_DIR)
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    
        counter += 1

    # Parallel processing using ProcessPoolExecutor
   #max_workers = 12  # Adjust based on your system's capabilities
    #with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Partial function to pass the output directory
     #   convert_func = partial(convert_to_binary, output_dir=OUTPUT_DIR)

        # Process all files in parallel
      #  executor.map(convert_func, csr_files)

    print(f"Conversion completed. All binary CSR files are stored in {OUTPUT_DIR}")
    print("loops: ", counter)

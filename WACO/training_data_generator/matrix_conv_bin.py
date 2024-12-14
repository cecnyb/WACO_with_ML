from scipy.io import mmread
from scipy.sparse import csr_matrix
import numpy as np

# Input and output file paths
input_file = "../dataset/ss_matrix_collection/all_csr_files/spiral_E.csr"
output_file = "../dataset/ss_matrix_collection/spiral_E.csr"

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

print(f"Binary CSR matrix saved to {output_file}")



from scipy import io
import numpy as np
from scipy.sparse import random, csr_matrix
import os

def generate_sparse_matrix(num_rows, num_cols, density, random_state=None, pattern=None):
    """
    Generate a sparse matrix in CSR format with specific characteristics.
    
    Args:
        num_rows (int): Number of rows.
        num_cols (int): Number of columns.
        density (float): Proportion of non-zero elements (0 < density <= 1).
        random_state (int): Seed for reproducibility.
        pattern (str): Type of non-zero pattern ('diagonal', 'block', 'random').

    Returns:
        csr_matrix: Sparse matrix in CSR format.
    """
    rng = np.random.default_rng(random_state)

    if pattern == "diagonal":
        rows = np.arange(min(num_rows, num_cols))
        cols = np.arange(min(num_rows, num_cols))
        data = rng.random(len(rows))
        return csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols))

    elif pattern == "block":
        block_size = min(num_rows, num_cols) // 10  # Example block size
        rows = rng.integers(0, num_rows, block_size)
        cols = rng.integers(0, num_cols, block_size)
        data = rng.random(block_size)
        return csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols))

    else:  # "random"
        return random(num_rows, num_cols, density=density, format="csr", random_state=random_state, dtype=np.float32)
    
    
    
def save_csr_matrix(csr, filename, output_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/random"):
    """
    Save a CSR matrix to the specified subdirectory of the WACO dataset directory.

    Args:
        csr (csr_matrix): The sparse matrix to save.
        filename (str): The output filename (without path).
        subdir (str): Subdirectory under WACO_HOME/dataset to save the file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Construct file path
    filepath = os.path.join(output_dir, filename)

    num_row, num_col = csr.shape
    nnz = csr.nnz

    # Prepare header and components
    header = np.array([num_row, num_col, nnz], dtype=np.int32)
    row_ptr = csr.indptr.astype(np.int32)
    col_indices = csr.indices.astype(np.int32)
    data = csr.data.astype(np.float32)

    # Save all components to the same file
    #csr_array = csr_matrix((data, col_indices, row_ptr), shape=(num_row, num_col))
    csr_array = np.concatenate((header, row_ptr, col_indices, data))
    #io.mmwrite(filepath, csr_array)

    
    print("array ", csr_array)


    csr_array.tofile(filepath)

    print(f"Saved CSR matrix to {filepath}")


# Directory for output
output_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/random"

# Generate and save matrices
matrices = [
    ("random_5", generate_sparse_matrix(100, 100, 0.05, random_state=42, pattern="random")),
    ("random_10", generate_sparse_matrix(100, 100, 0.10, random_state=42, pattern="random")),
    ("diagonal", generate_sparse_matrix(100, 100, density=1.0, random_state=42, pattern="diagonal")),
    ("block", generate_sparse_matrix(100, 100, density=0.10, random_state=42, pattern="block")),
]

matrix = random(10, 10, density=0.2, format="csr", dtype=np.float32)

# Print the matrix details
print("Generated matrix:")
print(matrix.toarray()) 

save_csr_matrix(matrix, "test_matrix.csr", output_dir)



print("saved matrix: ")

'''
def load_sparse_csr(filename):
    loader = np.load(filename, allow_pickle=True)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
'''

csr = np.fromfile("/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/random/test_matrix.csr")
print("csr first 100: ", csr[:100])
print("csr shape: ", csr.shape)
num_row,num_col,nnz = csr[0],csr[1],csr[2]
coo = np.zeros((nnz,2),dtype=int)
print("shape for coo: ", coo.shape)





#for name, matrix in matrices:
#    save_csr_matrix(matrix, f"{name}.csr", output_dir)
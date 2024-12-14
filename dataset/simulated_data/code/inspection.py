import numpy as np

def inspect_csr_file(filepath):
    with open(filepath, "rb") as f:
        # Read header
        header = np.fromfile(f, dtype=np.int16, count=3)
        num_row, num_col, nnz = header
        print(f"Header: num_row={num_row}, num_col={num_col}, nnz={nnz}")

        # Read row pointers
        indptr = np.fromfile(f, dtype=np.int16, count=num_row + 1)
        print(f"Row Pointers (indptr): {indptr}")

        # Read column indices
        indices = np.fromfile(f, dtype=np.int16, count=nnz)
        print(f"Column Indices (indices): {indices}")

        # Read data
        data = np.fromfile(f, dtype=np.uint8, count=nnz)
        print(f"Data Values (data): {data}")

# Replace with the actual file path
inspect_csr_file("/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/simulated_data/simulated_matrices/matrix0.csr")

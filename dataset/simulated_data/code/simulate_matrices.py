import numpy as np
from scipy.sparse import random, csr_matrix
import os


WACO_HOME = os.environ.get("WACO_HOME")

def simulate_matrix(num_rows, num_cols, density, random_state):
    sparce_matrix = random(num_rows, num_cols, density=density, format='csr', random_state= random_state)
    #Instead of the uniform values between 0 and 1, change the non-zero elements to integers between 1 and 10. This is less computationally demanding. 
    sparce_matrix.data = np.random.randint(1, 10, size=sparce_matrix.data.shape).astype(np.int32)

   # print("sprace data ", sparce_matrix.data)

    return sparce_matrix


def save_matrix(matrix, matrix_nr, file_path):
    """
    Saves a sparse matrix in dense format as a csr file.

    Parameters:
    - matrix: scipy.sparse.csr_matrix, the sparse matrix to save.
    - file_path: str, path to save the csr file.
    """
    # Convert to dense format
    #dense_matrix = matrix.toarray()
    os.makedirs(file_path, exist_ok=True)

    # Create the full file path
    output_file = os.path.join(file_path, f"matrix{matrix_nr}.csr")

    row_ptr = matrix.indptr
    col_indices = matrix.indices
    nnz = len(col_indices)  # Number of non-zero elements
    num_rows, num_cols = matrix.shape

    # Prepare the CSR structure as per the given example
    csr_array = np.zeros(3 + num_rows + 1 + nnz, dtype=np.int32)
    csr_array[0] = num_rows
    csr_array[1] = num_cols
    csr_array[2] = nnz
    csr_array[3:3 + num_rows + 1] = row_ptr
    csr_array[3 + num_rows + 1:] = col_indices

    # Save to file
    csr_array.tofile(output_file)


    '''
    with open(output_file, "wb") as f:
        # Write metadata (rows, cols, nnz)
        np.array([matrix.shape[0], matrix.shape[1], matrix.nnz], dtype=np.int32
        ).tofile(f)
        # Write row pointer, column indices, and values
        matrix.indptr.astype(np.int32
        ).tofile(f)
        matrix.indices.astype(np.int32
        ).tofile(f)
        matrix.data.astype(np.float32
        ).tofile(f)  
    '''
    print(f"Matrix saved to {output_file}.")



def simulate_sparce_matrices(num_matrices, num_rows_range, num_cols_range, density_range, save_path, random_state = None):
    """
    Simulates multiple sparse matrices with different characteristics.

    Parameters:
    - num_matrices: int, number of matrices to generate.
    - num_rows_range: tuple, range of rows (min_rows, max_rows).
    - num_cols_range: tuple, range of columns (min_cols, max_cols).
    - density_range: tuple, range of density (min_density, max_density).
    - save_path: str, folder path to save the matrices.
    - random_state: int, optional seed for reproducibility.

    Returns:
    - None (saves the matrices as .csv files).
    """

    np.random.seed(random_state)

    for i in range(num_matrices):
        matrix = simulate_matrix(np.random.randint(num_rows_range[0], num_rows_range[1]+1), np.random.randint(num_cols_range[0], num_cols_range[1]+1), np.random.uniform(density_range[0], density_range[1]), random_state)
     #   print("data", matrix.data)
        save_matrix(matrix, i, save_path)


if __name__ == "__main__":
    num_matrices = 2000
    rows_range = (50, 200)        # Min and max rows
    cols_range = (50, 200)        # Min and max columns
    density_range = (0.01, 0.05)
    random_state = 42

    # Simulate a sparse matrix
    
    sparse_matrix = simulate_sparce_matrices(1, rows_range, cols_range, density_range, f"{WACO_HOME}/dataset/simulated_data/simulated_matrices", random_state)


#This is just for checking that the matrices have the right format 
'''
    file_path1 = '/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/simulated_data/simulated_matrices/matrix0.csr'
    file_path ='/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/dataset/ss_matrix_collection/EX1_8x8_4.csr'

    csr = np.fromfile(file_path, dtype='<i4')
 
    num_row,num_col,nnz = csr[0],csr[1],csr[2]

    


    coo = np.zeros((nnz,2),dtype=int)
    print("shape for coo: ", coo.shape)
    print("size of csr[3+num_row+1:]: ", csr[3+num_row+1:].shape)
    coo[:,1] = csr[3+num_row+1:]
    bins = np.array(csr[4:num_row+4]) - np.array(csr[3:num_row+3])
    coo[:,0] = np.repeat(range(num_row), bins)

    csr = np.fromfile(file_path1, dtype='<i4')
    print("csr ", csr[0:100])
    num_row,num_col,nnz = csr[0],csr[1],csr[2]


    coo = np.zeros((nnz,2),dtype=int)
    print("shape for coo: ", coo.shape)
    print("size of csr[3+num_row+1:]: ", csr[3+num_row+1:].shape)
    coo[:,1] = csr[3+num_row+1:]
    bins = np.array(csr[4:num_row+4]) - np.array(csr[3:num_row+3])
    coo[:,0] = np.repeat(range(num_row), bins)
    



    with open(file_path, 'rb') as f:
      # Read header: num_row, num_col, nnz (all uint8)
        header = np.fromfile(f, dtype=np.int32
        , count=3)
        num_row, num_col, nnz = header

        # Read row pointers (indptr): size = num_row + 1
        indptr = np.fromfile(f, dtype=np.int32
        , count=num_row + 1)
        #print("Row Pointers (indptr):", indptr)

        # Read column indices: size = nnz
        indices = np.fromfile(f, dtype=np.int32
        , count=nnz)
        # print("Column Indices:", indices)

        # Read data (values): size = nnz
        data = np.fromfile(f, dtype=np.int32
        , count=nnz)
        #print("Data Values:", data)
    
    # Convert to COO format for inspection or return as CSR
    #bins =  np.diff(indptr)
    coo_rows = np.repeat(np.arange(num_row), np.diff(indptr))
    coo_cols = indices
    #coo_data = data

    coo = np.vstack((coo_rows, coo_cols)).T 


    print("num_rows and all that ", num_row, num_col, indices, nnz)
'''
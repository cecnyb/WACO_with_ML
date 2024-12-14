import os
import tarfile
import shutil
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor


input_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/extracted_matrices"
output_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/all_csr_files"
os.makedirs(output_dir, exist_ok = True)

def convert_matrix(filename):
    if filename.endswith('mtx'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename).replace('.mtx', '.csr')

        try:
            m = mmread(input_path)

            if not hasattr(m, 'tocsr'):
                # Convert a dense numpy.ndarray to a sparse CSR matrix
                m = csr_matrix(m)

            m_csr = m.tocsr()

            mmwrite(output_path, m_csr)
            print(f"Saved CSR matrix to {output_path}")
            os.remove(input_path)
        except Exception as e:
            print(f"Failed to process {input_path}: {e}")


if __name__ == "__main__":
    file_names = [f for f in os.listdir(input_dir) if f.endswith('.mtx')]

    with ProcessPoolExecutor(max_workers=20) as executor: 
        executor.map(convert_matrix, file_names)
    




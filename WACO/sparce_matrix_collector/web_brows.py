import ssgetpy
from ssgetpy import search, fetch
from scipy.io import mmread
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor
import os

'''
# Directory to store the downloaded matrices
output_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/raw_data" #".../dataset/ss_matrix_collection" 
os.makedirs(output_dir, exist_ok=True)

# Retrieve the list of all matrices in the SuiteSparse collection
print("Fetching the list of matrices from SuiteSparse...")
matrices = ssgetpy.search(limit=2893)
matrices.download(format="MM", destpath=output_dir)
'''



output_dir = "/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/raw_data" #".../dataset/ss_matrix_collection" 
os.makedirs(output_dir, exist_ok=True)



def download_matrix(matrix_id, output_dir):
    import ssgetpy
    try:
        ssgetpy.fetch(matrix_id)
    except Exception as e:
        print(f"Error downloading matrix {matrix_id}: {e}")
  


def download_matrices(start_id, end_id, output_dir, max_workers=15):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_matrix, matrix_id, output_dir): matrix_id
            for matrix_id in range(start_id, end_id)
        }
        for future in futures:
            matrix_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading matrix {matrix_id}: {e}")



if __name__ == "__main__":
    download_matrices(0, 2893, output_dir)


#cp -r /home/s0/ml4sys02/.ssgetpy/MM/vanHeukelum/* /home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/raw_data
#/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/extracted_matrices
#mkdir -pmkdir 

#find . -type f -name "*.mtx" | wc -l
#find . -type f -name "*.mtx" -exec mv {} /home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/extracted_matrices/ \;


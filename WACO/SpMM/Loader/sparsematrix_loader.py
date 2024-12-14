
import torch
import numpy as np
import MinkowskiEngine as ME
import os


def from_csr_old(filename) :
  waco_prefix = os.getenv("WACO_HOME")
  if waco_prefix is None : 
    print("Err : environment variable WACO_HOME is not defined")
    return 
  csr = np.fromfile(waco_prefix+"/dataset/ss_matrix_collection/all_csr_files_binary/"+filename+".csr", dtype='<i4')
  
  #csr = np.fromfile("/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/dataset/ss_matrix_collection/EX1_8x8_4.csr", dtype='<i4')
  #csr = np.fromfile("/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/random/block.csr")
  print("csr first 100: ", csr[:100])
  print("csr shape: ", csr.shape)
  num_row,num_col,nnz = csr[0],csr[1],csr[2]
  coo = np.zeros((nnz,2),dtype=int)
  print("shape for coo: ", coo.shape)
  print("size of csr[3+num_row+1:]: ", csr[3+num_row+1:].shape)
  coo[:,1] = csr[3+num_row+1:]
  print("coo done")
  #coo[:, 1] = csr[3+num_row+1:]
  bins = np.array(csr[4:num_row+4]) - np.array(csr[3:num_row+3])
  coo[:,0] = np.repeat(range(num_row), bins)
  print("coo: ", coo[:100])

  return num_row, num_col, nnz, coo


def from_csr_old2(filename):

 

  waco_prefix = os.getenv("WACO_HOME")
  
  if waco_prefix is None : 
    print("Err : environment variable WACO_HOME is not defined")
    return 
  
  filepath = waco_prefix+"/dataset/ss_matrix_collection/all_csr_files_binary/"+filename+".csr"

  with open(filepath, 'rb') as f:
      # Read header: num_row, num_col, nnz (all int32)
      header = np.fromfile(f, dtype=np.int32, count=3)
      num_row, num_col, nnz = header

      # Read row pointers (indptr): size = num_row + 1
      indptr = np.fromfile(f, dtype=np.int32, count=num_row + 1)
      #print("Row Pointers (indptr):", indptr)

      # Read column indices: size = nnz
      indices = np.fromfile(f, dtype=np.int32, count=nnz)
     # print("Column Indices:", indices)

      # Read data (values): size = nnz
      data = np.fromfile(f, dtype=np.float64, count=nnz)
      #print("Data Values:", data)
    

    # Convert to COO format for inspection or return as CSR
  #bins =  np.diff(indptr)
  coo_rows = np.repeat(np.arange(num_row), np.diff(indptr))
  coo_cols = indices
  #coo_data = data

  coo = np.vstack((coo_rows, coo_cols)).T 

  '''
  The above is equivalent to:
  bins = np.diff(indptr)
  print("Bins (non-zero counts per row):", bins)

  # Initialize COO matrix (row indices, column indices)
  coo = np.zeros((nnz, 2), dtype=int)

  # Fill column indices
  coo[:, 1] = indices

  # Fill row indices
  coo[:, 0] = np.repeat(np.arange(num_row), bins)
  '''
  
  ''' The belove is for printing purpose only:
  # Reconstruct the equivalent of csr
  csr_reconstructed = np.concatenate([
      header,        # [num_row, num_col, nnz]
      indptr,        # Row pointers
      indices,       # Column indices
      data.astype(np.int32)  # Data (cast to match dtype of original csr)
  ])

  # Inspect the first 100 elements
  print("First 100 elements of reconstructed csr:", csr_reconstructed[:100])

  combined_size = indices.size #+ data.size  # This corresponds to csr[3+num_row+1:].size
  print(f"Combined size of indices and data: {combined_size}")

  # COO shape
  print("Shape for coo:", coo.shape)

  # Check if they match
  if combined_size == coo.shape[0]:
      print("Shapes match!")
  else:
      print("Shapes do not match!")

  #print("coo: ", coo[:100])

  '''

  return num_row, num_col, nnz, coo


def from_csr(filename) :
  waco_prefix = os.getenv("WACO_HOME")
  if waco_prefix is None : 
    print("Err : environment variable WACO_HOME is not defined")
    return 

  csr = np.fromfile(waco_prefix+"/dataset/simulated_data"+filename+".csr", dtype='<i4')
  num_row,num_col,nnz = csr[0],csr[1],csr[2]
  coo = np.zeros((nnz,2),dtype=int)
  coo[:,1] = csr[3+num_row+1:]
  bins = np.array(csr[4:num_row+4]) - np.array(csr[3:num_row+3])
  coo[:,0] = np.repeat(range(num_row), bins)
  return num_row, num_col, nnz, coo


def collate_fn(list_data):
    coords_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
    )

    mtxnames_batch = [d["mtxname"] for d in list_data]
    shapes_batch = torch.stack([d["shape"] for d in list_data]) 

    return mtxnames_batch, coords_batch, features_batch, shapes_batch

class SparseMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
      waco_prefix = os.getenv("WACO_HOME")
      if waco_prefix == None : 
        print("Err : environment variable WACO_HOME is not defined")
        quit() 
      
      with open(filename) as f :
        self.names = f.read().splitlines() 


      # Preparing Data
      self.standardize = {}
      self.normalize = {}
      with open("/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/WACO/SDDMM/TrainingData/filtered_train.txt") as f :
        total_rows, total_cols, total_nnzs = [], [], []

        for filename in f.read().splitlines() :
          
          try:
            '''
            csr = np.fromfile(waco_prefix+"/dataset/ss_matrix_collection/all_csr_files_binary/"+filename+".csr", count=3, dtype='<i4')
            total_rows.append(csr[0])
            total_cols.append(csr[1])
            total_nnzs.append(csr[2])
            ''' 

            filepath = waco_prefix+"/dataset/ss_matrix_collection/all_csr_files_binary/"+filename+".csr"
            
            with open(filepath, 'rb') as f:
              # Read header: num_row, num_col, nnz (all int32)
              header = np.fromfile(f, dtype=np.int32, count=3)
              num_row, num_col, nnz = header

              print(f"Matrix dimensions: {num_row}x{num_col}, Non-Zeros: {nnz}")

              
              total_rows.append(num_row)
              total_cols.append(num_col)
              total_nnzs.append(nnz)
          
          except FileNotFoundError:
            print(f"File not found: {waco_prefix}/dataset/ss_matrix_collection/all_csr_files_binary/{filename}.csr")

        self.standardize["mean_rows"] = np.mean(total_rows)
        self.standardize["mean_cols"] = np.mean(total_cols)
        self.standardize["mean_nnzs"] = np.mean(total_nnzs)
        self.standardize["std_rows"] = np.std(total_rows)
        self.standardize["std_cols"] = np.std(total_cols)
        self.standardize["std_nnzs"] = np.std(total_nnzs)
    
    def __len__(self):
      return len(self.names)


    def __getitem__(self, idx):
      filename = self.names[idx]

      result = from_csr(filename)

      # Unpack the result
      num_row, num_col, nnz, coo = result
      #num_row, num_col, nnz, coo = from_csr(filename)
      
      # standardize
      num_row = (num_row - self.standardize["mean_rows"])/self.standardize["std_rows"]
      num_col = (num_col - self.standardize["mean_cols"])/self.standardize["std_cols"]
      nnz     = (nnz - self.standardize["mean_nnzs"])/self.standardize["std_nnzs"]
      
      # To ME Sparse Tensor
      coordinates = torch.from_numpy(coo).to(torch.int32)
      features = torch.ones((len(coo),1)).to(torch.float32)
      label = torch.tensor([[0]]).to(torch.float32)
      shape = torch.tensor([num_row, num_col, nnz]).to(torch.float32)

      return {
        "mtxname" : filename, 
        "coordinates" : coordinates, 
        "features" : features, 
        "label" : label, 
        "shape" : shape 
      }




import random
import os
import itertools
import time
import sys
import math
import numpy as np
import argparse
#from concurrent.futures import ProcessPoolExecutor
import os

WACO_HOME = os.environ.get("WACO_HOME")

result = ""
currinput = ""
num_row = 0
num_col = 0
num_nonzero = 0


def filter(l, j):
    i1 = l.index('i1')
    k1 = l.index('k1')
    j1 = l.index('j1')
    i0 = l.index('i0')
    k0 = l.index('k0')
    j0 = l.index('j0')

    if j < 8:
        lastj = j1 == 4 and j0 == 5
    else:
        lastj = j0 == 5

    return (i1 < i0) and (k1 < k0) and (j1 < j0) and lastj and (j1 > i1)

def process_file(file_name, input_dir):
    print("processing file: ", file_name)
    print("input_dir: ", input_dir)
    csr = np.fromfile(os.path.join(input_dir, file_name), count=3, dtype='<i4')
    num_row, num_col, num_nonzero = csr[0], csr[1], csr[2]

    cfgs = set()
    cfg = {}
    while len(cfgs) < 32:
        cfg['isplit'] = random.choice([1, 2, 4, 8, 16, 32])
        cfg['ksplit'] = random.choice([1, 2, 4, 8, 16, 32])
        cfg['jsplit'] = random.choice([1 << p for p in range(int(math.log(256, 2)))])
        cfg['rankorder'] = ['i1', 'k1', 'j1', 'i0', 'k0', 'j0']
        cfg['i1f'] = random.choice([0, 1])
        cfg['i0f'] = 1
        cfg['k1f'] = 0
        cfg['k0f'] = 1
        cfg['paridx'] = random.choice(['i1'])
        cfg['parnum'] = random.choice([48])
        cfg['parchunk'] = random.choice([1 << p for p in range(9)])
        isplit, ksplit, jsplit = cfg['isplit'], cfg['ksplit'], cfg['jsplit']
        rankorder = " ".join(cfg['rankorder'])
        i1f, i0f, k1f, k0f = cfg['i1f'], cfg['i0f'], cfg['k1f'], cfg['k0f']
        paridx, parnum, parchunk = cfg['paridx'], cfg['parnum'], cfg['parchunk']
        cfgs.add(f"{isplit} {ksplit} {jsplit} {rankorder} {i1f} {i0f} {k1f} {k0f} {paridx} {parnum} {parchunk}\n")

    while len(cfgs) < 100:
        cfg['isplit'] = random.choice([1, 2, 4, 8, 16, 32])
        cfg['ksplit'] = random.choice([1, 2, 4, 8, 16, 32])
        cfg['jsplit'] = random.choice([1 << p for p in range(int(math.log(256, 2)))])
        cfg['rankorder'] = random.choice([
            p for p in list(itertools.permutations(['i1', 'i0', 'k1', 'k0', 'j1', 'j0']))
            if filter(p, cfg['jsplit'])
        ])
        cfg['i1f'] = random.choice([0, 1])
        cfg['i0f'] = random.choice([0, 1])
        cfg['k1f'] = random.choice([0, 1])
        cfg['k0f'] = random.choice([0, 1])
        cfg['paridx'] = random.choice(['i1'])
        cfg['parnum'] = random.choice([48])
        cfg['parchunk'] = random.choice([1 << p for p in range(9)])
        isplit, ksplit, jsplit = cfg['isplit'], cfg['ksplit'], cfg['jsplit']
        rankorder = " ".join(cfg['rankorder'])
        i1f, i0f, k1f, k0f = cfg['i1f'], cfg['i0f'], cfg['k1f'], cfg['k0f']
        paridx, parnum, parchunk = cfg['paridx'], cfg['parnum'], cfg['parchunk']
        cfgs.add(f"{isplit} {ksplit} {jsplit} {rankorder} {i1f} {i0f} {k1f} {k0f} {paridx} {parnum} {parchunk}\n")

    # Write the configurations to a file
    file_path = f"{WACO_HOME}/WACO/training_data_generator/config/simulated_matrices/{file_name.replace('.csr', '')}.txt"
    f = open(file_path, 'w')
    #print("./config/{}.txt".format(file_name.replace(".csr", "")))
    for sched in cfgs : 
        f.write(sched)
    f.close()



if __name__ == '__main__':
    # Parse command-line arguments
   # parser = argparse.ArgumentParser(description="Generate Superschedules for .csr files in a given directory.")
   # parser.add_argument("--input_dir", required=False, help="Path to the input directory containing .csr files")
  #  args = parser.parse_args()

  #  input_dir = args.input_dir
   # if not os.path.isdir(input_dir):
   #     input_dir = '/home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/simulated_data/simulated_matrices'

    input_dir = f"{WACO_HOME}/dataset/simulated_data/simulated_matrices"
    # Get the list of files
    file_names = [file for file in os.listdir(input_dir) if file.endswith('.csr')]

    for file_name in file_names:
        print(file_name)
        process_file(file_name, input_dir)


    file_name_list = f"{WACO_HOME}/WACO/SpMM/TrainingData/train_similated.txt"
    with open(file_name_list, 'w') as f:
        for name in file_names:
            cleaned_name = os.path.splitext(name)[0]
            f.write(cleaned_name + '\n')


    

    # Process files in parallel
    #with ProcessPoolExecutor(max_workers=20) as executor:
     #   executor.map(process_file, file_names, itertools.repeat(input_dir))

   

#python SpMM_SuperSchedule_Generator_Par.py --input_dir /home/s0/ml4sys02/project/Workload-Aware-Co-Optimization/dataset/ss_matrix_collection/all_csr_files 
    
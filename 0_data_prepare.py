from glob import glob
file_list = glob('data/*.fa')
from sklearn.externals import joblib
import numpy as np
base2vec = {'A': np.array([1,0,0,0]), 'T': np.array([0,1,0,0]),'G': np.array([0,0,1,0]),'C': np.array([0,0,0,1])}
for file_idx, file in enumerate(file_list):
    if file_idx % 120 == 0:
        print('{:>3}% is done.'.format(int(file_idx/120)))
    sample_id = file.split('/')[1].split('_')[0]
    ouput_file = file.split('.')[0] + '.pkl'
    with open(file, 'r') as f:
        c = 0
        for line in f:
            if line[0] == '>':
                c += 1
    output = np.zeros((c, 125, 4))
    with open(file, 'r') as f:
        c = 0
        for line in f:
            if line[0] != '>':
                seq = line.replace('\n', '')
                for base_idx, base in enumerate(seq):
                    output[c, base_idx, :] = base2vec[base]
                c += 1
    joblib.dump(output, ouput_file)

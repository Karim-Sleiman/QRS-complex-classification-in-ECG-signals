import os
import wfdb
import numpy as np

data_dir = 'X/polymorph' 
data_dir_npz = 'X/SIMX' 
filename = 'I19'

# 462600 is max sampto
temp = 240000
record = wfdb.rdrecord(os.path.join(data_dir, filename), sampfrom=temp, sampto=15000 + temp)
lead_names = ['I', 'II', 'III', 'AVR', 
              'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
data = {}

for i in range(12):
    lead_data = record.p_signal[:, i]
    lead_label = record.sig_name[i]
    data[f'{lead_names[i]}'] = lead_data 
    data[f'{lead_names[i]}_label'] = lead_label
    
np.savez(os.path.join(data_dir_npz, f"{filename}_sliced-on_{temp}" + '.npz'), **data)
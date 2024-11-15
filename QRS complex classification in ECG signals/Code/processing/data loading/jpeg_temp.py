import os
import wfdb
import matplotlib.pyplot as plt

data_dir= 'X/polymorph' 
data_dir_good= 'X/polymorph/good data' 
filename = 'I01'

#462600 is max sampto
temp=5900
record = wfdb.rdrecord(os.path.join(data_dir, filename), sampfrom=temp, sampto=15000 + temp)

lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

fig, axs = plt.subplots(nrows=12, ncols=1, figsize=(200,100))

for i, ax in enumerate(axs):
    ax.plot(record.p_signal[:, i]) 
    ax.set_title(f'{lead_names[i]} - Lead {i+1}') 

plt.savefig(os.path.join(data_dir, f"{filename}_sliced-on_{temp}" + '_plot.jpg'))
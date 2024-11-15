import os
import wfdb
import matplotlib.pyplot as plt

data_dir= 'X/polymorph' 
filename = 'I75'

#462600 is max sampto: 30 min * 60 sec * 257 Hz = N of samples
record = wfdb.rdrecord(os.path.join(data_dir, filename))
QRS_SAMPLES = 15000
start_sample = 0
end_sample = 450000

for i in range(start_sample, end_sample, QRS_SAMPLES):
    j = i + QRS_SAMPLES
    if j > end_sample:
        j = end_sample 
    slice_filename = f"{filename}_slice_{i}_{j}.png"
    fig, axs = plt.subplots(nrows=12, ncols=1, figsize=(200,100))

for k, ax in enumerate(axs):
    ax.plot(record.p_signal[i:j, k]) 
    ax.set_title(f'Lead {k+1}')
    
plt.savefig(os.path.join(data_dir, slice_filename))
plt.close()
import os
import numpy as np
import matplotlib.pyplot as plt

#from scipy import signal
from scipy.signal import savgol_filter

input_dir = 'X/polymorph/npz-data'
output_dir = 'X/preprocessed-data-synchro'
def preprocess_data(data):
    # Savitzky-Golay smoothing filter
    smoothed_data = {}
    for lead, signal_data in data.items():
        smoothed_signal = savgol_filter(signal_data, window_length=51, polyorder=6)
        smoothed_data[lead] = smoothed_signal

    # Normalize the signals
    preprocessed_data = {}
    for lead, signal_data in smoothed_data.items():
        normalized_signal = (signal_data - np.mean(signal_data)) / np.std(signal_data)
        preprocessed_data[lead] = normalized_signal
    return preprocessed_data

# Iterate over all NPZ files in the input directory
for filename in os.listdir(input_dir): 
    if filename.endswith('.npz'):
        # Load the ECG signals from the NPZ file
        data = np.load(os.path.join(input_dir, filename))
        # Preprocess the ECG data
        preprocessed_data = preprocess_data(data)
        # Save the preprocessed data to the same NPZ file in the output directory
        output_filename = os.path.join(output_dir, filename)
        np.savez(output_filename, **preprocessed_data)

# Plot the preprocessed data of the first file
data = np.load(os.path.join(input_dir, os.listdir(input_dir)[0]))
preprocessed_data = preprocess_data(data)
fig, axs = plt.subplots(nrows=12, ncols=1, figsize=(200, 100))

for i, lead in enumerate(preprocessed_data.keys()):
    axs[i].plot(preprocessed_data[lead][:15000]) 
    axs[i].set_title(f'{lead} - Lead {i+1}')
    axs[i].set_xlim([0, 15000])

plt.tight_layout() 
plt.savefig(os.path.join(output_dir, 'preprocessed_data_plot.png'))
plt.show()

print("Preprocessing completed successfully.")
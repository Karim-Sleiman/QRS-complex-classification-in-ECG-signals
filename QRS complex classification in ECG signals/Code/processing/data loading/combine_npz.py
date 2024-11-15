import os
import numpy as np

def combine_npz_files(folder_path, output_file): 
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    combined_arrays = {}
    for file in npz_files:
        data = np.load(os.path.join(folder_path, file))
    for key in data.files:
        if key in combined_arrays:
            combined_arrays[key] = np.concatenate((combined_arrays[key], data[key]))
        else:
            combined_arrays[key] = data[key]

    np.savez(output_file, **combined_arrays) 
    print(f"Combined NPZ files saved as {output_file}")

folder_path = "X/npz-data"
output_file = "X/combined-npz"

combine_npz_files(folder_path, output_file)
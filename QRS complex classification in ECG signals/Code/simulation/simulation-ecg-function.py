import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pywt


# Real data:
column_index = 11 # leads 0 to 11
start_sample = 15220 # Specify the starting sample index
end_sample = 16600 # Specify the ending sample index
figsize = (10, 6)
df = pd.read_csv(r'X\monomorph\PVCVTECGData\1067472.csv') 
output_directory = 'E:/5th year/pfe/simulation task/123'
df_col = df.iloc[start_sample:end_sample, column_index]
plt.figure(figsize=figsize)
plt.plot(df_col)
#save_path = os.path.join(output_directory, 'ecg_plot_real_V6_1067472.png')
#plt.savefig(save_path)
#plt.close()
plt.show()

def p_wav(x, a, d1, d2, t, li, start, finish, p):
    wave = np.zeros_like(x)
    x1 = x[(x >= start) & (x <= t)]
    x2 = x[(x > t) & (x <= finish)]
    wave[(x >= start) & (x <= finish)] = np.concatenate((
    a * np.exp(-((x1 - t) / d1) ** 2) * np.sin(2 * np.pi * li * x1 + p),
    a * np.exp(-((x2 - t) / d2) ** 2) * np.sin(2 * np.pi * li * x2 + p)
    ))
    return wave

def q_wav(x, a, d, t, li, start, finish, p):
    wave = np.zeros_like(x)
    wave[(x >= start) & (x <= finish)] = -a * np.exp(-((x[(x >= start) & (x <= finish)] - t) / d) ** 2) * np.sin(2 * np.pi * li * x[(x >= start) & (x <= finish)] + p)
    return wave

def qrs_wav(x, a, d, li, start, finish, p):
    wave = np.zeros_like(x)
    wave[(x >= start) & (x <= finish)] = a * np.exp(-((x[(x >= start) & (x <= finish)] - 0.166) / d) ** 2) * np.sin(2 * np.pi * li * x[(x >= start) & (x <= finish)] + p)
    return wave

def s_wav(x, a, d, t, li, start, finish, p):
    wave = np.zeros_like(x)
    wave[(x >= start) & (x <= finish)] = -a * np.exp(-((x[(x >= start) & (x <= finish)] - t) / d) ** 2) * np.sin(2 * np.pi * li * x[(x >= start) & (x <= finish)] + p)
    return wave

def t_wav(x, a, d1, d2, t, li, start, finish, p):
    wave = np.zeros_like(x)
    x1 = x[(x >= start) & (x <= t)]
    x2 = x[(x > t) & (x <= finish)]
    wave[(x >= start) & (x <= finish)] = np.concatenate((
    a * np.exp(-((x1 - t) / d1) ** 2) * np.sin(2 * np.pi * li * x1 + p),
    a * np.exp(-((x2 - t) / d2) ** 2) * np.sin(2 * np.pi * li * x2 + p)
    ))
    return wave

def u_wav(x, a, d, t, li, start, finish, p):
    wave = np.zeros_like(x)
    wave[(x >= start) & (x <= finish)] = a * np.exp(-((x[(x >= start) & (x <= finish)] - t) / d) ** 2) * np.sin(2 * np.pi * li * x[(x >= start) & (x <= finish)] + p)
    return wave

def add_noise(signal, mean=0, std=0.01):
    noise = np.random.normal(mean, std, len(signal))
    return signal + noise

def zero_wave():
    return 0

def exponential_wave(x, a, b, start, finish):
    wave = np.zeros_like(x)
    wave[(x >= start) & (x <= finish)] = a * np.exp(b * (x[(x >= start) & (x <= finish)]))
    return wave

def calculate_signal_error(directory):
    # Get all CSV files in the directory 
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    # Ensure at least two CSV files are present
    if len(csv_files) < 2: 
        print("Error: At least two CSV files are required in the directory.")
        return  -1
    # Read the first two CSV files with specified encoding 
    df1 = pd.read_csv(csv_files[0], encoding='utf-8') 
    df2 = pd.read_csv(csv_files[1], encoding='utf-8')
    # Extract the signal values from each dataframe signal1 = df1['Signal'].values signal2 = df2['Signal'].values
    # Check if signal arrays have different lengths
    if len(signal1) != len(signal2):
        # Resample or interpolate the signal with shorter length
        if len(signal1) < len(signal2):
            f = interp1d(np.arange(len(signal1)), signal1)
            signal1 = f(np.linspace(0, len(signal1)-1, len(signal2)))
        else:
            f = interp1d(np.arange(len(signal2)), signal2)
            signal2 = f(np.linspace(0, len(signal2)-1, len(signal1)))

    # Calculate the root-mean-square error (RMSE)
    mse = np.mean((signal1 - signal2) ** 2)
    rmse = np.sqrt(mse)

    # Calculate the error percentage
    max_value = np.max([np.max(signal1), np.max(signal2)])
    error_percentage = (rmse / max_value) * 100
    # Print the error percentage 
    print(f"Error Percentage: {error_percentage}%")

def least_squares(wave):
    n = len(wave)
    t = np.arange(n)
    # Calculate the coefficients using the method of least squares
    A = np.vstack([t, np.ones(n)]).T
    m, c = np.linalg.lstsq(A, wave, rcond=None)[0]
    # Generate the modified wave using the obtained coefficients
    modified_wave = wave - (m * t + c)
    return modified_wave

def smooth_ecg_signal(signal):
    # Choose the wavelet function 
    wavelet = 'db4'
    
    # Set the number of decomposition levels
    num_levels = 6
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=num_levels)
    
    # Apply thresholding to the detail coefficients
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log2(len(signal))) 
    coeffs = [pywt.threshold(detail, value=threshold, mode='soft') if idx == num_levels else detail for idx, detail in enumerate(coeffs)]
    
    # Reconstruct the signal using the modified coefficients
    reconstructed_signal = pywt.waverec(coeffs, wavelet)
    return reconstructed_signal

x_pwav = np.arange(0.1, 0.6, 0.01)
x_qwav = np.arange(0.2, 0.67, 0.01)
x_qrswav = np.arange(0.01, 0.415, 0.01)
x_swav = np.arange(0.26, 0.7405, 0.01)
x_twav = np.arange(0.001, 1, 0.01)
x_uwav = np.arange(0.01, 3, 0.01)
x_e = np.arange(0, 0.6, 0.01)
li = 30 / 72
a_pwav = 2000
d1_pwav = 0.14
d2_pwav = 0.22
t_pwav = 0.25
phase_p =-0.05
a_qwav = 500
d_qwav = 0.036
t_qwav = 0.65
phase_q =0
a_qrswav = 46000
d_qrswav = 0.045
phase_r = 0
a_swav = 2000
d_swav = 0.066
t_swav = 0.2
phase_s = 0
a_twav = 6000
d1_twav = 0.4

d2_twav = 0.2
t_twav = 0.25
phase_t = 0
a_uwav = 0.035
d_uwav = 0.1
t_uwav = 0.433
phase_u = 0
e_a = 100
e_b = 4.9
start = 0
finish = 3

pwav = p_wav(x_pwav, a_pwav + 0.05, d1_pwav, d2_pwav, t_pwav, li, start, finish, phase_p)
qwav = q_wav(x_qwav, a_qwav, d_qwav, t_qwav - 0.1, li, 0.3, 0.75, phase_q)
qrswav = qrs_wav(x_qrswav, a_qrswav, d_qrswav, li, start, finish,phase_r)
swav = s_wav(x_swav, a_swav - 0.1, d_swav, t_qwav - 0.27, li, start, finish,phase_s)
twav = t_wav(x_twav, a_twav - 0.1, d1_twav, d2_twav, t_twav, li, start, finish,phase_t)
uwav = u_wav(x_uwav, a_uwav, d_uwav, t_qwav + 0.6, li, start, finish, phase_u) 

# Increase the distance between T and U wave
ewav = exponential_wave(x_e, e_a, e_b, start, finish)

# Calculate the common time axis
common_time = np.linspace(0, 3, 1400)

# Interpolate or resample each wave onto the common time axis
pwav_resampled = np.interp(common_time, x_pwav, pwav)
qwav_resampled = np.interp(common_time, x_qwav + 0.22, qwav)
qrswav_resampled = np.interp(common_time, x_qrswav + t_qwav + 0.16, qrswav)
swav_resampled = np.interp(common_time, x_swav + t_qwav + 0.05, swav)
twav_resampled = np.interp(common_time, x_twav + t_qwav + 1.1, twav)
uwav_resampled = np.interp(common_time, x_uwav + t_qwav + 1.4, uwav)
ewav_resampled = np.interp(common_time, x_e + t_qwav + 0.7, ewav)

# Calculate the Total signal
Total_1 = pwav_resampled + qwav_resampled + qrswav_resampled + swav_resampled + twav_resampled + uwav_resampled + ewav_resampled

# Least mean squares
Total_no_noise = least_squares(Total_1)

# work on an elevation
Total_no_noise += 1000

# Apply wavelet transform 5 times
Temp1 = smooth_ecg_signal(Total_no_noise)
Temp2 = smooth_ecg_signal(Temp1)
Temp3 = smooth_ecg_signal(Temp2)
Temp4 = smooth_ecg_signal(Temp3)
Temp5 = smooth_ecg_signal(Temp4)

# Add noise to the signals
Total = add_noise(Temp5)

# Plot the ECG signal with noise
plt.figure(figsize=(10, 6))

#plt.plot(x_pwav, pwav, color='cyan')
#plt.plot(x_qwav + 0.2, qwav, color='cyan')
#plt.plot(x_qrswav + t_qwav + 0.2 , qrswav, color='cyan')
#plt.plot(x_swav + t_qwav + 0.35, swav, color='cyan')
#plt.plot(x_twav + t_qwav + 1.1, twav, color='cyan')
#plt.plot(range(len(ewav_resampled)), ewav_resampled, color ='red')

plt.plot(range(len(Total)), Total, color='cyan')
plt.xlabel('Samples') 
plt.ylabel('Amplitude') 
plt.title('ECG Signal with Noise')
#plt.show()
# Save the plot as a PNG file 
save_path = os.path.join(output_directory, 'ecg_plot_sim.png')
plt.savefig(save_path)
plt.close()

# Save the simulated ECG signals as CSV files 
simulated_signals = {'Samples': range(len(Total)), 'Signal': Total}
df_simulated = pd.DataFrame(simulated_signals) 
csv_path = os.path.join(output_directory, 'ecg_signal_sim.csv')
df_simulated.to_csv(csv_path, index=False)

# Save the real ECG signal as a CSV file 
real_signal = {'Samples': range(len(df_col)), 'Signal': df_col}
df_real = pd.DataFrame(real_signal) 
csv_path = os.path.join(output_directory, 'ecg_signal_real_V6_1067472.csv')
df_real.to_csv(csv_path, index=False)

calculate_signal_error(output_directory)
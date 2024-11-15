import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

output_directory = 'E:/X/simulation task'

def rrprocess(flo=None, fhi=None, flostd=None, fhistd=None, lfhfratio=None, hrmean=None, hrstd=None, sampfreq=None, N=None):
    if flo is None:
        flo = 0.1
    if fhi is None:
        fhi = 0.25
    if flostd is None:
        flostd = 0.01
    if fhistd is None:
        fhistd = 0.01
    if lfhfratio is None:
        lfhfratio = 0.5
    if hrmean is None:
        hrmean = 60
    if hrstd is None:
        hrstd = 1
    if sampfreq is None:
        sampfreq = 1
    if N is None:
        N = 10000

    # Calculate number of low frequency components and number of high frequency components
    nlf = int(np.ceil(lfhfratio / (1 + lfhfratio) * N))
    nhf = N - nlf

    # Generate uniformly distributed random numbers
    U = np.random.rand(nlf) * (fhi - flo) + flo
    V = np.random.randn(nlf) * flostd
    X = U + V
    # Generate normal distributed random numbers
    Y = np.random.randn(nhf) * fhistd

    # Concatenate and shuffle low and high frequency components
    Z = np.concatenate((X, Y))
    np.random.shuffle(Z)

    # Adjust extrema parameters for mean heart rate
    hrfact = np.sqrt(hrmean / 60)
    Z = Z * hrfact

    # Add mean heart rate to the signal
    M = np.mean(Z)
    Z = Z - M + hrmean

    # Add Gaussian distributed noise
    Z = Z + np.random.randn(N) * hrstd

    return Z

def ecgsyn(sfecg=None, N=None, Anoise=None, hrmean=None, hrstd=None, lfhfratio=None, sfint=None, ti=None, ai=None, bi=None):
    if sfecg is None:
        sfecg = 256
    if N is None:
        N = 10000
    if Anoise is None:
        Anoise = 0.1
    if hrmean is None:
        hrmean = 60
    if hrstd is None:
        hrstd = 1
    if lfhfratio is None:
        lfhfratio = 0.5
    if sfint is None:
        sfint = 512
    if ti is None:
        ti = np.deg2rad([-70, -15, 0, 15, 100])

    # Convert angles to radians
    ti = np.deg2rad(ti)
    if ai is None:
        ai = np.array([1.2, -5, 30, -7.5, 0.75])
    if bi is None:
        bi = np.array([0.25, 0.1, 0.1, 0.1, 0.4])

    # Adjust extrema parameters for mean heart rate
    hrfact = np.sqrt(hrmean / 60)
    hrfact2 = np.sqrt(hrfact)
    bi = hrfact * np.array(bi)
    ti = np.multiply([hrfact2, hrfact, 1, hrfact, hrfact2], ti)

    # Check that sfint is an integer multiple of sfecg
    q = int(sfint / sfecg)
    qd = sfint / sfecg
    if q != qd: 
        raise ValueError("Internal sampling frequency (sfint) must be an integer multiple of the ECG sampling frequency (sfecg).")
    
    # Define frequency parameters for rr process
    flo = 0.1
    fhi = 0.25
    flostd = 0.01
    fhistd = 0.01
    fid = 1

    print(f"ECG sampled at {sfecg} Hz") 
    print(f"Approximate number of heart beats: {N}")
    print(f"Measurement noise amplitude: {Anoise}") 
    print(f"Heart rate mean: {hrmean} bpm") 
    print(f"Heart rate std: {hrstd} bpm") 
    print(f"LF/HF ratio: {lfhfratio}") 
    print(f"Internal sampling frequency: {sfint}") 
    print(" P Q R S T") 
    print(f"ti = {ti} radians") 
    print(f"ai = {ai}") 
    print(f"bi = {bi}")

    # Calculate time scales for rr and total output
    sampfreqrr = 1
    trr = 1 / sampfreqrr
    tstep = 1 / sfecg
    rrmean = 60 / hrmean

    # Calculate number of rr samples to be generated
    Nrr = int(np.ceil(N * tstep / trr))
    # Calculate total number of samples to be generated
    Ntot = int(np.ceil(N * tstep / trr * sfint / sfecg))

    # Calculate rr samples
    rr0 = rrprocess(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sampfreqrr, Nrr)

    # Upsample rr
    t = np.arange(0, Nrr) 
    f = interp1d(t, rr0, kind='cubic')
    tnew = np.arange(0, Nrr - 1, 1 / sfint)
    rr = f(tnew)

    # Calculate total number of samples to be generated
    Ntot = len(rr)

    # Initialize ecg
    ecg = np.zeros(Ntot)

    # Generate noise
    Anoise = Anoise * np.std(ecg)
    noise = Anoise * np.random.randn(Ntot)

    # Generate heartbeats
    ind = np.cumsum(rr)
    ti = np.concatenate(([0], np.cumsum(ti)))
    t = np.arange(1, Ntot + 1) / sfint

    for i in range(0, len(rr) - 1):
        t0 = np.arange(ind[i], ind[i + 1], 1)
        if t0[-1] >= len(ecg): # Skip if t0 is out of bounds
            continue
        ti_i = ti[i % len(ti)] if ti[i % len(ti)] != 0 else 1e-6 # Handle zero values in ti
        ai_i = ai[i % len(ai)] # Fix index out of range error
        P = bi[0] * np.random.randn() + ai_i * np.random.randn() * np.exp(-0.5 * ((t0 / ti_i) ** 2))
        Q = bi[1] * np.random.randn() + ai_i * np.random.randn() * np.exp(-0.5 * ((t0 / ti_i) ** 2))
        R = bi[2] * np.random.randn() + ai_i * np.random.randn() * np.exp(-0.5 * ((t0 / ti_i) ** 2))
        S = bi[3] * np.random.randn() + ai_i * np.random.randn() * np.exp(-0.5 * ((t0 / ti_i) ** 2))
        T = bi[4] * np.random.randn() + ai_i * np.random.randn() * np.exp(-0.5 * ((t0 / ti_i) ** 2))
        ecg[t0.astype(int)] = P + Q + R + S + T

    # Add noise
    ecg = ecg + noise
    return ecg


# Example usage
sfecg = 256
N = 10000
Anoise = 0.1
hrmean = 60
hrstd = 1
lfhfratio = 0.5
sfint = 512
ti = [-70, -15, 0, 15, 100]
ai = [1.2, -5, 30, -7.5, 0.75]
bi = [0.25, 0.1, 0.1, 0.1, 0.4]
ecg_signal = ecgsyn(sfecg, N, Anoise, hrmean, hrstd, lfhfratio, sfint, ti, ai, bi)

# Add noise to the signal
noise_amplitude = 1 
noise = noise_amplitude * np.random.randn(len(ecg_signal))
ecg_signal_with_noise = ecg_signal + noise

# Plot the ECG signal
figsize = (50, 10)
plt.figure(figsize=figsize)
t = np.arange(len(ecg_signal)) / sfecg
plt.plot(t, ecg_signal) 
plt.xlabel('Time (s)') 
plt.ylabel('ECG Amplitude') 
plt.title('Simulated ECG Signal')
save_path = os.path.join(output_directory, 'ecg_plymorphe_sim.png')
plt.savefig(save_path)
plt.close()

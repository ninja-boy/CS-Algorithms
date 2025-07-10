import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from numpy.linalg import norm

def generate_sine_signal(n, k, freq=5, fs=100):
    t = np.arange(n) / fs
    sum_signal = np.zeros(n)
    for i in range(1, k + 1):
        freq = i * 5
        sum_signal += np.sin(2 * np.pi * freq * t)
    return sum_signal

def measurement(m, n):
    Psi = idct(np.eye(n), norm='ortho')  # DCT basis matrix (n x n)
    Phi = np.random.randn(m, n)         # Measurement matrix (m x n)
    return Phi @ Psi                    # Combined measurement matrix (m x n)

def omp(y, A, tol=1e-6):
    m, n = A.shape
    r = y.copy()
    idx_set = []
    x_hat = np.zeros(n)
    num_iter = 0

    for _ in range(m):
        correlations = A.T @ r
        idx = np.argmax(np.abs(correlations))

        #if idx not in idx_set:
        idx_set.append(idx)        
        A_selected = A[:, idx_set]
        x_ls, _, _, _ = np.linalg.lstsq(A_selected, y, rcond=None)
        r = y - A_selected @ x_ls
        num_iter += 1
        if np.linalg.norm(r) < tol:
            break

    x_hat[idx_set] = x_ls
    return x_hat, num_iter

# --- Add 10 dB Gaussian noise ---
def add_noise(y, snr_db):
    """Add Gaussian noise to the signal y based on the specified SNR in dB."""
    signal_power = np.mean(np.abs(y)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*y.shape)
    return y + noise

# ---- Main Program ----
n = int(input("Enter length of sine wave signal (n): "))
m = int(input("Enter number of compressed samples (m such that m < n): "))
k = int(input("Enter sparsity level (k): "))

# Generate original sine signal
x_time = generate_sine_signal(n, k, 5)

# Transform to sparse domain (DCT)
x_sparse = dct(x_time, norm='ortho')

# Create measurement matrix and compress
A = measurement(m, n)
y = A @ x_sparse

# Add noise to the measurements
snr_db = int(input("Enter noise to be added (dB): "))  # Signal-to-noise ratio in dB 
y_noisy = add_noise(y, snr_db)

# Reconstruct sparse vector using OMP
x_sparse_rec, num_iter = omp(y, A)

# Inverse DCT to get back time-domain signal
x_time_rec = idct(x_sparse_rec, norm='ortho')

# Evaluation
print("\nReconstruction error (L2 norm):", norm(x_time - x_time_rec))
print("Iterations taken to converge:", num_iter)   

# Plotting original signals
plt.figure(figsize=(14, 6))
plt.plot(x_time, label="Original Sine Signal", linewidth=2)
plt.title("Sine Signal: Signal Reconstruction using DCT and OMP")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()

plt.axhline(0, color='black', linewidth=0.8)  # x-axis
plt.axvline(0, color='black', linewidth=0.8)  # y-axis
plt.show()

# Plotting ORIGINAL and RECONSTRUCTED SIGNALS
plt.figure(figsize=(14, 6))
plt.plot(x_time, label="Original Sine Signal", linewidth=2)
plt.plot(x_time_rec, '--r', label="Reconstructed Signal", linewidth=2)
plt.title("Sine Signal: Signal Reconstruction using DCT and OMP")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()

plt.axhline(0, color='black', linewidth=0.8)  # x-axis
plt.axvline(0, color='black', linewidth=0.8)  # y-axis
plt.show()
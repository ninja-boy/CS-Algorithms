import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from numpy.linalg import norm

def generate_sine_signal(n, freq=5, fs=100):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * freq * t) + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)

def measurement(m, n):
    Psi = dct(np.eye(n), norm='ortho')  # DCT basis matrix (n x n)
    Phi = np.random.randn(m, n)         # Measurement matrix (m x n)
    return Phi @ Psi                    # Combined measurement matrix (m x n)

def omp(y, A, tol=1e-6):
    m, n = A.shape
    r = y.copy()
    idx_set = []
    x_hat = np.zeros(n)

    for _ in range(m):
        correlations = A.T @ r
        idx = np.argmax(np.abs(correlations))

        #if idx not in idx_set:
        idx_set.append(idx)        
        A_selected = A[:, idx_set]
        x_ls, _, _, _ = np.linalg.lstsq(A_selected, y, rcond=None)
        r = y - A_selected @ x_ls

    x_hat[idx_set] = x_ls
    return x_hat

# ---- Main Program ----
n = int(input("Enter length of sine wave signal (n): "))
m = int(input("Enter number of compressed samples (m such that m < n): "))

#For singular sine wave, frequency can be taken as input
#frequency = float(input("Enter sine wave frequency (Hz): "))

#sampling_rate = float(input("Enter sampling rate (Hz): "))

# Generate original sine signal
x_time = generate_sine_signal(n, 5)

# Transform to sparse domain (DCT)
x_sparse = dct(x_time, norm='ortho')

# Create measurement matrix and compress
A = measurement(m, n)
y = A @ x_sparse

# Reconstruct sparse vector using OMP
x_sparse_rec = omp(y, A)

# Inverse DCT to get back time-domain signal
x_time_rec = idct(x_sparse_rec, norm='ortho')

# Evaluation
print("\nReconstruction error (L2 norm):", norm(x_time - x_time_rec))

# Plotting
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


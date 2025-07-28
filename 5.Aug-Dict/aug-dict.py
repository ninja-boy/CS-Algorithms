
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def add_noise(y, snr_db):
    """Add Gaussian noise to the signal y based on the specified SNR in dB."""
    signal_power = np.mean(np.abs(y)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*y.shape)
    return y + noise

def soft_thresholding(x, threshold):
    """Soft thresholding operator h_{alpha/L}"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

def ista(X, W_d, max_iter=200, tol=1e-6):
    alpha = 0.1
    L = 1.1 * np.linalg.norm(W_d.T @ W_d, 2)  # L > max eigenvalue of W_d.T @ W_d
    m, n = W_d.shape
    Z = np.zeros(n)
    for _ in range(max_iter):
        Z_old = Z.copy()
        gradient = W_d.T @ (W_d @ Z - X)
        Z -= gradient / L
        Z = soft_thresholding(Z - (1.0 / L) * gradient, alpha / L)
        if np.linalg.norm(Z - Z_old, ord = 2) < tol:
            break
    return Z

# 1. Signal Setup
n = int(input("Enter length of signal (n): "))  # Signal length
t = np.linspace(0, 1, n)

# Sinusoidal component (tone)
tone = np.cos(2 * np.pi * 10 * t)

# BPSK component
bits = np.random.randint(0, 2, n)
bpsk = (2 * bits - 1) * np.cos(2 * np.pi * 5 * t)

# Chirp component
f0, k = 5, 80
chirp = np.cos(2 * np.pi * (f0 * t + 0.5 * k * t**2))

# Mixed signal 
signal = tone + bpsk + chirp

# Sub-dictionary 1: DCT
D_dct = np.eye(n)
D_dct = dct(D_dct, norm='ortho').T

# Sub-dictionary 2: DFT (magnitude real part only)
D_dft = np.zeros((n, n))
for k in range(n):
    D_dft[:, k] = np.real(np.exp(2j * np.pi * k * np.arange(n) / n))
D_dft /= np.linalg.norm(D_dft, axis=0)

# Sub-dictionary 3: Chirp-like atoms
def generate_chirp_atoms(n, num_atoms):
    t = np.linspace(0, 1, n)
    D_chirp = np.zeros((n, num_atoms))
    for i in range(num_atoms):
        f0 = np.random.uniform(5, 20)
        k = np.random.uniform(10, 100)
        D_chirp[:, i] = np.cos(2 * np.pi * (f0 * t + 0.5 * k * t**2))
    D_chirp /= np.linalg.norm(D_chirp, axis=0)
    return D_chirp

D_chirp = generate_chirp_atoms(n, n)

# Concatenate dictionaries
D_total = np.concatenate([D_dct, D_dft, D_chirp], axis=1)

# Compressed Measurement
m = int(input("Enter number of compressed samples (m < n): "))  # Compressed samples
Phi = np.random.randn(m, n)# / np.sqrt(k)  # Measurement matrix
y = Phi @ signal

# Add noise to the measurements
snr_db = int(input("Enter noise to be added (dB): "))  # Signal-to-noise ratio in dB
y_noisy = add_noise(y, snr_db)

# Construct sensing matrix (Theta)
A = Phi @ D_total

# Sparse Recovery using ISTA
z = ista(y_noisy, A)
signal_hat = D_total @ z

# Calculate reconstruction error
error = np.linalg.norm(signal - signal_hat)
print(f"Reconstruction error (L2 norm): {error:.4f}")

# Plot original
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label="Original Mixed Signal")
plt.title("Mixed Signal Reconstruction from Compressed Measurements")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Plot original and reconstructed signals
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label="Original Mixed Signal")
plt.plot(t, signal_hat, '--', label="Reconstructed Signal (OMP)")
plt.title("Mixed Signal Reconstruction from Compressed Measurements")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

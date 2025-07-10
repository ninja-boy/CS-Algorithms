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

def soft_thresholding(x, threshold):
    """Soft thresholding operator h_{alpha/L}"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

def ista(X, W_d, max_iter=1000, tol=1e-6):
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

# ---- Main Program ----
import numpy as np
from scipy.fftpack import idct, dct
from numpy.linalg import norm
import matplotlib.pyplot as plt

n = int(input("Enter length of sine wave signal (n): "))
m = int(input("Enter number of compressed samples (m such that m < n): "))
k = int(input("Enter sparsity level (k): "))

# Generate original sine signal
x_time = generate_sine_signal(n, k, 5)

# Transform to sparse domain (DCT)
x_sparse = dct(x_time, norm='ortho')

# Create measurement matrix
A = measurement(m, n)

# Create compressed measurements
y = A @ x_sparse

# Apply ISTA to recover the sparse signal
x_sparse_rec = ista(y, A)

# Transform back to time domain
x_time_rec = idct(x_sparse_rec, norm='ortho')

# Calculate error
error = norm(x_time - x_time_rec)
print(f"Reconstruction error (L2 norm): {error:.4f}")
# Print number of iterations
#print(f"Number of iterations in ISTA: {num_iter}")

#plot only original
plt.figure(figsize=(12, 6))
plt.plot(x_time, label="Original Sine Signal", linewidth=2)
plt.title("Sine Signal: Signal Reconstruction using DCT and ISTA")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()

plt.axhline(0, color='black', linewidth=0.8)  # x-axis
plt.axvline(0, color='black', linewidth=0.8)  # y-axis
plt.show()

# Plot original and reconstructed signals
plt.figure(figsize=(12, 6))
plt.plot(x_time, label="Original Sine Signal", linewidth=2)
plt.plot(x_time_rec, '--r', label="Reconstructed Signal", linewidth=2)
plt.title("Sine Signal: Signal Reconstruction using DCT and ISTA")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()

plt.axhline(0, color='black', linewidth=0.8)  # x-axis
plt.axvline(0, color='black', linewidth=0.8)  # y-axis
plt.show()


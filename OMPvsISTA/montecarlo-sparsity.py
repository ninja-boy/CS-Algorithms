import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from numpy.linalg import norm
import random

def generate_sine_signal(n, k, freq=5, fs=100):
    t = np.arange(n) / fs
    sum_signal = np.zeros(n)
    for i in range(1, k + 1):
        freq = random.randrange(1, 100)
        sum_signal += np.sin(2 * np.pi * freq * t)
    return sum_signal


def measurement(m, n):
    Psi = idct(np.eye(n), norm='ortho')
    Phi = np.random.randn(m, n)
    return Phi @ Psi

def omp(y, A, tol=1e-6):
    m, n = A.shape
    r = y.copy()
    idx_set = []
    x_hat = np.zeros(n)
    for _ in range(m):
        correlations = A.T @ r
        idx = np.argmax(np.abs(correlations))
        if idx not in idx_set:
            idx_set.append(idx)
        A_selected = A[:, idx_set]
        x_ls, _, _, _ = np.linalg.lstsq(A_selected, y, rcond=None)
        r = y - A_selected @ x_ls
    x_hat[idx_set] = x_ls
    return x_hat

def soft_thresholding(x, threshold):
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

def monte_carlo_trial(sampling_rate, k, n=64, m=32):
    x_time = generate_sine_signal(n, k, 5, sampling_rate)
    x_sparse = dct(x_time, norm='ortho')
    A = measurement(m, n)
    y = A @ x_sparse
    rec_ista = ista(y, A)
    rec_omp = omp(y, A)
    rec_ista_t = idct(rec_ista, norm='ortho')
    error_ista = norm(x_time - rec_ista_t, ord=2)
    rec_omp_t = idct(rec_omp, norm='ortho')
    error_omp = norm(x_time - rec_omp_t, ord=2)
    return error_ista, error_omp    

# ---- Monte Carlo Simulation and Plotting ----
num_trials = 50
k_values = np.arange(1, 31, 2)  # Sparsity levels
sampling_rate = 100


plt.figure(figsize=(10, 6))
avg_errors_ista = []
avg_errors_omp = []
for k in k_values:
    errors_ista = []
    errors_omp = []
    for _ in range(num_trials):
        error_ista, error_omp = monte_carlo_trial(sampling_rate, k)
        errors_ista.append(error_ista)
        errors_omp.append(error_omp)
    avg_errors_ista.append(np.mean(errors_ista))
    avg_errors_omp.append(np.mean(errors_omp))

plt.plot(k_values, avg_errors_ista, label='ISTA', marker='o')
plt.plot(k_values, avg_errors_omp, label='OMP', marker='x')
plt.title("Monte Carlo: Avg. Reconstruction Error vs. Sparsity Level (k)")
plt.xlabel("Sparsity Level (k)")
plt.xticks(k_values)
plt.ylabel("Average Reconstruction Error (L2 norm)")
plt.legend()
plt.grid(True)
plt.show()
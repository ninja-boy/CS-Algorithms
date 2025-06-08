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

def monte_carlo_trial(sampling_rate, k, n=128, m=80):
    x_time = generate_sine_signal(n, k, 5, sampling_rate)
    x_sparse = dct(x_time, norm='ortho')
    A = measurement(m, n)
    y = A @ x_sparse
    x_sparse_rec = omp(y, A)
    x_time_rec = idct(x_sparse_rec, norm='ortho')
    error = norm(x_time - x_time_rec, ord=2)
    return error
    

# ---- Monte Carlo Simulation and Plotting ----
num_trials = 50
k_values = np.arange(1, 11, 1)  # Sparsity levels
sampling_rate = 100


plt.figure(figsize=(10, 6))
avg_errors = []
for k in k_values:
    errors = []
    for _ in range(num_trials):
        errors.append(monte_carlo_trial(sampling_rate, k))
    avg_errors.append(np.mean(errors))  

plt.plot(k_values, avg_errors, marker='o')

plt.title("Monte Carlo: Avg. Reconstruction Error vs. Sparsity Level (k)")
plt.xlabel("Sparsity Level (k)")
plt.xticks(k_values)
plt.ylabel("Average Reconstruction Error (L1 norm)")
plt.grid(True)
plt.show()

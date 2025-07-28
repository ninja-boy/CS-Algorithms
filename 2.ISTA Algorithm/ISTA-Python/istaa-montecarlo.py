import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from numpy.linalg import norm

def generate_sine_signal(n, freq=5, fs=100):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * freq * t) + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)

def measurement(m, n):
    Psi = idct(np.eye(n), norm='ortho')
    Phi = np.random.randn(m, n)
    return Phi @ Psi

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

def monte_carlo_trial(n, m, sampling_rate):
    x_time = generate_sine_signal(n, 5, sampling_rate)
    x_sparse = dct(x_time, norm='ortho')
    A = measurement(m, n)
    y = A @ x_sparse
    x_sparse_rec = ista(y, A)
    x_time_rec = idct(x_sparse_rec, norm='ortho')
    error = norm(x_time - x_time_rec, ord=2)
    return error

# ---- Monte Carlo Simulation and Plotting ----
num_trials = 50
n_values = [64, 128, 256]
m_values = np.arange(2, 65, 2)  # Number of measurements
sampling_rate = 100



plt.figure(figsize=(10, 6))

for n in n_values:
    avg_errors = []
    for m in m_values:
        if m >= n:
            avg_errors.append(np.nan)
            continue
        errors = []
        for _ in range(num_trials):
            errors.append(monte_carlo_trial(n, m, sampling_rate))
        avg_errors.append(np.mean(errors))
    plt.plot(m_values, avg_errors, marker='o', label=f'n={n}')

plt.title("Monte Carlo: Avg. Reconstruction Error vs. Number of Measurements")
plt.xlabel("Number of Measurements (m)")
plt.ylabel("Average Reconstruction Error (L1 norm)")
plt.legend()
plt.grid(True)
plt.show()
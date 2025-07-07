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

def monte_carlo_trial(sampling_rate, snr_db, n=128, m=80):
    x_time = generate_sine_signal(n, 5, sampling_rate)
    x_sparse = dct(x_time, norm='ortho')
    A = measurement(m, n)
    y = A @ x_sparse
    y_noisy = add_noise(y, snr_db)
    rec_ista = ista(y_noisy, A)
    rec_omp = omp(y_noisy, A)
    #rec_cod = cod(y, A, idct(np.eye(n), norm='ortho'), 0.01, 1000)
    #rec_cod_t = idct(rec_cod, norm='ortho')
    rec_ista_t = idct(rec_ista, norm='ortho')
    error_ista = norm(x_time - rec_ista_t, ord=2)
    rec_omp_t = idct(rec_omp, norm='ortho')
    error_omp = norm(x_time - rec_omp_t, ord=2)
    #error_cod = norm(x_time - rec_cod_t, ord=2)
    return error_ista, error_omp, #error_cod


def add_noise(y, snr_db):
    """Add Gaussian noise to the signal y based on the specified SNR in dB."""
    signal_power = np.mean(np.abs(y)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*y.shape)
    return y + noise
    


# ---- Monte Carlo Simulation and Plotting ----
num_trials = 50
n = 128  # Length of the signal
m = 64  # Number of measurements
noise_val = np.arange(0, 51, 5)  # Noise levels in dB
sampling_rate = 100

plt.figure(figsize=(10, 6))
avg_errors_ista = []
avg_errors_omp = []
for noise in noise_val:
    errors_ista = []
    errors_omp = []
    for _ in range(num_trials):
        error_ista, error_omp = monte_carlo_trial(sampling_rate, noise)
        errors_ista.append(error_ista)
        errors_omp.append(error_omp)
    avg_errors_ista.append(np.mean(errors_ista))
    avg_errors_omp.append(np.mean(errors_omp))

plt.plot(noise_val, avg_errors_ista, label='ISTA', marker='o')
plt.plot(noise_val, avg_errors_omp, label='OMP', marker='x')

plt.title("Monte Carlo: Avg. Reconstruction Error vs. Noise added (dB)")
plt.xlabel("Noise Added (dB)")
plt.legend()
plt.xticks(noise_val)
plt.ylabel("Average Reconstruction Error (L2 norm)")
plt.grid(True)
plt.show()
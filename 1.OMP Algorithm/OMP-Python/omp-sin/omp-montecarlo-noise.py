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

def monte_carlo_trial(sampling_rate, snr_db, n=128, m=80):
    x_time = generate_sine_signal(n, 5, sampling_rate)
    x_sparse = dct(x_time, norm='ortho')
    A = measurement(m, n)
    y = A @ x_sparse
    y_noisy = add_noise(y, snr_db)
    x_sparse_rec = omp(y_noisy, A)
    x_time_rec = idct(x_sparse_rec, norm='ortho')
    error = norm(x_time - x_time_rec, ord=2)
    return error


def add_noise(y, snr_db):
    """Add Gaussian noise to the signal y based on the specified SNR in dB."""
    signal_power = np.mean(np.abs(y)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*y.shape)
    return y + noise
    


# ---- Monte Carlo Simulation and Plotting ----
num_trials = 50
noise_val = np.arange(0, 51, 5)  # Noise levels in dB
sampling_rate = 100

plt.figure(figsize=(10, 6))
avg_errors = []
for noise in noise_val:
    errors = []
    for _ in range(num_trials):
        errors.append(monte_carlo_trial(sampling_rate, noise))
    avg_errors.append(np.mean(errors))  # <-- Move this line outside the inner loop

plt.plot(noise_val, avg_errors, marker='o')

plt.title("Monte Carlo: Avg. Reconstruction Error vs. Noise added (dB)")
plt.xlabel("Noise Added (dB)")
plt.xticks(noise_val)
plt.ylabel("Average Reconstruction Error (L1 norm)")
plt.grid(True)
plt.show()

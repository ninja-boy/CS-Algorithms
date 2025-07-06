import numpy as np
from scipy.fftpack import dct, idct
from numpy.linalg import norm
import matplotlib.pyplot as plt

def generate_sine_signal(n, freq=5, fs=100):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * freq * t) + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)

def measurement_cod(m, n):
    Psi = idct(np.eye(n), norm='ortho')
    Phi = np.random.randn(m, n)
    Phi = Phi / np.linalg.norm(Phi, axis=1, keepdims=True)
    return Phi, Psi, Phi @ Psi

def cod(y, theta, Psi, alpha=0.05, num_iter=50):
    n = theta.shape[1]
    z = np.zeros((n, 1))
    B = theta.T @ y.reshape(-1, 1)
    S = np.eye(n) - theta.T @ theta
    for _ in range(num_iter):
        z_bar = np.sign(B) * np.maximum(np.abs(B) - alpha, 0)
        k = np.argmax(np.abs(z - z_bar))
        delta = z_bar[k, 0] - z[k, 0]
        B = B + S[:, [k]] * delta
        z[k, 0] = z_bar[k, 0]
    x_rec = Psi @ z
    return x_rec.flatten()

def monte_carlo_trial_cod(n, m, sampling_rate, alpha=0.05, num_iter=50):
    x_time = generate_sine_signal(n, 5, sampling_rate)
    Phi, Psi, theta = measurement_cod(m, n)
    y = Phi @ x_time.reshape(-1, 1)
    x_time_rec = cod(y, theta, Psi, alpha=alpha, num_iter=num_iter)
    error = norm(x_time - x_time_rec, ord=2)
    return error

# ---- Monte Carlo Simulation and Plotting for CoD ----
num_trials = 20
n_values = [64, 128, 256]
m_values = np.arange(2, 65, 2)
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
            errors.append(monte_carlo_trial_cod(n, m, sampling_rate))
        avg_errors.append(np.mean(errors))
    plt.plot(m_values, avg_errors, marker='o', label=f'n={n}')

plt.title("Monte Carlo: Avg. Reconstruction Error vs. Number of Measurements (CoD)")
plt.xlabel("Number of Measurements (m)")
plt.ylabel("Average Reconstruction Error (L2 norm)")
plt.legend()
plt.grid(True)
plt.show()
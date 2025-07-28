

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# Parameters
n = 128                  # Length of signal
f1 = 5
f2=10
f3=20                  # Frequency of sine wave (Hz)
fs = 512                # Sampling rate (Hz)

# Generate sine wave
t = np.arange(n) / fs
x1 = np.sin(2 * np.pi * f1 * t)
x2 = np.sin(2 * np.pi * f2 * t)
x3 = np.sin(2 * np.pi * f3 * t)
# Combine sine waves
x = x1 + x2 + x3

# Apply DCT (Type-II, orthonormal)
x_dct1 = dct(x1, norm='ortho')
x_dct2 = dct(x2, norm='ortho')
x_dct3 = dct(x3, norm='ortho')
x_dct = dct(x, norm='ortho')


# Plot time domain signal
plt.figure(figsize=(16, 9))
plt.subplot(2, 4, 1)
plt.plot(t, x1, label="Sine Wave 1")
plt.title("Original Sine Wave (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 4, 2)
plt.plot(t, x2, label="Sine Wave 1")
plt.title("Original Sine Wave (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 4, 3)
plt.plot(t, x3, label="Sine Wave 1")
plt.title("Original Sine Wave (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot DCT coefficients
plt.subplot(2, 4, 5)
plt.stem(np.arange(n), x_dct1, basefmt=' ', linefmt='r-', markerfmt='ro')
plt.title("DCT of Sine Wave (Frequency Domain)")
plt.xlabel("DCT Coefficient Index")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(2, 4, 6)
plt.stem(np.arange(n), x_dct2, basefmt=' ', linefmt='r-', markerfmt='ro')
plt.title("DCT of Sine Wave (Frequency Domain)")
plt.xlabel("DCT Coefficient Index")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(2, 4, 7)
plt.stem(np.arange(n), x_dct3, basefmt=' ', linefmt='r-', markerfmt='ro')
plt.title("DCT of Sine Wave (Frequency Domain)")
plt.xlabel("DCT Coefficient Index")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(2, 4, 4)
plt.plot(t, x, label="Sine Wave 1", color='blue')
plt.title("Sine Wave sum (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 4, 8)
plt.stem(np.arange(n), x_dct, basefmt=' ', linefmt='r-', markerfmt='ro')
plt.title("DCT of Sine Wave (Frequency Domain)")
plt.xlabel("DCT Coefficient Index")
plt.ylabel("Magnitude")
plt.grid(True)


plt.tight_layout()
plt.show()


# experiment 1 to find a tone

import numpy as np
import matplotlib.pyplot as plt

fs = 1000       # samples per second
N = 256         # samples in one frame
f_tone = 50    # Hz
f_weak = 120    # Hz, the faint second tone
a_weak = 0.001  # its amplitude, 1000x smaller than the main tone

n = np.arange(N)  # n = the sample numbers 0 to N-1
t = n / fs        # t = the time of each sample in seconds (sample number divided by fs)
x = np.sin(2 * np.pi * f_tone * t) + a_weak * np.sin(2 * np.pi * f_weak * t)  # x = a sine wave at f_tone plus the weak tone
X = np.fft.rfft(x)  # X = the FFT of x
freqs = np.fft.rfftfreq(N, 1/fs)  # freqs = the frequency of each bin
power = np.abs(X)**2  # power = magnitude of X, squared
w = 0.5 * (1 - np.cos(2 * np.pi * n / (N)))  # w = the Hann window (formula above, using n, N, np.cos, np.pi)
power_hann = np.abs(np.fft.rfft(x * w))**2  # power_hann = same as power, but take the FFT of x times w


print(f"Frequency of the largest bin: {freqs[np.argmax(power)]} Hz")

plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.xlabel("time (s)")
plt.ylabel("amplitude")
plt.subplot(2, 1, 2)
plt.xlabel("frequency (Hz)")
plt.ylabel("power")
plt.semilogy(freqs, power, label="no window")
plt.semilogy(freqs, power_hann, label="Hann")
plt.legend()

plt.tight_layout()
plt.savefig("docs/figures/window.png", dpi=150)
plt.show()
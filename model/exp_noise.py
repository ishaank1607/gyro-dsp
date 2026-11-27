import numpy as np
import matplotlib.pyplot as plt

fs = 1000
N = 256
M = 32              # number of frames to average
noise_level = 0.1
a_weak = 0.03
dc = 2.0

t = np.arange(N * M) / fs       # enough samples for M frames
rng = np.random.default_rng(1)  # random numbers, same ones every run
noise = noise_level * rng.standard_normal(N * M)

x = np.sin(2 * np.pi * 50 * t) + a_weak * np.sin(2 * np.pi * 120 * t) + dc + noise
freqs = np.fft.rfftfreq(N, 1/fs)  # freqs = the frequency of each bin
w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N)))  # w = the Hann window

def frame_power(frame):
    # remove the mean, apply w, then return the power of the rfft
    frame_clean = frame - np.mean(frame)
    return np.abs(np.fft.rfft(frame_clean * w))**2

total = np.zeros(N // 2 + 1)    # one running sum per bin
for k in range(M):
    frame = x[k * N : (k + 1) * N]
    total += frame_power(frame)

# average = total divided by M
average = total / M
# single = power of the first frame only
single = frame_power(x[0 * N : (0 + 1) * N])

plt.semilogy(freqs, single, label="Single Frame")
plt.semilogy(freqs, average, label="Averaged")
plt.xlabel("frequency (Hz)")
plt.ylabel("power")
plt.legend()
plt.savefig("docs/figures/noise_reduction.png", dpi=150)
plt.show()
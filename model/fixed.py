import numpy as np

fs = 1000
N = 256
n = np.arange(N)
t = n / fs
dc = 16000

def to_q15(v):
    v = v * 32768
    v = np.round(v)
    v = np.clip(v, -32768, 32767)
    return v.astype(np.int64)

def q15_mult(a, b):
    return np.clip((a * b + (1 << 14)) >> 15, -32768, 32767)

def remove_dc(frame):
    total = np.sum(frame)
    mean = (total + 128) >> 8 
    return np.clip(frame - mean, -32768, 32767)

if __name__ == "__main__":
    w_float = 0.5 * (1 - np.cos(2 * np.pi * n / (N)))
    w_q = to_q15(w_float)

    x_int = np.round(12000 * np.sin(2 * np.pi * 50 * t) + dc).astype(np.int64)

    x_clean = remove_dc(x_int)
    y_int = q15_mult(x_clean, w_q)
    y_ref = (x_int - np.mean(x_int)) * w_float
    error = y_int - y_ref

    print(w_q[0], w_q[64], w_q[128])
    print(np.max(np.abs(error)))
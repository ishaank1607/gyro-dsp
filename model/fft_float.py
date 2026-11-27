# fft float script
import numpy as np

def bit_reverse(i, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r

def butterfly(a, b, W):
    t = W * b
    return a + t, a - t

def fft(x):
    N = len(x)
    bits = int(np.log2(N))
    A = np.zeros(N, dtype=complex)
    for i in range(N):
        A[bit_reverse(i, bits)] = x[i]

    for stage in range(bits):
        half = 1 << (stage)
        size = half << 1             
        for start in range(0, N, size):  
            for j in range(half):    
                top = start + j
                bottom = top + half
                k = (j) << (bits - 1 - stage)
                W = np.exp(-2j * np.pi * k / N)
                A[top], A[bottom] = butterfly(A[top], A[bottom], W)
    return A

x = np.random.default_rng(1).standard_normal(8)
print(np.max(np.abs(fft(x) - np.fft.fft(x))))
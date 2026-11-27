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

print([bit_reverse(i, 3) for i in range(8)])

W = np.exp(-2j * np.pi * 2 / 8)
print(W)
print(butterfly(1, 1, W))
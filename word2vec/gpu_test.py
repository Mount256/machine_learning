import numpy as np
import cupy as cp
import time

np_arr = np.random.randint(1, 10000, size=(1000, 1000, 8, 8), dtype='int32')
cp_arr = cp.random.randint(1, 10000, size=(1000, 1000, 8, 8), dtype='int32')

t = time.perf_counter()
for _ in range(100):
    np_arr + np_arr
t = time.perf_counter() - t
print(f"运行时间是: {t}s")

t = time.perf_counter()
for _ in range(100):
    cp_arr + cp_arr
t = time.perf_counter() - t
print(f"运行时间是: {t}s")

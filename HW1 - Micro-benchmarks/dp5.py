import numpy as np
import time


# Micro-benchmark for the dot product function 
def benchmark(N, repetitions):

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    
    # Run the benchmark
    times = []
    for _ in range(repetitions):
        start = time.time()
        np.dot(A,B)
        end = time.time()
        times.append(end - start)
    
    # Calculate average time for the second half of the repetitions
    avg_time = np.mean(times[repetitions // 2:])
    bandwidth = (2 * N * A.itemsize) / (1024**3 * avg_time)  # in GB/sec
    flops = (2 * N) / (avg_time * 1e9)  # in GFLOP/sec
    
    return avg_time, bandwidth, flops

# Parameters
N_small = 1000000
repetitions_small = 1000
N_large = 300000000
repetitions_large = 20
# Benchmark for small N
avg_time_small, bandwidth_small, flops_small = benchmark(N_small, repetitions_small)
# Benchmark for large N
avg_time_large, bandwidth_large, flops_large = benchmark(N_large, repetitions_large)
# Print results nicely
print("Average time (small N):", avg_time_small)
print("Bandwidth (small N):", bandwidth_small)
print("FLOPS (small N):", flops_small)
print("Average time (large N):", avg_time_large)
print("Bandwidth (large N):", bandwidth_large)
print("FLOPS (large N):", flops_large)
# Return results
avg_time_small, bandwidth_small, flops_small, avg_time_large, bandwidth_large, flops_large

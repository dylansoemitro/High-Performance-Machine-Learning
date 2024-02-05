#include <time.h>
#include <stdio.h>
#include <stdlib.h>


// Function to compute the dot product of two vectors
float dpunroll(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j=0;j<N;j+=4){
        R += pA[j]*pB[j] + pA[j+1]*pB[j+1] \
        + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3];
    }
    return R;
    
}
// Micro-benchmark for the dot product function
int main(int argc, char *argv[]) {
    if (argc != 3){
        printf("Usage: %s <N> <Repetitions>\n", argv[0]);
        return 1;
    }

    long N = atol(argv[1]);
    int repetitions = atoi(argv[2]);
    float *pA = (float *)malloc(N*sizeof(float));
    float *pB = (float *)malloc(N*sizeof(float));
    
     for (long i=0; i<N; i++){
        pA[i] = 1.0f;
        pB[i] = 1.0f;
    }

    struct timespec start, end;
    double times[repetitions];
    double total_time = 0.0;
    double avg_time, bandwidth, flops;
    float a;
    for (int i=0; i<repetitions; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        a = dpunroll(N, pA, pB);
        printf("%f\n", a);
        clock_gettime(CLOCK_MONOTONIC, &end);

        times[i] = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.0; 
        // Only keep the times for the second half of repetitions 
        if (i >= repetitions / 2) total_time += times[i];
    }

    avg_time = total_time / (repetitions / 2); // Calculate the average time for the second half of repetitions
    bandwidth = (2.0 * N * sizeof(float) / (1024 * 1024 * 1024)) / avg_time; // Calculate bandwidth in GB/sec
    flops = (2.0 * N) / avg_time / 1e9; // Calculate FLOPs (floating-point operations per second) in GFLOP/sec

    // Print the results
    printf("N: %ld <T>: %f sec B: %f GB/sec F: %f GFLOP/sec\n", N, avg_time, bandwidth, flops);

    free(pA);
    free(pB);
    return 0;
}
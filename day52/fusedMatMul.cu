#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1024
#define MEM 16

__global__ void init(float *x, unsigned int seed)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        curandState state;
        curand_init(seed, (row * N + col), 0, &state);
        x[row * N + col] = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

__global__ void matMulFusedQuant(float *a, float *b, float *c, float scale_a, float scale_b)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int32_t acc = 0;
        for (int k = 0; k < N; k++)
        {
            int8_t qa = max(-128, min(127, __float2int_rn(a[row * N + k] / scale_a)));
            int8_t qb = max(-128, min(127, __float2int_rn(b[k * N + col] / scale_b)));

            acc += (int32_t)qa * (int32_t)qb;
        }

        float result = acc * scale_a * scale_b;
        c[row * N + col] = result;
    }
}

int main()
{
    float *a, *b, *c_fused, *c_cpu;
    int size = N * N * sizeof(float);

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_fused, size);
    cudaMallocManaged(&c_cpu, size);

    dim3 n_threads(MEM, MEM);
    dim3 n_blocks((N + MEM - 1) / MEM, (N + MEM - 1) / MEM);

    cudaStream_t strm1, strm2;
    cudaStreamCreate(&strm1);
    cudaStreamCreate(&strm2);

    init<<<n_blocks, n_threads, 0, strm1>>>(a, time(NULL));
    init<<<n_blocks, n_threads, 0, strm2>>>(b, time(NULL));
    cudaStreamSynchronize(strm1);
    cudaStreamSynchronize(strm2);

    cudaMemset(c_fused, 0, size);
    cudaMemset(c_cpu, 0, size);

    float scale_a = 0.02f;
    float scale_b = 0.02f;

    clock_t start_cpu = clock();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float val = 0.0f;
            for (int k = 0; k < N; k++)
                val += a[i * N + k] * b[k * N + j];
            c_cpu[i * N + j] = val;
        }
    }
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;

    cudaEvent_t start, stop;
    float fused_time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulFusedQuant<<<n_blocks, n_threads>>>(a, b, c_fused, scale_a, scale_b);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fused_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int errors = 0;
    for (int i = 0; i < N * N; i++)
    {
        float ref = c_cpu[i];
        float got = c_fused[i];
        if (fabsf(ref - got) > 5.0f)
        {
            if (++errors < 10)
                printf("Mismatch at %d: CPU=%.2f, Quant=%.2f\n", i, ref, got);
        }
    }

    if (errors == 0)
        printf("Fused quantized matmul verified!\n");
    else
        printf("Fused quantized matmul has %d mismatches.\n", errors);

    printf("CPU time: %.2f ms\n", cpu_time);
    printf("Fused Quantized GPU time: %.2f ms\n", fused_time);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c_fused);
    cudaFree(c_cpu);
    cudaStreamDestroy(strm1);
    cudaStreamDestroy(strm2);

    return 0;
}

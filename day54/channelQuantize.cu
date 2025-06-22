#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <algorithm>

#define N 1024
#define TILE 16

__global__ void init(float *x, unsigned int seed)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        curandState state;
        curand_init(seed, row * N + col, 0, &state);
        x[row * N + col] = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

__global__ void quantize(const float* input, int8_t* output, const float* scale, const int8_t* zp, bool per_row)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        int channel = per_row ? row : col;
        float s = scale[channel];
        int8_t z = zp[channel];
        int idx = row * N + col;
        int q = __float2int_rn(input[idx] / s) + z;
        output[idx] = max(-128, min(127, q));
    }
}

__global__ void dequantizeTiled(const int8_t* A, const int8_t* B, float* C, const float* scale_a, const float* scale_b, const int8_t* zp_a, const int8_t* zp_b)
{
    __shared__ int8_t Asub[TILE][TILE];
    __shared__ int8_t Bsub[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE; ++t)
    {
        int a_idx = row * N + t * TILE + threadIdx.x;
        int b_idx = (t * TILE + threadIdx.y) * N + col;

        Asub[threadIdx.y][threadIdx.x] = (row < N && (t * TILE + threadIdx.x) < N) ? A[a_idx] : 0;
        Bsub[threadIdx.y][threadIdx.x] = ((t * TILE + threadIdx.y) < N && col < N) ? B[b_idx] : 0;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
        {
            int32_t a_val = (int32_t)Asub[threadIdx.y][k] - (int32_t)zp_a[row];
            int32_t b_val = (int32_t)Bsub[k][threadIdx.x] - (int32_t)zp_b[col];
            sum += (float)(a_val * b_val);
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        float scale = scale_a[row] * scale_b[col];
        C[row * N + col] = sum * scale;
    }
}

void qparams(const float* input, float* scale, int8_t* zp, bool per_row)
{
    for (int i = 0; i < N; ++i)
    {
        float min_val = 1e30f, max_val = -1e30f;
        for (int j = 0; j < N; ++j)
        {
            float val = per_row ? input[i * N + j] : input[j * N + i];
            min_val = fminf(min_val, val);
            max_val = fmaxf(max_val, val);
        }
        float range = max_val - min_val;
        float s = (range == 0.0f) ? 1.0f : range / 255.0f;
        int8_t z = static_cast<int8_t>(roundf(-min_val / s));

        scale[i] = s;
        zp[i] = z;
    }
}

int main()
{
    float *a_fp32, *b_fp32, *c_fp32;
    int8_t *a_int8, *b_int8;
    float *scale_a, *scale_b;
    int8_t *zp_a, *zp_b;

    size_t size_f32 = N * N * sizeof(float);
    size_t size_i8 = N * N * sizeof(int8_t);

    cudaMallocManaged(&a_fp32, size_f32);
    cudaMallocManaged(&b_fp32, size_f32);
    cudaMallocManaged(&c_fp32, size_f32);
    cudaMallocManaged(&a_int8, size_i8);
    cudaMallocManaged(&b_int8, size_i8);
    cudaMallocManaged(&scale_a, N * sizeof(float));
    cudaMallocManaged(&scale_b, N * sizeof(float));
    cudaMallocManaged(&zp_a, N * sizeof(int8_t));
    cudaMallocManaged(&zp_b, N * sizeof(int8_t));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    init<<<blocks, threads>>>(a_fp32, time(NULL));
    init<<<blocks, threads>>>(b_fp32, time(NULL));
    cudaDeviceSynchronize();

    qparams(a_fp32, scale_a, zp_a, true);
    qparams(b_fp32, scale_b, zp_b, false);

    quantize<<<blocks, threads>>>(a_fp32, a_int8, scale_a, zp_a, true);
    quantize<<<blocks, threads>>>(b_fp32, b_int8, scale_b, zp_b, false);
    cudaDeviceSynchronize();

    cudaMemset(c_fp32, 0, size_f32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dequantizeTiled<<<blocks, threads>>>(a_int8, b_int8, c_fp32, scale_a, scale_b, zp_a, zp_b);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("Tiled Dequantized INT8 MatMul Time: %.2f ms\n", time_ms);

    cudaFree(a_fp32); cudaFree(b_fp32); cudaFree(c_fp32);
    cudaFree(a_int8); cudaFree(b_int8);
    cudaFree(scale_a); cudaFree(scale_b);
    cudaFree(zp_a); cudaFree(zp_b);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

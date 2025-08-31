#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <algorithm>

#define N 1024
#define TILE 16
#define LAYERS 3

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

__global__ void relu(float* x, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] > 0.f ? x[i] : 0.f;
}

__global__ void accum_stats(const float* x, float* abs_sum, float* sq_sum) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) 
    {
        float v = x[row * N + col];
        atomicAdd(&abs_sum[col], fabsf(v));
        atomicAdd(&sq_sum[col], v * v);
    }
}

__global__ void quantize_awq(const float* B, const float* scale_b, const float* s_k, int8_t* Bq) 
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < N && c < N) 
    {
        float inv = 1.f / scale_b[r];
        float v = B[r * N + c] * s_k[r] * inv;
        int q = __float2int_rn(fminf(127.f, fmaxf(-127.f, v)));
        Bq[r * N + c] = (int8_t)q;
    }
}

__global__ void dequantizeTiled(const float* A, const int8_t* Bq, float* C, const float* inv_s, const float* scale_b) 
{
    __shared__ float Asub[TILE][TILE];
    __shared__ int8_t Bsub[TILE][TILE];
    __shared__ float invTile[TILE];
    __shared__ float dTile[TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < N / TILE; ++t) 
    {
        int a_idx = row * N + t * TILE + threadIdx.x;
        int b_idx = (t * TILE + threadIdx.y) * N + col;
        Asub[threadIdx.y][threadIdx.x] = (row < N && (t * TILE + threadIdx.x) < N) ? A[a_idx] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] = ((t * TILE + threadIdx.y) < N && col < N) ? Bq[b_idx] : 0;

        if (threadIdx.y == 0) 
        {
            int kA = t * TILE + threadIdx.x;
            invTile[threadIdx.x] = (kA < N) ? inv_s[kA] : 1.0f;
            dTile[threadIdx.x] = (kA < N) ? scale_b[kA] : 1.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) 
        {
            float a_val = Asub[threadIdx.y][k] * invTile[k];
            float b_val = (float)Bsub[k][threadIdx.x] * dTile[k];
            sum += a_val * b_val;
        }
        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = sum;
}

void compute_s(const float* abs_mean, const float* sq_mean, const float* B, float* s_k) 
{
    std::vector<float> s_tmp(N), s_best(N);
    float best = INFINITY;
    const float eps = 1e-8f;
    const float smax = 8.f;
    float gammas[] = {0.f, 0.25f, 0.5f, 0.75f, 1.f};

    #pragma unroll
    for (float gamma : gammas) 
    {
        for (int k = 0; k < N; ++k) s_tmp[k] = powf(fmaxf(abs_mean[k], eps), gamma);

        float mean_s = 0.f;

        for (int k = 0; k < N; ++k) mean_s += s_tmp[k];

        mean_s /= N;

        for (int k = 0; k < N; ++k) 
        {
            s_tmp[k] /= mean_s;
            s_tmp[k] = fminf(smax, fmaxf(1.f / smax, s_tmp[k]));
        }

        float err = 0.f;

        for (int k = 0; k < N; ++k) 
        {
            float amax = 0.f;
            const float* row = B + k * N;
            #pragma unroll
            for (int j = 0; j < N; ++j) 
            {
                float w = row[j] * s_tmp[k];
                float aw = fabsf(w);
                if (aw > amax) amax = aw;
            }

            float d = amax > 0.f ? amax / 127.f : 1.f;
            float row_err = 0.f;

            #pragma unroll
            for (int j = 0; j < N; ++j) 
            {
                float w = row[j] * s_tmp[k];
                int q = (int)lrintf(fminf(127.f, fmaxf(-127.f, w / d)));
                float dq = (float)q * d;
                float diff = dq - w;
                row_err += diff * diff;
            }

            err += row_err * sq_mean[k];
        }

        if (err < best) 
        {
            best = err;
            for (int k = 0; k < N; ++k) s_best[k] = s_tmp[k];
        }
    }

    for (int k = 0; k < N; ++k) s_k[k] = s_best[k];
}

void compute_scale_b_host(const float* B, const float* s_k, float* scale_b) 
{
    for (int k = 0; k < N; ++k) 
    {
        float amax = 0.f;
        const float* row = B + k * N;

        #pragma unroll
        for (int j = 0; j < N; ++j) 
        {
            float w = row[j] * s_k[k];
            float aw = fabsf(w);
            if (aw > amax) amax = aw;
        }
        scale_b[k] = amax > 0.f ? amax / 127.f : 1.f;
    }
}

int main() 
{
    float *a_fp32, *z_fp32, *stats_abs, *stats_sq;
    float *b_fp32[LAYERS], *scale_b[LAYERS], *s_k[LAYERS], *inv_s[LAYERS];
    int8_t *b_int8[LAYERS];
    size_t size_f32 = N * N * sizeof(float);
    size_t size_i8 = N * N * sizeof(int8_t);

    cudaMallocManaged(&a_fp32, size_f32);
    cudaMallocManaged(&z_fp32, size_f32);
    cudaMallocManaged(&stats_abs, N * sizeof(float));
    cudaMallocManaged(&stats_sq, N * sizeof(float));

    for (int l = 0; l < LAYERS; ++l) 
    {
        cudaMallocManaged(&b_fp32[l], size_f32);
        cudaMallocManaged(&b_int8[l], size_i8);
        cudaMallocManaged(&scale_b[l], N * sizeof(float));
        cudaMallocManaged(&s_k[l], N * sizeof(float));
        cudaMallocManaged(&inv_s[l], N * sizeof(float));
    }

    dim3 threads2d(TILE, TILE);
    dim3 blocks2d((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    dim3 threads1d(256);
    dim3 blocks1d((N * N + 255) / 256);
    init<<<blocks2d, threads2d>>>(a_fp32, 1234);

    for (int l = 0; l < LAYERS; ++l) init<<<blocks2d, threads2d>>>(b_fp32[l], 4321u + l);
    cudaDeviceSynchronize();

    float *x_cur = a_fp32;
    float *x_next = z_fp32;

    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int l = 0; l < LAYERS; ++l) 
    {
        cudaMemset(stats_abs, 0, N * sizeof(float));
        cudaMemset(stats_sq, 0, N * sizeof(float));
        accum_stats<<<blocks2d, threads2d>>>(x_cur, stats_abs, stats_sq);
        cudaDeviceSynchronize();

        for (int k = 0; k < N; ++k) 
        {
            stats_abs[k] /= (float)N;
            stats_sq[k]  /= (float)N;
        }

        compute_s(stats_abs, stats_sq, b_fp32[l], s_k[l]);

        for (int k = 0; k < N; ++k) inv_s[l][k] = 1.f / s_k[l][k];

        compute_scale_b_host(b_fp32[l], s_k[l], scale_b[l]);

        quantize_awq<<<blocks2d, threads2d>>>(b_fp32[l], scale_b[l], s_k[l], b_int8[l]);
        cudaDeviceSynchronize();
        dequantizeTiled<<<blocks2d, threads2d>>>(x_cur, b_int8[l], x_next, inv_s[l], scale_b[l]);
        relu<<<blocks1d, threads1d>>>(x_next, N * N);
        cudaDeviceSynchronize();

        float* tmp = x_cur; x_cur = x_next; x_next = tmp;
    }
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    float time_ms; cudaEventElapsedTime(&time_ms, start, stop);
    printf("AWQ pipeline time: %.2f ms\n", time_ms);

    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
    cudaFree(a_fp32); 
    cudaFree(z_fp32); 
    cudaFree(stats_abs); 
    cudaFree(stats_sq);

    for (int l = 0; l < LAYERS; ++l) 
    {
        cudaFree(b_fp32[l]); 
        cudaFree(b_int8[l]); 
        cudaFree(scale_b[l]); 
        cudaFree(s_k[l]); 
        cudaFree(inv_s[l]);
    }
    return 0;
}
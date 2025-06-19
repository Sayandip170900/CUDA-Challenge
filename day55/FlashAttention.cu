#include <iostream>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <algorithm>

#define SMEM (48 * 1024)
#define token_len 128
#define embed_dim 32

constexpr int block_row = 32;
constexpr int block_col = 32;

// Ensure shared memory fits within limit
static_assert((3 * block_row * embed_dim + block_row * block_col) * 4 <= SMEM, "Shared memory usage exceeds shared memory size");

constexpr int total_block_row = (token_len + block_row - 1) / block_row;
constexpr int total_block_col = (token_len + block_col - 1) / block_col;

// Forward kernel with always-on causal masking
__global__ void flashAttentionForward(const float *Q, const float *K, const float *V, float *O, float *L, float *M, const float scale)
{
    extern __shared__ float mem[];

    // Shared memory layout: Qi, Kj, Vj, score buffer
    float *Qi = mem;
    float *Kj = Qi + block_row * embed_dim;
    float *Vj = Kj + block_col * embed_dim;
    float *S  = Vj + block_col * embed_dim;

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int row_block = blockIdx.x;

    // Initialize softmax stats and output
    for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
    {
        int global_i = row_block * block_row + i;
        M[global_i] = -INFINITY;
        L[global_i] = 0.0f;
        for (int d = 0; d < embed_dim; d++) 
        {
            O[global_i * embed_dim + d] = 0.0f;
        }
    }

    // Load Q into shared memory
    for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
    {
        for (int d = thread_y; d < embed_dim; d += blockDim.y) 
        {
            int global_i = row_block * block_row + i;
            Qi[i * embed_dim + d] = (global_i < token_len) ? Q[global_i * embed_dim + d] : 0.0f;
        }
    }
    __syncthreads();

    // Loop over column tiles (K, V)
    for (int col_block = 0; col_block < total_block_col; col_block++) 
    {
        if (col_block * block_col > (row_block + 1) * block_row - 1) 
            break;

        // Load K and V into shared memory
        for (int j = thread_x; j < block_col && (col_block * block_col + j) < token_len; j += blockDim.x) 
        {
            for (int d = thread_y; d < embed_dim; d += blockDim.y) 
            {
                int global_j = col_block * block_col + j;
                Kj[j * embed_dim + d] = (global_j < token_len) ? K[global_j * embed_dim + d] : 0.0f;
                Vj[j * embed_dim + d] = (global_j < token_len) ? V[global_j * embed_dim + d] : 0.0f;
            }
        }
        __syncthreads();

        // Compute QK^T with causal masking
        for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
        {
            for (int j = thread_y; j < block_col && (col_block * block_col + j) < token_len; j += blockDim.y) 
            {
                int global_i = row_block * block_row + i;
                int global_j = col_block * block_col + j;

                float score = 0.0f;
                for (int d = 0; d < embed_dim; d++) 
                {
                    score += Qi[i * embed_dim + d] * Kj[j * embed_dim + d];
                }
                S[i * block_col + j] = (global_j > global_i) ? -INFINITY : score * scale;
            }
        }
        __syncthreads();

        // Online softmax update + output accumulation
        if (thread_y == 0) {
            for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
            {
                int global_i = row_block * block_row + i;
                float old_max = M[global_i];
                float old_sum = L[global_i];
                float current_max = old_max;
                float current_sum = 0.0f;

                // Update max and normalization sum
                for (int j = 0; j < block_col && (col_block * block_col + j) < token_len; j++) 
                {
                    float score = S[i * block_col + j];
                    if (score > current_max) 
                    {
                        float corr = expf(current_max - score);
                        current_sum = current_sum * corr + 1.0f;
                        current_max = score;
                    } 
                    else 
                    {
                        current_sum += expf(score - current_max);
                    }
                }

                float final_max = fmaxf(old_max, current_max);
                float old_corr = expf(old_max - final_max);
                float new_corr = expf(current_max - final_max);
                float norm = old_corr * old_sum + new_corr * current_sum;

                M[global_i] = final_max;
                L[global_i] = norm;

                // Accumulate weighted V
                for (int d = 0; d < embed_dim; d++) 
                {
                    float new_contrib = 0.0f;
                    for (int j = 0; j < block_col && (col_block * block_col + j) < token_len; j++) 
                    {
                        float weight = expf(S[i * block_col + j] - final_max);
                        new_contrib += weight * Vj[j * embed_dim + d];
                    }
                    float old_output = O[global_i * embed_dim + d];
                    O[global_i * embed_dim + d] = (old_corr * old_output + new_contrib) / norm;
                }
            }
        }
        __syncthreads();
    }
}

/* __global__ void flashAttentionForwardWarpOptimized(const float *Q, const float *K, const float *V, float *O, float *L, float *M, const float scale)
{
    extern __shared__ float mem[];
    
    float *Qi = mem;
    float *Kj = mem + block_row * embed_dim;
    float *Vj = mem + block_row * embed_dim + block_col * embed_dim;
    float *S = mem + block_row * embed_dim + 2 * block_col * embed_dim;
    
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int row_block = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
    {
        int global_i = row_block * block_row + i;
        M[global_i] = -INFINITY;
        L[global_i] = 0.0f;
        for (int d = 0; d < embed_dim; d++) 
        {
            O[global_i * embed_dim + d] = 0.0f;
        }
    }

    for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
    {
        for (int d = thread_y; d < embed_dim; d += blockDim.y) 
        {
            int global_i = row_block * block_row + i;
            if (global_i < token_len) 
            {
                Qi[i * embed_dim + d] = Q[global_i * embed_dim + d];
            } 
            else 
            {
                Qi[i * embed_dim + d] = 0.0f;
            }
        }
    }
    __syncthreads();
    
    for (int col_block = 0; col_block < total_block_col; col_block++) 
    {
        for (int j = thread_x; j < block_col && (col_block * block_col + j) < token_len; j += blockDim.x) 
        {
            for (int d = thread_y; d < embed_dim; d += blockDim.y) 
            {
                int global_j = col_block * block_col + j;
                if (global_j < token_len) 
                {
                    Kj[j * embed_dim + d] = K[global_j * embed_dim + d];
                    Vj[j * embed_dim + d] = V[global_j * embed_dim + d];
                } 
                else
                {
                    Kj[j * embed_dim + d] = 0.0f;
                    Vj[j * embed_dim + d] = 0.0f;
                }
            }
        }
        __syncthreads();
        
        for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
        {
            for (int j = thread_y; j < block_col && (col_block * block_col + j) < token_len; j += blockDim.y) 
            {
                int global_i = row_block * block_row + i;
                int global_j = col_block * block_col + j;
                
                float score = 0.0f;
                for (int d = 0; d < embed_dim; d++) 
                {
                    score += Qi[i * embed_dim + d] * Kj[j * embed_dim + d];
                }
                
                if (global_j > global_i) 
                {
                    S[i * block_col + j] = -INFINITY;
                } 
                else 
                {
                    S[i * block_col + j] = score * scale;
                }
            }
        }
        __syncthreads();
        
        //Use warp shuffle for parallel reduction
        if (thread_y == 0) 
        {
            for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
            {
                int global_i = row_block * block_row + i;
                
                float old_max = M[global_i];
                float old_sum = L[global_i];
                
                // Warp-level online softmax
                float local_max = -INFINITY;
                float local_sum = 0.0f;
                
                // Each thread processes multiple elements
                for (int j = lane_id; j < block_col && (col_block * block_col + j) < token_len; j += 32) 
                {
                    float score = S[i * block_col + j];
                    
                    if (score > local_max) 
                    {
                        float correction = expf(local_max - score);
                        local_sum = local_sum * correction + 1.0f;
                        local_max = score;
                    } 
                    else 
                    {
                        local_sum += expf(score - local_max);
                    }
                }
                
                // Warp reduction for max and sum
                for (int offset = 16; offset > 0; offset >>= 1) 
                {
                    float other_max = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
                    float other_sum = __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
                    
                    if (other_max > local_max) 
                    {
                        float correction = expf(local_max - other_max);
                        local_sum = local_sum * correction + other_sum;
                        local_max = other_max;
                    } 
                    else if (other_max < local_max) 
                    {
                        float correction = expf(other_max - local_max);
                        local_sum = local_sum + other_sum * correction;
                    } 
                    else 
                    {
                        local_sum += other_sum;
                    }
                }
                
                if (lane_id == 0) 
                {
                    // Combine with global statistics
                    float final_max = fmaxf(old_max, local_max);
                    float old_correction = expf(old_max - final_max);
                    float new_correction = expf(local_max - final_max);
                    
                    L[global_i] = old_correction * old_sum + new_correction * local_sum;
                    M[global_i] = final_max;
                    
                    // Update output
                    for (int d = 0; d < embed_dim; d++) 
                    {
                        float old_output = O[global_i * embed_dim + d];
                        
                        float new_contrib = 0.0f;
                        for (int j = 0; j < block_col && (col_block * block_col + j) < token_len; j++) 
                        {
                            float weight = expf(S[i * block_col + j] - final_max);
                            new_contrib += weight * Vj[j * embed_dim + d];
                        }
                        
                        O[global_i * embed_dim + d] = (old_correction * old_output + new_contrib) / L[global_i];
                    }
                }
            }
        }
        __syncthreads();
    }
} */

__global__ void flashAttentionBackward(const float *Q, const float *K, const float *V, const float *O, const float *dO, const float *L, const float *M, float *dQ, float *dK, float *dV, const float scale)
{
    extern __shared__ float mem[];

    // Shared memory layout: Qi, Kj, Vj, dOi, score, prob, dS
    float *Qi  = mem;
    float *Kj  = Qi  + block_row * embed_dim;
    float *Vj  = Kj  + block_col * embed_dim;
    float *dOi = Vj  + block_col * embed_dim;
    float *S   = dOi + block_row * embed_dim;
    float *P   = S   + block_row * block_col;
    float *dS  = P   + block_row * block_col;

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int row_block = blockIdx.x;

    // Load Q and dO into shared memory
    for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
    {
        for (int d = thread_y; d < embed_dim; d += blockDim.y) 
        {
            int global_i = row_block * block_row + i;
            Qi[i * embed_dim + d]  = (global_i < token_len) ? Q[global_i * embed_dim + d]  : 0.0f;
            dOi[i * embed_dim + d] = (global_i < token_len) ? dO[global_i * embed_dim + d] : 0.0f;
        }
    }

    // Zero out dQ
    for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
    {
        for (int d = thread_y; d < embed_dim; d += blockDim.y) 
        {
            int global_i = row_block * block_row + i;
            dQ[global_i * embed_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    // Loop over column tiles
    for (int col_block = 0; col_block < total_block_col; col_block++) 
    {
        if (col_block * block_col > (row_block + 1) * block_row - 1) 
            break;

        // Load K and V into shared memory
        for (int j = thread_x; j < block_col && (col_block * block_col + j) < token_len; j += blockDim.x) 
        {
            for (int d = thread_y; d < embed_dim; d += blockDim.y) 
            {
                int global_j = col_block * block_col + j;
                Kj[j * embed_dim + d] = (global_j < token_len) ? K[global_j * embed_dim + d] : 0.0f;
                Vj[j * embed_dim + d] = (global_j < token_len) ? V[global_j * embed_dim + d] : 0.0f;
            }
        }
        __syncthreads();

        // Recompute scores and softmax probs
        for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
        {
            for (int j = thread_y; j < block_col && (col_block * block_col + j) < token_len; j += blockDim.y) 
            {
                int global_i = row_block * block_row + i;
                int global_j = col_block * block_col + j;

                float score = 0.0f;
                for (int d = 0; d < embed_dim; d++) 
                {
                    score += Qi[i * embed_dim + d] * Kj[j * embed_dim + d];
                }
                float scaled_score = (global_j > global_i) ? -INFINITY : score * scale;
                S[i * block_col + j] = scaled_score;
                P[i * block_col + j] = (global_j > global_i) ? 0.0f : expf(scaled_score - M[global_i]) / L[global_i];
            }
        }
        __syncthreads();

        // Compute dV = P^T * dO
        for (int j = thread_x; j < block_col && (col_block * block_col + j) < token_len; j += blockDim.x) 
        {
            for (int d = thread_y; d < embed_dim; d += blockDim.y) 
            {
                float grad_sum = 0.0f;
                for (int i = 0; i < block_row && (row_block * block_row + i) < token_len; i++) 
                {
                    grad_sum += P[i * block_col + j] * dOi[i * embed_dim + d];
                }
                int global_j = col_block * block_col + j;
                if (global_j < token_len) 
                {
                    atomicAdd(&dV[global_j * embed_dim + d], grad_sum);
                }
            }
        }
        __syncthreads();

        // Compute dS
        for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
        {
            int global_i = row_block * block_row + i;
            float diag_term = 0.0f;
            for (int d = 0; d < embed_dim; d++) 
            {
                diag_term += dOi[i * embed_dim + d] * O[global_i * embed_dim + d];
            }
            for (int j = thread_y; j < block_col && (col_block * block_col + j) < token_len; j += blockDim.y) 
            {
                int global_j = col_block * block_col + j;
                if (global_j > global_i) 
                {
                    dS[i * block_col + j] = 0.0f;
                } 
                else 
                {
                    float dot = 0.0f;
                    for (int d = 0; d < embed_dim; d++) 
                    {
                        dot += dOi[i * embed_dim + d] * Vj[j * embed_dim + d];
                    }
                    dS[i * block_col + j] = P[i * block_col + j] * (dot - diag_term);
                }
            }
        }
        __syncthreads();

        // Compute dQ = dS * K
        for (int i = thread_x; i < block_row && (row_block * block_row + i) < token_len; i += blockDim.x) 
        {
            for (int d = thread_y; d < embed_dim; d += blockDim.y) 
            {
                float grad_sum = 0.0f;
                for (int j = 0; j < block_col && (col_block * block_col + j) < token_len; j++) 
                {
                    grad_sum += dS[i * block_col + j] * Kj[j * embed_dim + d];
                }
                int global_i = row_block * block_row + i;
                dQ[global_i * embed_dim + d] += grad_sum * scale;
            }
        }
        __syncthreads();

        // Compute dK = dS^T * Q
        for (int j = thread_x; j < block_col && (col_block * block_col + j) < token_len; j += blockDim.x) 
        {
            for (int d = thread_y; d < embed_dim; d += blockDim.y) 
            {
                float grad_sum = 0.0f;
                for (int i = 0; i < block_row && (row_block * block_row + i) < token_len; i++) 
                {
                    grad_sum += dS[i * block_col + j] * Qi[i * embed_dim + d];
                }
                int global_j = col_block * block_col + j;
                if (global_j < token_len) 
                {
                    atomicAdd(&dK[global_j * embed_dim + d], grad_sum * scale);
                }
            }
        }
        __syncthreads();
    }
}

void FlashAttention( float *h_Q, float *h_K, float *h_V, float *h_O, float *h_dO, float *h_dQ, float *h_dK, float *h_dV)
{
    float *d_Q, *d_K, *d_V, *d_O, *d_L, *d_M;
    float *d_dQ, *d_dK, *d_dV, *d_dO;
    
    size_t matrix_size = token_len * embed_dim * sizeof(float);
    size_t vector_size = token_len * sizeof(float);
    
    cudaMallocManaged(&d_Q, matrix_size);
    cudaMallocManaged(&d_K, matrix_size);
    cudaMallocManaged(&d_V, matrix_size);
    cudaMallocManaged(&d_O, matrix_size);
    cudaMallocManaged(&d_L, vector_size);
    cudaMallocManaged(&d_M, vector_size);
    cudaMallocManaged(&d_dQ, matrix_size);
    cudaMallocManaged(&d_dK, matrix_size);
    cudaMallocManaged(&d_dV, matrix_size);
    cudaMallocManaged(&d_dO, matrix_size);

    memcpy(d_Q, h_Q, matrix_size);
    memcpy(d_K, h_K, matrix_size);
    memcpy(d_V, h_V, matrix_size);
    memcpy(d_dO, h_dO, matrix_size);
    
    cudaMemset(d_dQ, 0, matrix_size);
    cudaMemset(d_dK, 0, matrix_size);
    cudaMemset(d_dV, 0, matrix_size);
    
    float scale = 1.0f / sqrtf(embed_dim);

    int threads_per_block = 512;
    int tile_rows = block_row;

    int x = std::min(tile_rows, threads_per_block);
    int y = std::max(1, threads_per_block / x);

    int grid_rows = (token_len + block_row - 1) / block_row;

    dim3 block_dim(x, y);
    dim3 grid_dim(grid_rows);

    size_t mem_forward = (block_row + 2*block_col) * embed_dim * sizeof(float) + block_row * block_col * sizeof(float);
    size_t mem_backward = 2 * (block_row + block_col) * embed_dim * sizeof(float) + 3 * block_row * block_col * sizeof(float);

    cudaEvent_t start, forward, backward;
    cudaEventCreate(&start);
    cudaEventCreate(&forward);
    cudaEventCreate(&backward);

    cudaEventRecord(start);
    
    flashAttentionForward<<<grid_dim, block_dim, mem_forward>>>(d_Q, d_K, d_V, d_O, d_L, d_M, scale);
    cudaEventRecord(forward);
    cudaDeviceSynchronize();
    
    flashAttentionBackward<<<grid_dim, block_dim, mem_backward>>>(d_Q, d_K, d_V, d_O, d_dO, d_L, d_M, d_dQ, d_dK, d_dV, scale);
    cudaEventRecord(backward);
    cudaDeviceSynchronize();
    
    memcpy(h_O, d_O, matrix_size);
    memcpy(h_dQ, d_dQ, matrix_size);
    memcpy(h_dK, d_dK, matrix_size);
    memcpy(h_dV, d_dV, matrix_size);

    float forward_ms = 0, backward_ms = 0, total_ms = 0;
    cudaEventElapsedTime(&forward_ms, start, forward);
    cudaEventElapsedTime(&backward_ms, forward, backward);
    cudaEventElapsedTime(&total_ms, start, backward);

    std::cout << "Forward Pass Latency:" << forward_ms  << " ms" << std::endl;
    std::cout << "Backward Pass Latency:" << backward_ms << " ms" << std::endl;
    std::cout << "Total Latency:" << total_ms    << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(forward);
    cudaEventDestroy(backward);
    
    cudaFree(d_Q); 
    cudaFree(d_K); 
    cudaFree(d_V); 
    cudaFree(d_O);
    cudaFree(d_L); 
    cudaFree(d_M);
    cudaFree(d_dQ); 
    cudaFree(d_dK); 
    cudaFree(d_dV); 
    cudaFree(d_dO);
}

int main() {
    float *h_Q = new float[token_len * embed_dim];
    float *h_K = new float[token_len * embed_dim];
    float *h_V = new float[token_len * embed_dim];
    float *h_O = new float[token_len * embed_dim];
    float *h_dO = new float[token_len * embed_dim];
    float *h_dQ = new float[token_len * embed_dim];
    float *h_dK = new float[token_len * embed_dim];
    float *h_dV = new float[token_len * embed_dim];

    for (int i = 0; i < token_len * embed_dim; i++) 
    {
        h_Q[i] = 2.0f * rand() / RAND_MAX - 1.0f;
        h_K[i] = 2.0f * rand() / RAND_MAX - 1.0f;
        h_V[i] = 2.0f * rand() / RAND_MAX - 1.0f;
        h_dO[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    }

    FlashAttention(h_Q, h_K, h_V, h_O, h_dO, h_dQ, h_dK, h_dV);
    
    const int print_tokens = std::min(3, token_len);
    const int print_dims = std::min(3, embed_dim);

    auto print_matrix = [&](const char* name, float* matrix) 
    {
        std::cout << "\n" << name << " (showing " << print_tokens << "x" << print_dims << "):" << std::endl;
        for (int i = 0; i < print_tokens; i++) 
        {
            for (int j = 0; j < print_dims; j++) 
            {
                std::cout << matrix[i * embed_dim + j] << " ";
            }
            std::cout << std::endl;
        }
    };

    print_matrix("Query", h_Q);
    print_matrix("Key", h_K);
    print_matrix("Value", h_V);
    print_matrix("Output", h_O);
    print_matrix("Gradient dQ", h_dQ);
    print_matrix("Gradient dK", h_dK);
    print_matrix("Gradient dV", h_dV);

    
    delete[] h_Q; 
    delete[] h_K; 
    delete[] h_V; 
    delete[] h_O;
    delete[] h_dQ; 
    delete[] h_dK; 
    delete[] h_dV; 
    delete[] h_dO;
    
    return 0;
}
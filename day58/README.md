# Day 58: Activation Aware Weight Quantization

This log captures a simplified, end-to-end AWQ pipeline for a multi-layer neural network on the GPU using CUDA. It focuses on activation-driven quantization, tiled compute, and timing of the full pass.

## Objective
Implement and time an AWQ pipeline that:
1. Collects activation statistics from a representative dataset.
2. Searches for optimal per-channel scaling factors \(s_k\) for weights.
3. Quantizes weights to int8.
4. Performs matrix multiplication with on-the-fly dequantization.
5. Measures end-to-end performance.

## Key Learnings
- **AWQ algorithm.** Optimal weight quantization depends on input activations. By analyzing activation statistics, better per-channel scaling factors reduce error on important channels.
- **Mixed host–device execution.** Some steps run on GPU (stats, matmul). Others run on CPU (search for quant params), reflecting common deployment patterns.
- **Parallel statistics gathering.** Atomic additions accumulate per-column |x| and x² efficiently on GPU.
- **Tiled dequantized matmul.** A tiled kernel performs GEMM while dequantizing on the fly. Shared memory improves locality.
- **Quantization ↔ dequantization loop.** FP32 → int8 using \(s_k\) and per-row scales; dequantize during compute using the same factors.
- **Performance measurement.** cudaEvent timing covers all layers and host steps to give true pipeline time.

## Implementation Details

### Constants
- **N**: 1024 (square matrices)
- **TILE**: 16 (tile size for shared-memory matmul)
- **LAYERS**: 3 (simulated depth)

### GPU Kernels
- **`init`**  
  Initializes matrices with random FP32 using `curand_uniform` to simulate weights and activations.
- **`relu`**  
  Element-wise ReLU: `x = max(0, x)`.
- **`accum_stats`**  
  Iterates over activation matrix `x` and accumulates per-column statistics using `atomicAdd`: sum of absolute values and sum of squares.
- **`quantize_awq`**  
  Quantizes FP32 weight matrix `B` to `int8_t Bq`. The formula applies both per-column AWQ scale `s_k` and per-row scale `scale_b`.
- **`dequantizeTiled`**  
  Performs tiled GEMM with on-the-fly dequantization. Loads tiles of activations `Asub` and quantized weights `Bsub` into shared memory. Also loads `inv_s` and `scale_b` for immediate use. Inner loop multiplies while applying dequantization factors.

### Host Functions
- **`compute_s`**  
  CPU-side approximate search for \(s_k\). Iterates over a small set of gamma values and computes total quantization error to select the best per-channel scalers.
- **`compute_scale_b_host`**  
  Computes per-row scaling factors `scale_b` based on the chosen \(s_k\).

### `main` Flow
- Allocates unified memory (`cudaMallocManaged`) for matrices and statistics arrays.
- Defines grid and block dimensions for all kernels.
- Initializes input activations `a_fp32` and per-layer weights `b_fp32[l]`.
- For each of **3 layers**:
  1. `accum_stats` on current activations.
  2. `compute_s` on host.
  3. `compute_scale_b_host` on host.
  4. `quantize_awq` to produce `Bq`.
  5. `dequantizeTiled` to multiply with on-the-fly dequantization.
  6. `relu` for nonlinearity.
- Uses `cudaEvent` to measure total time covering kernels and host-side work.

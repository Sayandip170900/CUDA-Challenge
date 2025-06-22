# Day 57: Multi-GPU CUDA Kernel for 2D Convolution with Shared Memory and Nsight Profiling

**Objective:**
- **Implement Efficient 2D Convolution Across Multiple GPUs:** Design a CUDA-based multi-stream multi-GPU pipeline that performs 2D convolution using shared memory tiling for RGB images.
- **Overlap Data Movement with Compute:** Use pinned memory and CUDA streams to overlap H2D/D2H transfers with kernel execution.
- **Profile Kernel and System Behavior:** Evaluate GPU throughput, kernel execution latency, memory transfer patterns, and system-level performance using Nsight Systems.

---

### CUDA Kernel and Optimization
- **Shared Memory Tiling:**
    - Implemented 2D convolution using `IN_TILE_DIM = 32` with a `RADIUS = 1` filter window.
    - Tiled `tile[IN_TILE_DIM][IN_TILE_DIM]` and `filter[FILTER_SIZE][FILTER_SIZE]` were loaded into shared memory.
- **Streamed GPU Execution:**
    - Each GPU uses `NUM_STREAMS = 2` to overlap async transfers and kernel execution.
    - Stream cycling allows concurrent H2D → kernel → D2H pipelines.
- **Pinned Host Memory:**
    - Used `cudaMallocHost` for input/output host buffers to support non-blocking memory transfers.
- **Dynamic Shared Memory:**
    - Avoided memory bottlenecks by localizing intermediate loads within the kernel using `__syncthreads()` for tile synchronization.

### Pipeline Details
- **Multi-GPU Launch:** Each GPU spawns a worker thread.
- **Task Queue:** A bounded queue distributes pre-decoded OpenCV images to GPU workers.
- **Memory-Aware Prefetching:** A dedicated prefetch thread filters and decodes input images in RAM before GPU upload.

---

## Kernel-Level Profiling Results

### CUDA GPU Kernel Summary

| Kernel Name                                   | Time (%) | Total Time (ns) | Avg (ns) | Min (ns) | Max (ns) | Calls |
|----------------------------------------------|----------|------------------|----------|----------|----------|-------|
| `conv2D(float*, float*, float*, int, int)`   | 91.8     | 6,941,948        | 34,537   | 3,456    | 231,779  | 201   |
| `he_normal(float*, unsigned int)`            | 8.2      | 617,125          | 77,140   | 72,289   | 89,441   | 8     |

---

### CUDA Memory Transfer Summary (by Time)

| Operation                  | Total Time (ns) | Count | Avg (ns) | Max (ns) | Time (%) |
|----------------------------|------------------|--------|----------|----------|----------|
| Host-to-Device             | 106,649,284      | 201    | 530,593  | 4,038,415| 50.8     |
| Device-to-Host             | 103,191,548      | 201    | 513,391  | 3,508,514| 49.2     |

---

### CUDA Memory Transfer Summary (by Size)

| Operation                  | Total (MB) | Count | Avg (MB) | Max (MB) |
|----------------------------|------------|--------|----------|----------|
| Host-to-Device             | 2,060.36   | 201    | 10.251   | 71.361   |
| Device-to-Host             | 2,060.36   | 201    | 10.251   | 71.361   |

---

## GPU-Wise Execution Summary

| GPU ID | Processed | Total Time (s) | Avg/Image (s) | Throughput (img/s) |
|--------|-----------|----------------|----------------|---------------------|
| 0      | 10        | 1.38132        | 0.13813        | 7.24                |
| 1      | 10        | 1.60363        | 0.16036        | 6.24                |
| 2      | 7         | 1.32394        | 0.18913        | 5.29                |
| 3      | 8         | 1.11546        | 0.13943        | 7.17                |
| 4      | 8         | 1.20238        | 0.15030        | 6.65                |
| 5      | 5         | 1.70777        | 0.34155        | 2.93                |
| 6      | 10        | 1.16663        | 0.11666        | 8.57                |
| 7      | 9         | 1.81637        | 0.20182        | 4.95                |

---

## Overall Summary

```text
Total Processing Time: 5.39573 seconds
Total Images:          68
Total GPUs:            8
Overall Throughput:    12.60 images/sec

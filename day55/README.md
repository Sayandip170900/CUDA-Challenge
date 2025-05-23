# Day 55: FlashAttention: Fused CUDA Kernel for Softmax and MatMul in Transformer Attention

**Objective:**
- **Implement Flash Attention Forward and Backward in CUDA:** Develop GPU kernels that implement the Flash Attention algorithm efficiently, covering both the forward and backward passes.
- **Reduce Memory Bottlenecks Using Shared Memory:** Use shared memory (SRAM) to minimize global memory accesses, enabling better throughput for attention computations over long sequences.
- **Measure Kernel Latency:** Evaluate the performance of both passes using CUDA timing APIs to understand execution efficiency.

**Key Learnings:**
- **Flash Attention Algorithm:**
    - Learned the principle behind Flash Attention, which improves efficiency by reducing high-bandwidth memory traffic and using SRAM to store intermediate matrices like queries, keys, and values.
- **Tiling for Memory Efficiency:**
    - Implemented a tiled approach in CUDA where blocks of the input (Q, K, V) are loaded into shared memory and reused across threads to reduce global memory access.
- **Shared Memory Management:**
    - Used `extern __shared__` memory to dynamically partition shared memory into segments for Qi, Kj, Vj, S, P, and dS.
- **Online Softmax Computation:**
    - Implemented numerically stable softmax using online max (M) and normalization factor (L) computation. This avoids storing the full attention matrix.
- **Backward Pass Implementation:**
    - Developed a backward pass that computes gradients for dQ, dK, and dV using intermediate values from the forward pass, minimizing recomputation and memory overhead.
- **Atomic Operations:**
    - Used `atomicAdd()` for thread-safe accumulation of dK and dV across blocks, ensuring correctness under parallel updates.
- **CUDA Kernel Structure for Complex Algorithms:**
    - Structured multiple shared memory buffers and tiling loops with proper synchronization (`__syncthreads()`), enabling efficient large-scale computation inside a single kernel.
- **Performance Measurement:**
    - Measured forward, backward, and total latency using `cudaEventRecord` and `cudaEventElapsedTime` for precise benchmarking.

**Code Implementation Details:**

- **Constants:**
    - `SMEM`: Total shared memory allocated per block (48 KB).
    - `token_len`: Sequence length of 128.
    - `embed_dim`: Embedding dimension of 32.
    - `block_row`, `block_col`: Tile dimensions set to 32×32 for efficient shared memory tiling.
    - `total_block_row`, `total_block_col`: Computed to ensure full sequence coverage based on `block_row` and `block_col`.
    - `static_assert`: Ensures shared memory usage stays within the hardware-imposed limit.

- **`flashAttentionForward` Kernel:**
    - Computes QKᵀ scores in tiled blocks.
    - Performs online softmax row-wise using updated max and sum values (M, L).
    - Applies attention weights to V and accumulates the result in `O` using stable normalization.
    - Uses shared memory for Qi, Kj, Vj, and attention score matrix S.

- **`flashAttentionBackward` Kernel:**
    - Recomputes QKᵀ and softmax weights.
    - Uses intermediate values from forward pass (M and L) to apply gradients.
    - Computes gradients `dQ`, `dK`, and `dV`, using matrix multiplies and reductions.
    - Accumulates `dK` and `dV` using `atomicAdd()` due to shared global memory writes across threads.

- **`FlashAttention` Host Function:**
    - Allocates memory for Q, K, V, dO, and outputs using `cudaMallocManaged`.
    - Computes shared memory size dynamically for forward and backward kernels.
    - Launches forward and backward kernels using calculated block/grid dimensions.
    - Uses `cudaEventRecord` to measure:
        - Forward pass latency.
        - Backward pass latency.
        - Total latency.
    - Frees all allocated device memory.

- **`main` Function:**
    - Initializes host-side Q, K, V, and dO with random values.
    - Calls `FlashAttention()` to run the GPU kernels.
    - Prints a 3x3 slice of the matrices (Q, K, V, O, dQ, dK, dV) for quick inspection.

**Output Analysis:**

- **Latency Results:**
    - **Forward Pass Latency:** 0.6992 ms
    - **Backward Pass Latency:** 0.68608 ms
    - **Total Latency:** 1.38528 ms

- **Sample Matrix Values (3x3):**

```text
Query:
0.680375 0.823295 -0.444451
0.499542 0.168977 -0.74905
0.461459 0.841829 0.0648819

Key:
-0.211234 -0.604897 0.10794
-0.262673 -0.511175 0.586941
-0.343251 0.369513 -0.824713

Value:
0.566198 -0.329554 -0.0452059
-0.411679 -0.69522 -0.671796
0.480877 0.306261 -0.479006

Output:
0.00981674 0.013525 0.0224509
0.000579609 0.0185876 0.0159275
0.0158678 0.0192494 0.0114489

Gradient dQ:
0.00846144 0.00487566 0.00132402
-0.00142898 -0.00505878 -0.0172243
-0.0118888 0.0134579 0.0207399

Gradient dK:
0.00725392 -0.0102753 -0.00365077
0.012385 0.0102099 0.0236704
0.0076368 0.0120091 0.0135568

Gradient dV:
-0.0712795 -0.00727661 -0.00910686
-0.0345493 -0.0283635 0.00945967
-0.0471181 0.000656001 -0.015904
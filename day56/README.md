## Day 56: Multi-GPU Kernel Launch in CUDA

**Objective:**
This project demonstrates how to explicitly select and launch CUDA kernels on multiple GPUs available in a system. The goal is to verify multi-GPU visibility and basic kernel execution across different devices.

**Key Learnings:**

* **Enumerating GPUs:** Learned how to query the number of available CUDA-enabled GPUs in the system using `cudaGetDeviceCount()`.
* **Device Selection:** Understood the importance of `cudaSetDevice(deviceId)` to explicitly select which GPU subsequent CUDA operations (like kernel launches, memory allocations, etc.) will target.
* **Multi-GPU Kernel Launch:** Demonstrated launching the same kernel independently on different GPUs within a single host application loop.
* **Thread and Block Identification:** Utilized built-in CUDA variables (`blockIdx.x`, `threadIdx.x`) within the kernel to identify and print the unique ID of each thread and its block, confirming successful execution on the targeted device.
* **Host-Device Synchronization:** Employed `cudaDeviceSynchronize()` after each kernel launch to ensure that all work on the currently selected device completes before proceeding to the next device. This is crucial for ordered output and preventing race conditions in simple multi-GPU loops.

---
**Code Implementation Details:**

* **`multi_gpu` Global Function:**
    * A simple CUDA kernel that takes an integer `idx` as an argument.
    * Inside the kernel, it uses `printf` to display a message indicating which GPU (`idx`), block (`blockIdx.x`), and thread (`threadIdx.x`) the message originated from.
    * This function serves as a basic "hello world" for multi-GPU execution.

* **`main` Function:**
    * **Get Device Count:** Calls `cudaGetDeviceCount(&n)` to determine the total number of GPUs (`n`) present in the system.
    * **Iterate Through GPUs:** A `for` loop iterates from `0` to `n-1`, representing each available GPU.
    * **Set Device:** Inside the loop, `cudaSetDevice(i)` is called. This function sets the current CUDA device for the host thread to GPU `i`. All subsequent CUDA API calls will operate on this device until `cudaSetDevice` is called again.
    * **Kernel Launch:** The `multi_gpu` kernel is launched on the currently selected device (`i`). The kernel is configured to launch with 2 blocks and 2 threads per block (`<<<2, 2>>>`). The loop variable `i` is passed as an argument to the kernel, so each kernel knows which GPU it's running on.
    * **Synchronize Device:** `cudaDeviceSynchronize()` is called after each kernel launch. This ensures that the kernel on the *current* device completes execution and its `printf` statements are flushed before the loop moves to the next GPU and potentially sets a new device.

---
**Output Analysis:**

The provided output confirms the successful execution of the program on a system with 8 A100 GPUs on SDSU DGX server:

```text
Total GPUs: 8

Launching kernel on GPU 0
Hello from GPU 0 (block 1, thread 0)
Hello from GPU 0 (block 1, thread 1)
Hello from GPU 0 (block 0, thread 0)
Hello from GPU 0 (block 0, thread 1)

Launching kernel on GPU 1
Hello from GPU 1 (block 1, thread 0)
Hello from GPU 1 (block 1, thread 1)
Hello from GPU 1 (block 0, thread 0)
Hello from GPU 1 (block 0, thread 1)

Launching kernel on GPU 2
Hello from GPU 2 (block 0, thread 0)
Hello from GPU 2 (block 0, thread 1)
Hello from GPU 2 (block 1, thread 0)
Hello from GPU 2 (block 1, thread 1)

Launching kernel on GPU 3
Hello from GPU 3 (block 0, thread 0)
Hello from GPU 3 (block 0, thread 1)
Hello from GPU 3 (block 1, thread 0)
Hello from GPU 3 (block 1, thread 1)

Launching kernel on GPU 4
Hello from GPU 4 (block 1, thread 0)
Hello from GPU 4 (block 1, thread 1)
Hello from GPU 4 (block 0, thread 0)
Hello from GPU 4 (block 0, thread 1)

Launching kernel on GPU 5
Hello from GPU 5 (block 1, thread 0)
Hello from GPU 5 (block 1, thread 1)
Hello from GPU 5 (block 0, thread 0)
Hello from GPU 5 (block 0, thread 1)

Launching kernel on GPU 6
Hello from GPU 6 (block 0, thread 0)
Hello from GPU 6 (block 0, thread 1)
Hello from GPU 6 (block 1, thread 0)
Hello from GPU 6 (block 1, thread 1)

Launching kernel on GPU 7
Hello from GPU 7 (block 1, thread 0)
Hello from GPU 7 (block 1, thread 1)
Hello from GPU 7 (block 0, thread 0)
Hello from GPU 7 (block 0, thread 1)
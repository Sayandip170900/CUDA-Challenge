#include <stdio.h>
#include <cuda_runtime.h>

__global__ void multi_gpu(int idx) 
{
    printf("Hello from GPU %d (block %d, thread %d)\n", idx, blockIdx.x, threadIdx.x);
}

int main() {
    int n;
    cudaGetDeviceCount(&n);
    printf("Total GPUs: %d\n", n);

    for (int i = 0; i < n; ++i) 
    {
        cudaSetDevice(i);

        printf("Launching kernel on GPU %d\n", i);

        multi_gpu<<<2, 2>>>(i);

        cudaDeviceSynchronize();
    }

    return 0;
}

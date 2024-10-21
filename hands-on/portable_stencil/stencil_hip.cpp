#include <hip/hip_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256
#define RADIUS 3

__global__ void stencil_1d(const int *in, int *out, int n) {
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

  int g_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (g_index < n) {
    int s_index = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    temp[s_index] = in[g_index];
    if (threadIdx.x < RADIUS) {
      temp[s_index - RADIUS] = g_index - RADIUS < 0 ? 0 : in[g_index - RADIUS];
      temp[s_index + BLOCK_SIZE] = g_index + BLOCK_SIZE < n ? in[g_index + BLOCK_SIZE] : 0;
    }
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
      result += temp[s_index + offset];
    }

    // Store the result
    out[g_index] = result;
  }
}

int main() {
  int n = 1024;
  int size = n * sizeof(int);

  int *h_in, *h_out;
  hipStream_t stream;

  // Allocating pinned (page-locked) host memory
  hipHostMalloc(&h_in, size);
  hipHostMalloc(&h_out, size);

  for (int i = 0; i < n; i++) {
    h_in[i] = i;
  }

  int *d_in, *d_out;
  hipStreamCreate(&stream);
  hipMallocAsync(&d_in, size, stream);
  hipMallocAsync(&d_out, size, stream);

  // Asynchronous memory copy to device
  hipMemcpyAsync(d_in, h_in, size, hipMemcpyHostToDevice, stream);

  int blockSize = BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;

  // Asynchronous kernel launch
  hipLaunchKernelGGL(stencil_1d, dim3(gridSize), dim3(blockSize), 0, stream, d_in, d_out, n);

  // Asynchronous memory copy back to host
  hipMemcpyAsync(h_out, d_out, size, hipMemcpyDeviceToHost, stream);

  // Wait for stream to complete
  hipStreamSynchronize(stream);

  // Verify the result
  for (int i = 0; i < n; i++) {
    std::cout << h_out[i] << " ";
  }

  // Freeing pinned host memory
  hipHostFree(h_in);
  hipHostFree(h_out);

  hipFreeAsync(d_in, stream);
  hipFreeAsync(d_out, stream);
  hipStreamDestroy(stream);

  return 0;
}

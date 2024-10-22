#include <hip/hip_runtime.h>
#include <iostream>

constexpr int error_exit_code = -1;

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(error_exit_code);                                                     \
        }                                                                                   \
    }

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
  HIP_CHECK(hipHostMalloc(&h_in, size));
  HIP_CHECK(hipHostMalloc(&h_out, size));

  for (int i = 0; i < n; i++) {
    h_in[i] = i;
  }

  int *d_in, *d_out;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMallocAsync(&d_in, size, stream));
  HIP_CHECK(hipMallocAsync(&d_out, size, stream));

  // Asynchronous memory copy to device
  HIP_CHECK(hipMemcpyAsync(d_in, h_in, size, hipMemcpyHostToDevice, stream));

  int blockSize = BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;

  // Asynchronous kernel launch
  hipLaunchKernelGGL(stencil_1d, dim3(gridSize), dim3(blockSize), 0, stream, d_in, d_out, n);

  // Asynchronous memory copy back to host
  HIP_CHECK(hipMemcpyAsync(h_out, d_out, size, hipMemcpyDeviceToHost, stream));

  // Wait for stream to complete
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify the result
  for (int i = 0; i < n; i++) {
    std::cout << h_out[i] << " ";
  }

  // Freeing pinned host memory
  HIP_CHECK(hipHostFree(h_in));
  HIP_CHECK(hipHostFree(h_out));

  HIP_CHECK(hipFreeAsync(d_in, stream));
  HIP_CHECK(hipFreeAsync(d_out, stream));
  HIP_CHECK(hipStreamDestroy(stream));

  return 0;
}

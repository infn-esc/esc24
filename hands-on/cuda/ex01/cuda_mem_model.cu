// C++ standard headers
#include <cassert>
#include <iostream>
#include <vector>

// CUDA headers
#include <cuda_runtime.h>

// local headers
#include "cuda_check.h"

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main()
{
  // Choose one CUDA device
  CUDA_CHECK(cudaSetDevice(MYDEVICE));

  // Create a CUDA stream to execute asynchronous operations on this device
  cudaStream_t queue;
  CUDA_CHECK(cudaStreamCreate(&queue));

  // Pointer and dimension for host memory
  int dimA = 8;
  std::vector<float> h_a(dimA);

  // Allocate and initialize host memory
  for (int i = 0; i < dimA; ++i)
  {
    h_a[i] = i;
  }

  // Pointers for device memory
  float *d_a, *d_b;

  // Part 1 of 5: allocate the device memory
  size_t memSize = dimA * sizeof(float);
  CUDA_CHECK(cudaMallocAsync(___));
  CUDA_CHECK(cudaMallocAsync(___));

  // Part 2 of 5: host to device memory copy
  // Hint: the raw pointer to the underlying array of a vector
  // can be obtained by calling std::vector<T>::data()
  CUDA_CHECK(cudaMemcpyAsync(___));

  // Part 3 of 5: device to device memory copy
  CUDA_CHECK(cudaMemcpyAsync(___));

  // Clear the host memory
  std::fill(h_a.begin(), h_a.end(), 0);

  // Part 4 of 5: device to host copy
  CUDA_CHECK(cudaMemcpyAsync(___));

  // Part 5 of 5: free the device memory
  CUDA_CHECK(cudaFreeAsync(___));
  CUDA_CHECK(cudaFreeAsync(___));

  // Wait for all asynchronous operations to complete
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // Verify the data on the host is correct
  for (int i = 0; i < dimA; ++i)
  {
    assert(h_a[i] == (float)i);
  }

  // Destroy the CUDA stream
  CUDA_CHECK(cudaStreamDestroy(queue));

  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  std::cout << "Correct!" << std::endl;

  return 0;
}

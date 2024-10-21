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

// Part 3 of 5: implement the kernel
__global__ void myFirstKernel(___)
{
  ___
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  CUDA_CHECK(cudaSetDevice(MYDEVICE));

  // Create a CUDA stream to execute asynchronous operations on this device
  cudaStream_t queue;
  CUDA_CHECK(cudaStreamCreate(&queue));

  // Your problem's size
  int N = 64;

  // Define the grid and block size
  int numThreadsPerBlock = 8;
  int numBlocks          = N / numThreadsPerBlock;

  // Allocate and initialize host memory
  // hint: the vector is empty, you might want to allocate some memory ...
  std::vector<int> h_a;

  // Pointer for the device memory
  int* d_a;

  // Part 1 of 5: allocate the device memory
  size_t memSize = N * sizeof(int);
  CUDA_CHECK(cudaMallocAsync(___));

  // Part 2 of 5: configure and launch kernel
  myFirstKernel<<<___, ___, 0, queue>>>(___);
  //Check for any errors that occurred during kernel launch
  CUDA_CHECK(cudaGetLastError());

  // Part 4 of 5: copy data from device to host asynchronously
  CUDA_CHECK(cudaMemcpyAsync(___));

  // Free the device memory
  CUDA_CHECK(cudaFreeAsync(d_a, queue));

  // Wait for all asynchronous operations to complete
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // Part 5 of 5: verify that the data returned to the host is correct
  for (int i = 0; i < N; ++i) {
    assert(h_a[i] == i + 42);
  }

  // Destroy the CUDA stream
  CUDA_CHECK(cudaStreamDestroy(queue));

  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  std::cout << "Correct, good work!" << std::endl;

}

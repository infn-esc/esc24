// C++ standard headers
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// CUDA headers
#include <cuda_runtime.h>

// local headers
#include "cuda_check.h"

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

// Part 4 of 8: implement the kernel
__global__ void block_sum(const int* input,
                          int* per_block_results,
                          const size_t n)
{
  // fill me
  __shared__ int sdata[choose_your_favorite_size_here];
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(void)
{
  std::random_device rd; // Will be used to obtain a seed for the random engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(-10, 10);
  // Create array of 256ki elements
  const int num_elements = 1 << 18;
  // Generate random input on the host
  std::vector<int> h_input(num_elements);
  for (auto& elt : h_input) {
    elt = distrib(gen);
  }

  const int host_result = std::accumulate(h_input.begin(), h_input.end(), 0);
  std::cerr << "Host sum: " << host_result << std::endl;

  // Part 1 of 8: choose a device and create a CUDA stream

  // Part 2 of 8: copy the input data to device memory
  int* d_input;

  // Part 3 of 8: allocate memory for the partial sums
  // How much space does it need?
  int* d_partial_sums_and_total;

  // Part 5 of 8: launch one kernel to compute, per-block, a partial sum.
  // How much shared memory does it need?
  block_sum<<<num_blocks, block_size>>>(d_input, d_partial_sums_and_total,
                                        num_elements);
  CUDA_CHECK(cudaGetLastError());

  // Part 6 of 8: compute the sum of the partial sums
  block_sum<<<>>>();
  CUDA_CHECK(cudaGetLastError());

  // Part 7 of 8: copy the result back to the host
  int device_result = 0;

  std::cout << "Device sum: " << device_result << std::endl;

  // Part 8 of 8: deallocate device memory and destroy the CUDA stream

  return 0;
}

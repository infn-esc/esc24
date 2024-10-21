#include <CL/sycl.hpp>
#include <iostream>

#define BLOCK_SIZE 256
#define RADIUS 3

class StencilKernel;

void stencil_1d(sycl::queue &q, const int *in, int *out, int n) {
  q.submit([&](sycl::handler &h) {
    sycl::range<1> global_range(n);
    sycl::range<1> local_range(BLOCK_SIZE);

    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local>
        temp_acc(BLOCK_SIZE + 2 * RADIUS, h);

    h.parallel_for<class StencilKernel>(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> item) {
      int g_index = item.get_global_id(0);
      int s_index = item.get_local_id(0) + RADIUS;

      if (g_index < n) {
        // Read input elements into shared memory
        temp_acc[s_index] = in[g_index];
        if (item.get_local_id(0) < RADIUS) {
          temp_acc[s_index - RADIUS] = (g_index - RADIUS < 0) ? 0 : in[g_index - RADIUS];
          temp_acc[s_index + BLOCK_SIZE] = (g_index + BLOCK_SIZE < n) ? in[g_index + BLOCK_SIZE] : 0;
        }
        item.barrier(sycl::access::fence_space::local_space);

        // Apply the stencil
        int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
          result += temp_acc[s_index + offset];
        }

        // Store the result
        out[g_index] = result;
      }
    });
  }).wait();
}

int main() {
  int n = 1024;
  int size = n * sizeof(int);

  int *h_in = sycl::malloc_host<int>(n, sycl::default_selector());
  int *h_out = sycl::malloc_host<int>(n, sycl::default_selector());

  for (int i = 0; i < n; i++) {
    h_in[i] = i;
  }

  sycl::queue q(sycl::default_selector{});

  // Allocate device memory
  int *d_in = sycl::malloc_device<int>(n, q);
  int *d_out = sycl::malloc_device<int>(n, q);

  // Copy memory to device
  q.memcpy(d_in, h_in, size).wait();

  // Execute the kernel
  stencil_1d(q, d_in, d_out, n);

  // Copy result back to host
  q.memcpy(h_out, d_out, size).wait();

  // Verify the result
  for (int i = 0; i < n; i++) {
    std::cout << h_out[i] << " ";
  }

  // Free memory
  sycl::free(d_in, q);
  sycl::free(d_out, q);
  sycl::free(h_in, q);
  sycl::free(h_out, q);

  return 0;
}

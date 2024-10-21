# GPU Portability
#### Mapping CUDA Concepts to Other Models

- Thread Organization
  - CUDA: Grids, blocks, and threads.
  - OpenMP: Teams and threads.
  - SYCL: ND-ranges, work groups, and work items.
  - HIP: Similar to CUDA, with minor syntax changes.

- Kernel Execution
  - CUDA: __global__ functions launched with <<<>>> syntax.
  - OpenMP: #pragma omp target directives to offload code.
  - SYCL: Using queue.submit() with lambda functions.
  - - HIP: __global__ functions launched with hipLaunchKernelGGL.

- Memory Management
  - Explicit Control: All models require managing data movement between host and device.
  - Unified Memory Options: Some models offer unified memory spaces to simplify development.

- Memory Allocation
  - CUDA: cudaMalloc(&ptr, size);
  - HIP: hipMalloc(&ptr, size);
  - SYCL: malloc_device<int>(size, queue);
  - OpenMP: Memory mapped with map clauses.

### Kernel Launch Variations

#### CUDA

Launching kernels with execution configuration: `kernel<<<gridDim, blockDim>>>(args);`

#### HIP

Similar syntax with slight differences: `hipLaunchKernelGGL(kernel, gridDim, blockDim, sharedMem, stream, args);`

#### SYCL

Using command groups and lambda expressions:

```cpp
queue.submit([&](handler &h) {
    h.parallel_for(nd_range<1>(globalRange, localRange), [=](nd_item<1> item) {
    // Kernel code
    });
});
```

#### OpenMP

Offloading code blocks with pragmas:

```cpp

#pragma omp target teams distribute parallel for
for (int i = 0; i < N; i++) {
  // Loop body
}
```

#### Examples

Have a look in the `hands-on/portable_stencil` directory for examples of CUDA, HIP, SYCL, and OpenMP code.


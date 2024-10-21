---
title: Introduction to CUDA
layout: main
section: parallelism
---


The CUDA Runtime API reference manual is a very useful source of information:
<a href="http://docs.nvidia.com/cuda/cuda-runtime-api/index.html" target="_blank">http://docs.nvidia.com/cuda/cuda-runtime-api/index.html</a>

If you haven't already done it, load the CUDA module

```shell
$ module load compilers/cuda-12.1

```

Check that your environment is correctly configured to compile CUDA code by running:
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

Compile and run the `deviceQuery` application:
```bash
$ cd hands-on/cuda/utils/deviceQuery
$ make
```

You can get some useful information about the features and the limits that you will find on the device you will be running your code on. For example:

```shell
$ ./deviceQuery 
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Tesla V100-SXM2-32GB"
  CUDA Driver Version / Runtime Version          11.4 / 11.4
  CUDA Capability Major/Minor version number:    7.0
  Total amount of global memory:                 32510 MBytes (34089730048 bytes)
  (80) Multiprocessors, ( 64) CUDA Cores/MP:     5120 CUDA Cores
  GPU Max Clock rate:                            1530 MHz (1.53 GHz)
  Memory Clock rate:                             877 Mhz
  Memory Bus Width:                              4096-bit
  L2 Cache Size:                                 6291456 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        98304 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 6 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 58 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.4, CUDA Runtime Version = 11.4, NumDevs = 1
Result = PASS
```

Some of you are sharing the same machine and some time measurements can be influenced by other users running at the very same moment. It can be necessary to run time measurements multiple times.

### Exercise 1: CUDA Memory Model

In this exercise you will learn what heterogeneous memory model means, by demonstrating the difference between host and device memory spaces.

1. Allocate device memory;
2. Copy the host array h_a to d_a on the device;
3. Copy the device array d_a to the device array d_b;
4. Copy the device array d_b to the host array h_a;
5. Free the memory allocated for d_a and d_b.
6. Compile and run the program by running:

```bash
$ nvcc cuda_mem_model.cu -o ex01
$ ./ex01
```

7. Repeat the exercise using CUDA Streams with `cudaMallocAsync`, `cudaFreeAsync` and `cudaMemcpyAsync`.
   1. Hint: Remember to check that asynchronous operations have completed before using their results.
8. Measure the interconnection bandwidth between CPU and GPU by measuring the time needed to transfer an array of `100 MB` in size.

### Exercise 2: Launch a kernel

By completing this exercise you will learn how to configure and launch a simple CUDA kernel.

1. Allocate device memory;
2. Configure the kernel to run using a one-dimensional grid of one-dimensional blocks (i.e. using only the `x` coordinate for `blockIdx` and `threadIdx`);
3. Each GPU thread should set one element of the array to:

   `d_a[i] = i + 42;`
4. Copy the results to the host memory;
5. Check the correctness of the results

### Exercise 3: Two-dimensional grid

M is a matrix of NxN integers.

1. Set N=4
2. Write a kernel that sets each element of the matrix to its linear index (e.g. M[2,3] = 2*N + 3), by making use of two-dimensional grid and blocks. (Two-dimensional means using the x and y coordinates).
3. Copy the result to the host and check that it is correct.
4. Try with a rectangular matrix 19x67. 

Hint: check the kernel launch parameters.
Hint: fix the number of threads per block in each dimension and find the number of blocks needed to cover the full matrix. Pay attention not to write or read out of the matrix boundaries.

### Exercise 4: Parallel Reduction

Given an array `a[N]`, the reduction sum `Sum` of a is the sum of all its elements: `Sum=a[0]+a[1]+...a[N-1]`.

1. Implement a block-wise parallel reduction (using the shared memory).
2. For each block, save the partial sum.
3. Sum all the partial sums together.
4. Check the result comparing with the host result.
5. Measure the throughput of your reduction kernel using CUDA Events (see exercise 4)

* Bonus: Can you implement a one-step reduction? Measure and compare the throughput of the two versions.
* Challenge: The cumulative sum of an array `a[N]` is another array `b[N]`, the sum of prefixes of `a`:
`b[i] = a[0] + a[1] + … + a[i]`. Implement a cumulative sum kernel assuming that the size of the input array is multiple of the block size.


### Challenge: Histogram

The purpose of this lab is to implement an efficient histogramming algorithm for an input array of integers within a given range.

Each integer will map into a single bin, so the values will range from 0 to (NUM_BINS - 1).

The histogram bins will use unsigned 32-bit counters that must be saturated at 127 (i.e. no roll back to 0 allowed).

The input length can be assumed to be less than 2ˆ32. `NUM_BINS` is fixed at 4096 for this lab.
This can be split into two kernels: one that does a histogram without saturation, and a final kernel that cleans up the bins if they are too large. These two stages can also be combined into a single kernel.

### Atomics <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions" target="_blank">[1]</a>

An atomic function performs a read-modify-write atomic operation on one 32-bit or 64-bit word residing in global or shared memory.
The operation is atomic in the sense that it is guaranteed to be performed without interference from other threads.

```C++
int atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAdd(unsigned long long int* address,
                                 unsigned long long int val);
float atomicAdd(float* address, float val);
double atomicAdd(double* address, double val);
__half2 atomicAdd(__half2 *address, __half2 val);
__half atomicAdd(__half *address, __half val);
```

reads the 16-bit, 32-bit or 64-bit word old located at the address address in global or shared memory, computes (old + val), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns old.

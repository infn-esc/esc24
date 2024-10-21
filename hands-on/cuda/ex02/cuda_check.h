#ifndef cuda_check_h
#define cuda_check_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// CUDA headers
#include <cuda_runtime.h>

inline void cuda_check(const char* file, int line, const char* cmd, cudaError_t result) {
  if (__builtin_expect(result == cudaSuccess, true))
    return;

  std::ostringstream out;
  out << "\n";
  out << file << ", line " << line << ":\n";
  out << "CUDA_CHECK(" << cmd << ");\n";
  out << cudaGetErrorName(result) << ": " << cudaGetErrorString(result);
  throw std::runtime_error(out.str());
}

#define CUDA_CHECK(ARG) (cuda_check(__FILE__, __LINE__, #ARG, (ARG)))

#endif  // cuda_check_h

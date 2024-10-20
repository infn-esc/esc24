
#include <pybind11/pybind11.h>
#include <memory>

namespace py = pybind11;

// Step 1: Define the Kernel abstract class
// NOTE: Keep in mind that you'll have to override it from Python
class Kernel {};

// Step 2: Define the GaussianKernel class
class GaussianKernel : public Kernel {
private:
public:
};

// Step 3: Define the ExponentialKernel class
class ExponentialKernel : public Kernel {
private:
public:
};


PYBIND11_MODULE(Kernels, m) {
  // Step 4: Bind the 3 classes defined so far
  
  // Step 5: Bind the make_gaussian and make_exponential functions
  m.def("make_gaussian", /* ... */)
  m.def("make_expoenential", /* ... */)
}

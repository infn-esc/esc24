
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(ArrayOperations, m) {
  // Step 1: Bind a function taking two NumPy arrays of floats and returning their
  // dot product
  // Hint: use an STL algorithm
  m.def("dot", /* ... */);

  // Step 2: Bind a function taking two NumPy arrays of floats and returning their
  // cross product (as another NumPy array)
  m.def("cross", /* ... */);
}

// Bonus: try to implement the functions using both the request and unckecked methods
// and check that the results are the same

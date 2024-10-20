
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

namespace py = pybind11;

template <typename T>
class Matrix {
  private:
	// Step 1: Define the data members
  
  public:
	// Step 2: Define the constructors and the getters
	
	// Step 3: Overload the arithmetic and access operators
};

PYBIND11_MODULE(Matrix, m) {
	// Step 4: Define the class and the buffer protocol

	// Step 5: Define locally and bind the __str__, __len__ operators

	// Step 6: Expose the matrix as a Python buffer
}

cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_C_COMPILER "/shared/software/compilers/gcc-12.3.0_sl7/bin/gcc")
set(CMAKE_CXX_COMPILER "/shared/software/compilers/gcc-12.3.0_sl7/bin/g++")

project(pybind-HandsOn VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(pybind11)

# Exercise 01
# pybind11_add_module(HEP SHARED 01_particle.cpp)

# Exercise 02
# pybind11_add_module(Kernels SHARED 02_kernels.cpp)

# Exercise 03
# pybind11_add_module(Matrix SHARED 03_matrix.cpp)

# Exercise 04
# add_executable(accumulate 04_list_accumulate.cpp)
# target_link_libraries(accumulate PRIVATE pybind11::embed)
# target_link_libraries(accumulate PRIVATE dl)
# target_link_libraries(accumulate PRIVATE pthread)
# target_link_libraries(accumulate PRIVATE util)

# Exercise 05
# pybind11_add_module(ArrayOperations SHARED 05_array_operations.cpp)

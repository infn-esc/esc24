---
title: Binding Python and C++ with pybind11
layout: main
section: pybind
---

## Exercise 4: List accumulation
Define a function that takes a Python list of integers and computes the sum of the elements.
Inside the `main` function create a list, fill it with integers from
0 to 10 and compute the sum using the module defined before.
Hints:
- use c++20 ranges and views to fill the list
- you can cast the element inside the list with `x.cast<T>()`

## Exercise 5: Numpy array operations
Define a module binding two functions:
- one computing the dot product of two NumPy arrays
- one computing the cross product of two NumPy arrays
As a bonus, try to implement the functions both using the `request` and the `unchecked`
methods.

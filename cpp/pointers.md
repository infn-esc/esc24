---
title: Managing resources
layout: main
section: cpp
---

The goal of this exercise is to get some confidence with the smart pointers
provided by the C++ standard library:
[`unique_ptr`](https://en.cppreference.com/w/cpp/memory/unique_ptr) and
[`shared_ptr`](https://en.cppreference.com/w/cpp/memory/shared_ptr).

## Exercise 1

In the C language, memory is allocated with `malloc` and released with `free`.
Both functions are available also in C++, possibly in namespace `std`. Consider
[`c_alloc.cpp`]({% include hands_on_base.md %}/{{ page.section }}/c_alloc.cpp),
where a bunch of memory is allocated and released with those functions. Adapt
the code to manage the allocation through a smart pointer with an appropriate
deleter.

Note also the use of
[`std::span`](https://en.cppreference.com/w/cpp/container/span), available since
C++20, to represent a contiguous sequence of objects.

## Exercise 2

Open the test program [`dir.cpp`]({% include hands_on_base.md %}/{{ page.section
}}/dir.cpp) and, following the hints in the file, complete the existing code to:

* create a smart pointer managing a DIR resource obtained with the
  [`opendir`](https://man7.org/linux/man-pages/man3/opendir.3.html) function call
* associate a deleter to that smart pointer
* implement a function to read the names of the files in that
  directory, using [`readdir_r`](https://man7.org/linux/man-pages/man3/readdir_r.3.html)
* check if the deleter is called at the right moment
* hide the creation of the smart pointer behind a factory function
* populate a vector of `FILE`s, each properly wrapped in a smart pointer,
  obtained [`fopen`](http://man7.org/linux/man-pages/man3/fopen.3.html)ing the
  regular files in that directory

You can experiment with both kinds of smart pointers and with the type
of the function arguments.

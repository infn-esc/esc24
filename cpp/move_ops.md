---
title: Move operations
layout: main
section: cpp
---

The goal of this exercise is to get some confidence with move
semantics, in particular to implement the move operations of a class:
the move constructor and the move assignment.

Open the test program [`string.cpp`]({% include hands_on_base.md %}/{{ page.section }}/string.cpp) and complete the existing code to:

* Complete the set of the special member functions so that String is copyable and movable
  
* Add `operator[]` (const and non-const) to access a character at a given position

* Add a `c_str()` member function to access the underlying C-style string

* Use a smart pointer instead of a raw pointer. Note that `std::unique/shared_ptr` and corresponding
  `make_unique/shared` support arrays

The commands to build and run the code are:

```shell
$ cd {{ site.github.repository_name }}/hands-on/{{ page.section }}
$ g++ -Wall -Wextra -o string string.cpp
$ ./string
...
```

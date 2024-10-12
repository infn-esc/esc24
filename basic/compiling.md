---
title: How to compile
layout: main
section: basic
---

The goal of this exercise is to understand how to compile and run test programs
on `esc`. Let's do it with `g++ v. 12.3`, which by default compiles in C++17
mode. It's also good practice to enable a high level of warnings.

Build [pi_time.cpp]({{ site.github.repository_url }}/blob/main/hands-on/basic/pi_time.cpp):

```shell
[studentN@hpc-201-11-40 ~]$ cd {{ site.github.repository_name }}/hands-on/{{ page.section }}
[studentN@hpc-201-11-40 basic]$ g++ -Wall -Wextra -o pi_time pi_time.cpp
[studentN@hpc-201-11-40 basic]$ ./pi_time 1000
pi = 3.14159 for 1000 iterations in 4.6336e-05 s
```

Run the code multiple times with different arguments.

Compile the code with different optimization levels (`-O0` to `-O3` and
`-Ofast`) and see if the execution time changes.

Look at the code and see how `std::chrono` is used to do some basic time
measurements, thanks to its
[`steady_clock`](https://en.cppreference.com/w/cpp/chrono/steady_clock).

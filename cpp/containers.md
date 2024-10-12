---
title: Containers
layout: main
section: cpp
---

The goal of this exercise is to appreciate the importance of
manipulating data structures that are contiguous in memory.

Take [`containers.cpp`]({% include hands_on_base.md %}/{{ page.section
}}/containers.cpp). It implements for an `std::vector<int>` two operations:

* `fill()`, which fills the container with `N` integers randomly
generated. Each integer is inserted in a random position.

* `process()`, which applies a reduction operation to the
previously-filled container.

Inspect, build and run `containers.cpp`, also through `perf`, varying
the number of elements:

```shell
$ cd {{ site.github.repository_name }}/hands-on/{{ page.section }}
$ g++ -Wall -Wextra -O3 -o containers containers.cpp
$ ./containers 100000
vector fill: 0.526868 s
vector process: 2.9868e-05 s
$ perf stat -d ./containers
```

Then extend the program to implement `fill()` and `process()` for an
`std::list<int>`. Compare the performance obtained with the two
containers.

---
title: Use of algorithms and function objects
layout: main
section: cpp
---

The goal of these exercises is to get some confidence with the C++ entities that
are _callable_: functions, function objects and lambdas. You will use them to
customize the behaviour of some of the algorithms provided by the C++ standard
library.

## Exercise 1

The program [`algo.cpp`]({% include hands_on_base.md %}/{{ page.section }}/algo.cpp)
generates and prints a vector of randomly-chosen integer numbers.

Build it and run it a few times to verify that it works as expected:

```shell
$ cd {{ site.github.repository_name }}/hands-on/{{ page.section }}
$ g++ -O3 -Wall -Wextra -o algo algo.cpp
$ ./algo
...
```

Then extend the program in order to:

* sum all the elements of the vector
* compute the average of the first half and of the second half of the vector
* move the three central numbers to the beginning
* remove duplicate elements

The source code contains some hints about what algorithms can be used to
implement the requested functionality.

[cppreference.com](http://en.cppreference.com/w/cpp/algorithm) is an excellent
source of documentation for the algorithms (and C++ in general).

## Exercise 2

Consider the program [`algo_par.cpp`]({% include hands_on_base.md %}/{{
page.section }}/algo_par.cpp) and extend it in order to:

* sum the elements of the vector with the `std::reduce` algorithm, using the
  sequential execution policy
* sum the elements of the vector with the `std::reduce` algorithm, using the
  parallel execution policy
* sort the elements of the vector with the `std::sort` algorithm, using the
  sequential execution policy
* sort the elements of the vector with the `std::sort` algorithm, using the
  parallel execution policy

Compare the execution times.

The enable the use of the parallel algorithms you need to link against oneTBB,
so remember to [enable it]({{ site.baseurl }}/basic/environment.html).

```shell
g++ -Wall -Wextra -O3 algo_par.cpp -ltbb
```

Parallel programming and oneTBB will be better introduced in subsequent lectures
on parallel programming.

## Exercise 3

`std::default_random_engine` usually is an alias for a [_linear congruential_
generator](https://en.wikipedia.org/wiki/Linear_congruential_generator). Let's
consider _minstd_rand0_, which produces a sequence according to

$$
x_{n+1} = 16807 x_n \mod (2^{31} - 1)
$$

Write a class `LinearCongruential` whose constructor initializes the sequence
with a seed (with a default value of 1) and an `operator()` that updates the
internal value (the $x_n$) and returns it. The type of the numbers involved in
the computations is `unsigned long int`.

Print a few numbers and check that they correspond to what is produced by
`std::default_random_engine`.

## Exercise 4

Modify the program [`algo_functions.cpp`]({% include hands_on_base.md %}/{{
page.section }}/algo_functions.cpp) to:

* multiply all the elements of the vector
* compute the mean and the standard deviation
* sort the vector in descending order
* move the even numbers to the beginning
* create another vector with the squares of the numbers in the first vector
* find the first multiple of 3 or 7
* erase from the vector all the multiples of 3 or 7

For the implementation follow the hints in the source code.

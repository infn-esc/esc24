/*
Copyright (C) 2024 Andrea Bocci
SPDX-License-Identifier: GNU General Public License v3.0 or later

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

void measure(bool verbose, std::vector<std::uint64_t> v) {
  const auto start = std::chrono::steady_clock::now();
  std::sort(v.begin(), v.end());
  const auto finish = std::chrono::steady_clock::now();
  if (verbose) {
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << "ms\n";
  }
  assert(std::ranges::is_sorted(v));
};

void repeat(std::vector<std::uint64_t> const& v, size_t times, size_t skip = 0) {
  for (size_t i = 0; i < skip; ++i) {
    measure(false, v);
  }
  for (size_t i = 0; i < times; ++i) {
    measure(true, v);
  }
}

int main() {
  const std::size_t size = 1'000'000;
  const std::size_t skip = 1;
  const std::size_t repeats = 10;

  std::vector<std::uint64_t> v(size);
  std::mt19937 gen{std::random_device{}()};
  std::ranges::generate(v, gen);

  std::cout << "sequential sort\n";
  repeat(v, repeats, skip);
  std::cout << '\n';

  // TODO
  //   - change the sequential sort to use std::execution::seq
  //   - try the other execution policies (par, unseq, par_unseq)
}

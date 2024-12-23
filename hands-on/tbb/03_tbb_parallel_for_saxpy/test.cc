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

#include <chrono>
#include <cstdint>
//#include <format>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

template <typename T>
void axpy(T a, T x, T y, T& z) {
  z = a * x + y;
}

template <typename T>
void sequential_axpy(T a, std::vector<T> const& x, std::vector<T> const& y, std::vector<T>& z) {
  std::size_t size = x.size();
  for (std::size_t i = 0; i < size; ++i) {
    axpy(a, x[i], y[i], z[i]);
  }
}

template <typename T>
void measure_sequential(T a, std::vector<T> const& x, std::vector<T> const& y) {
  std::vector<T> z(x.size(), 0);
  auto start = std::chrono::steady_clock::now();
  sequential_axpy(a, x, y, z);
  auto finish = std::chrono::steady_clock::now();
  float ms = std::chrono::duration_cast<std::chrono::duration<float>>(finish - start).count() * 1000.f;
  //std::cout << std::format("{:6.1f}", ms) << " ms\n";
  std::cout << std::fixed << std::setprecision(1) << std::setw(6) << ms << " ms\n";
}

int main() {
  const std::size_t size = 100'000'000;
  const std::size_t times = 10;

  std::mt19937 gen{std::random_device{}()};
  std::uniform_real_distribution<float> dis{-std::numbers::pi, std::numbers::pi};
  float a = dis(gen);
  std::vector<float> x(size);
  std::ranges::generate(x, [&] { return dis(gen); });
  std::vector<float> y(size);
  std::ranges::generate(y, [&] { return dis(gen); });

  std::cout << "sequential axpy\n";
  for (size_t i = 0; i < times; ++i)
    measure_sequential(a, x, y);
  std::cout << '\n';

  // TODO
  //   - write a parallel version of sequential_axpy using tbb::parallel_for
  //   - measure its performance compared to sequential_axpy

}

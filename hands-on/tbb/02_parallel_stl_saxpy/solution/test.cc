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
#include <chrono>
#include <cstdint>
#include <execution>
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
void parallel_axpy(auto policy, T a, std::vector<T> const& x, std::vector<T> const& y, std::vector<T>& z) {
  std::transform(policy, x.begin(), x.end(), y.begin(), z.begin(), [a](T x, T y) -> T {
    T z;
    axpy(a, x, y, z);
    return z;
  });
}

template <typename T>
void measure(auto policy, T a, std::vector<T> const& x, std::vector<T> const& y) {
  std::vector<T> z(x.size(), 0);
  auto start = std::chrono::steady_clock::now();
  parallel_axpy(policy, a, x, y, z);
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

  std::cout << "std::execution::seq\n";
  for (size_t i = 0; i < times; ++i)
    measure(std::execution::seq, a, x, y);
  std::cout << '\n';

  std::cout << "std::execution::unseq\n"; 
  for (size_t i = 0; i < times; ++i)
      measure(std::execution::unseq, a, x, y);
  std::cout << '\n';

  std::cout << "std::execution::par\n";
  for (size_t i = 0; i < times; ++i)
    measure(std::execution::par, a, x, y);
  std::cout << '\n';

  std::cout << "std::execution::par_unseq\n"; 
  for (size_t i = 0; i < times; ++i)
      measure(std::execution::par_unseq, a, x, y);
  std::cout << '\n';
}

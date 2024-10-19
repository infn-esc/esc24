#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <execution>
#include <iostream>
#include <random>
#include <vector>

bool is_sorted(std::vector<std::uint64_t> const& v) {
  if (v.empty()) {
    return true;
  }

  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i] < v[i - 1])
      return false;
  }

  return true;
}

void measure(auto policy, bool verbose, std::vector<std::uint64_t> v) {
  const auto start = std::chrono::steady_clock::now();
  std::sort(policy, v.begin(), v.end());
  const auto finish = std::chrono::steady_clock::now();
  if (verbose) {
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << "ms\n";
  }
  assert(is_sorted(v));
};

void repeat(auto policy, std::vector<std::uint64_t> const& v, size_t times, size_t skip = 0) {
  for (size_t i = 0; i < skip; ++i) {
    measure(policy, false, v);
  }
  for (size_t i = 0; i < times; ++i) {
    measure(policy, true, v);
  }
}

int main() {
  const std::size_t size = 1'000'000;
  const std::size_t skip = 1;
  const std::size_t repeats = 10;

  std::vector<std::uint64_t> v(size);
  std::mt19937 gen{std::random_device{}()};
  std::ranges::generate(v, gen);

  std::cout << "std::execution::seq\n";
  repeat(std::execution::seq, v, repeats, skip);
  std::cout << '\n';

  std::cout << "std::execution::unseq\n";
  repeat(std::execution::unseq, v, repeats, skip);
  std::cout << '\n';

  std::cout << "std::execution::par\n";
  repeat(std::execution::par, v, repeats, skip);
  std::cout << '\n';

  std::cout << "std::execution::par_unseq\n";
  repeat(std::execution::par_unseq, v, repeats, skip);
  std::cout << '\n';
}

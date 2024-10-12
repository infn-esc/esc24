#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <utility>

using Duration = std::chrono::duration<float>;

std::pair<double, Duration> pi(int n)
{
  assert(n > 0);

  auto const start = std::chrono::steady_clock::now();

  auto const step = 1. / n;
  auto sum = 0.;
  for (int i = 0; i != n; ++i) {
    auto x = (i + 0.5) * step;
    sum += 4. / (1. + x * x);
  }

  auto const end = std::chrono::steady_clock::now();

  return {step * sum, end - start};
}

int main(int argc, char* argv[])
{
  int const n = (argc > 1) ? std::atoi(argv[1]) : 10;

  auto const [value, time] = pi(n);

  std::cout << "pi = " << value << " for " << n << " iterations"
            << " in " << time.count() << " s\n";
}

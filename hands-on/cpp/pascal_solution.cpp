#include <array>
#include <cassert>
#include <iostream>

constexpr int N = 8;

template<int N>
constexpr auto make_pascal()
{
  // from 0 to N included
  std::array<std::array<int, N+1>, N+1> result{};

  // first row
  result[0][0] = 1;

  for (int i = 1; i != N+1; ++i) {
    result[i][0] = 1;
    for (int j = 1; j != i; ++j) {
      result[i][j] = result[i-1][j-1] + result[i-1][j];
    }
    result[i][i] = 1;
  }

  return result;
}

constexpr auto pascal_table = make_pascal<N>();

constexpr auto pascal(int i, int j)
{
  assert(0 <= i && i <= N);
  assert(0 <= j && j <= i);
  return pascal_table[i][j];
}

int main()
{
  return pascal(5, 4);
}

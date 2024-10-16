#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

void init(std::vector<int> &v, int N);

// call it as `./a.out 16`
int main(int argc, char *argv[]) {
  int const N = argc > 1 ? std::atoi(argv[1]) : 8;
  std::vector<int> v(10);
  init(v, N);
}

void init(std::vector<int> &v, int N) {
  std::iota(v.begin(), std::next(v.begin(), N), 0);
}

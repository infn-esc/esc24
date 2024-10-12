#include <cstdlib>
#include <iostream>
#include <span>

void do_something_with(std::span<int> a);

int main()
{
  // allocate memory for 1000 int's
  int const SIZE = 1000;
  auto p = static_cast<int*>(std::malloc(SIZE * sizeof(int)));
  do_something_with({p, SIZE});
  std::free(p);
}

void do_something_with(std::span<int> a)
{
  std::fill(a.begin(), a.end(), 42);
}
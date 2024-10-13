#include <iostream>
#include <random>

class LinearCongruential
{
  unsigned long x_;

 public:
  LinearCongruential(unsigned long s = 1) : x_(s)
  {
  }
  auto operator()()
  {
    return x_ = (16807UL * x_) % ((1UL << 31) - 1);
  }
};

int main()
{
  LinearCongruential eng;
  std::default_random_engine stdeng;
  std::cout << eng() << '\t' << stdeng() << '\n';
  std::cout << eng() << '\t' << stdeng() << '\n';
  std::cout << eng() << '\t' << stdeng() << '\n';
  std::cout << eng() << '\t' << stdeng() << '\n';
}
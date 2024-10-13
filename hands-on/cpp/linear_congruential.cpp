#include <iostream>
#include <random>

class LinearCongruential
{
  // ...
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
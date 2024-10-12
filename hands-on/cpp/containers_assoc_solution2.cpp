#include <vector>
#include <list>
#include <set>
#include <unordered_set>
#include <chrono>
#include <iostream>
#include <random>
#include <cassert>
#include <cstdlib>

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<float>;

std::default_random_engine eng{std::random_device{}()};
using Distribution = std::uniform_int_distribution<>;
Distribution dist;

template<typename C, typename T = void>
struct is_associative : std::false_type {};

// associative containers have the `key_type` nested type
template<typename C>
struct is_associative<C, std::void_t<typename C::key_type>> : std::true_type {};

template<typename C>
constexpr auto is_associative_v = is_associative<C>::value;

template<typename Container>
auto fill_impl(Container& cont, int N, std::false_type /* sequential */)
{
  for (int i = 0; i != N; ++i) {
    // generate a number between 0 and the current size of the container
    auto n = dist(eng, Distribution::param_type{0, static_cast<int>(cont.size())});
    // advance n positions in the container
    auto it = cont.begin();
    std::advance(it, n);
    // insert the number itself in that position
    cont.insert(it, n);
  }
}

template<typename Container>
auto fill_impl(Container& cont, int N, std::true_type /* associative */)
{
  for (int i = 0; i != N; ++i) {
    cont.insert(i);
  }
}

template<typename Container>
Duration fill(Container& cont, int N)
{
  assert(N >= 0);

  auto start = Clock::now();

  cont.clear();

  fill_impl(cont, N, is_associative<Container>{});

  assert(static_cast<int>(cont.size()) == N);

  return Clock::now() - start;
}

template<typename Container>
Duration process(Container const& cont)
{
  auto start = Clock::now();

  // the volatile is to avoid complete removal by the optimizer
  auto volatile v = std::accumulate(std::begin(cont), std::end(cont), 0, [](int a, int n) {
      return a ^ n;
    });
  (void)v; // to silence a warning about unused variable

  return Clock::now() - start;
}

int main(int argc, char* argv[])
{
  int const N = (argc > 1) ? std::atoi(argv[1]) : 10000;

  std::vector<int> v;
  std::cout << "vector fill: " << fill(v, N).count() << " s\n";
  std::cout << "vector process: " << process(v).count() << " s\n";
  std::list<int> l;
  std::cout << "list fill: " << fill(l, N).count() << " s\n";
  std::cout << "list process: " << process(l).count() << " s\n";
  std::set<int> s;
  std::cout << "set fill: " << fill(s, N).count() << " s\n";
  std::cout << "set process: " << process(s).count() << " s\n";
  std::unordered_set<int> u;
  std::cout << "unordered set fill: " << fill(u, N).count() << " s\n";
  std::cout << "unordered set process: " << process(u).count() << " s\n";
}

#include <iostream>
using SomeType = int;

SomeType *factory();
void do_something(SomeType *);

int main() {
  auto t = factory();
  do_something(t);
}

SomeType *factory() { return new SomeType{}; }

void do_something(SomeType *) { throw 1; }

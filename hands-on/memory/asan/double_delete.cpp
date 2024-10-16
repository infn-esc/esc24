int *factory();

int main() {
  auto t = factory();

  delete t;
  delete t;
}

int *factory() { return new int; }

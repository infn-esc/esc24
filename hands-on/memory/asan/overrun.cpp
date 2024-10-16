#include <cstdlib>

int main() {
  char *c = static_cast<char *>(std::malloc(10 * sizeof(char)));
  c[10] = 'c';
  return 0;
}

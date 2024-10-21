#ifndef ParticleSoA_h
#define ParticleSoA_h
#include <vector>

struct ParticleSoA {
  ParticleSoA() = default;

  double *x;
  double *y;
  double *z;
  int *id;
};
#endif

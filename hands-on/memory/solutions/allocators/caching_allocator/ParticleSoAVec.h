#ifndef ParticleSoAVec_h
#define ParticleSoAVec_h

#include <iostream>
#include <vector>

struct ParticleSoAVec {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<int> id;
};

#endif

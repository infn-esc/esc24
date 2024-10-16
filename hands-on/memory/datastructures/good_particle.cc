#include "Particle.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

// utility class for timing
class Timer {
public:
  Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}

  void reset() { start_time_ = std::chrono::high_resolution_clock::now(); }

  double elapsed() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time =
        end_time - start_time_;
    return elapsed_time.count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_time_;
};

// function to initialize AoS
void initializeAoS(/*args*/) {
  // your code here
}

// function for computation on AoS
void calculateAoS(/*args*/) {
  // your code here
}

// function to initialize SoA
void initializeSoA(/*args*/) {
  // your code here
}

// function for computation on SoA
void calculateSoA(/*args*/) {
  // your code here
}

int main() {

  std::cout << "sizeof(GoodParticle) " << sizeof(GoodParticle) << std::endl;

  //  //Uncomment this code for Exercise 1
  //  std::random_device rd;
  //  std::mt19937 gen(2);
  //
  //  std::vector<double> pDist;
  //  std::vector<double> xDist;
  //  std::vector<double> zDist;
  //  std::vector<double> yDist;
  //  std::vector<double> massDist;
  //
  //  constexpr int NIter = 1000000;
  //  constexpr double x_max = 200;
  //  constexpr int Npart = 1 << 10;
  //
  //  auto fillVec = [&](std::vector<double>& vec, double min, double max) {
  //      vec.reserve(Npart);
  //      std::uniform_real_distribution<double> dist(min, max);
  //      std::generate_n(std::back_inserter(vec), Npart, [&]{ return dist(gen);
  //      });
  //  };
  //
  //
  //  fillVec(pDist, 10., 100.);
  //  fillVec(xDist, -100., 100.);
  //  fillVec(yDist, -100., 100.);
  //  fillVec(zDist, -300., 300.);
  //  fillVec(massDist, 10., 100.);
  //
  //  auto const t = std::uniform_real_distribution<double>(-2., 2.)(gen);
  //
  //  Timer timer; //starting the clock
  //  std::vector<GoodParticle> good_particles;
  //  initializeAoS(/*args*/);
  //
  //  calculateAoS(/*args*/);;
  //  std::cout << "AoS Elapsed time: " << timer.elapsed() << " ms " << "\n";
  //
  //  timer.reset();
  //  ParticleSoA soa_particles(Npart);
  //  initializeSoA(/*args*/);
  //
  //  calculateSoA(/*args*/);
  //  std::cout << "SoA Elapsed time: " << timer.elapsed() << " ms " << "\n";

  return 0;
}

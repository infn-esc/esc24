#include "Particle.h"
#include <iostream>
#include <chrono>
#include <random>

//utility class for timing
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

//function to initialize AoS
void initializeAoS(std::vector<GoodParticle>& particles, const std::vector<double>& pxDist, const std::vector<double>& xDist, const std::vector<double>& yDist, const std::vector<double>& zDist, const std::vector<double>& massDist, int Npart) {
  particles.resize(Npart);
  for (int i = 0; i < Npart; ++i) {
    GoodParticle p;
    p.id_ = i;
    p.px_ = pxDist[i];
    p.py_ = pxDist[i];
    p.pz_ = pxDist[i];
    p.x_ = xDist[i];
    p.y_ = yDist[i];
    p.z_ = zDist[i];
    p.mass_ = massDist[i];
    p.energy_ = 0.;
    p.name_ = "Particle";
    particles[i] = p;
  }
}
//function for computation on AoS
void calculateAoS(std::vector<GoodParticle>& particles, const double t, double x_max, int NIter) {
  for (int iter = 0; iter < NIter; ++iter) {
    for (int i = 0; i < particles.size(); ++i) {
      auto p = particles[i];
      p.x_ += p.px_ / p.mass_ * t;
      if (p.x_ < 0. || p.x_ > x_max) {
        p.hit_x_ = true;
        p.px_ *= -1.;
      } else {
        p.hit_x_ = false;
      }
    }
  }
}

//function to initialize SoA
void initializeSoA(ParticleSoA& particles, const std::vector<double>& pxDist, const std::vector<double>& xDist, const std::vector<double>& yDist, const std::vector<double>& zDist, const std::vector<double>& massDist, int Npart) {
  particles.id_.resize(Npart);
  particles.px_.resize(Npart);
  particles.py_.resize(Npart);
  particles.pz_.resize(Npart);
  particles.x_.resize(Npart);
  particles.y_.resize(Npart);
  particles.z_.resize(Npart);
  particles.mass_.resize(Npart);
  particles.name_.resize(Npart);
  particles.energy_.resize(Npart);
  particles.hit_x_.resize(Npart);

  for (int i = 0; i < Npart; ++i) {
    particles.id_[i] = i;
    particles.px_[i] = pxDist[i];
    particles.py_[i] = pxDist[i];
    particles.pz_[i] = pxDist[i];
    particles.x_[i] = xDist[i];
    particles.y_[i] = yDist[i];
    particles.z_[i] = zDist[i];
    particles.mass_[i] = massDist[i];
    particles.name_[i] = std::string("Particle");
    particles.energy_[i] = 0.;
  }
}

//function for computation on SoA
void calculateSoA(std::vector<double>& x_, std::vector<double>& px_, const std::vector<float>& mass_, std::vector<bool>& hit_x_, const double t, const double x_max, const int NPart, const int NIter){
  for (int iter = 0; iter < NIter; ++iter) {
    for (int i = 0; i < NPart; ++i) {
      x_[i] += px_[i] / mass_[i] * t;
      if (x_[i] < 0. || x_[i] > x_max) {
        hit_x_[i] = true;
        px_[i] *= -1;
      } else {
        hit_x_[i] = false;
      }
    }
  }
}

//function for computation on SoA by passing the full SoA instead of only the member you need for the computation
void calculateSoA(ParticleSoA& particles, const double t, double x_max, int NPart, int NIter) {
  for (int iter = 0; iter < NIter; ++iter) {
    auto& x_ = particles.x_;
    auto& px_ = particles.px_;
    auto& mass_ = particles.mass_;
    auto& hit_x_ = particles.hit_x_;

    for (int i = 0; i < NPart; ++i) {
      x_[i] += px_[i] / mass_[i] * t;
      if (x_[i] < 0. || x_[i] > x_max) {
        hit_x_[i] = true;
        px_[i] *= -1;
      } else {
        hit_x_[i] = false;
      }
    }
  }
}


/*
 * Try to compile with the different optimization. Let's see if the SoA layout helps also the compiler in optimizing the code
 */

int main() {
  std::random_device rd;
  std::mt19937 gen(2);

  std::vector<double> pxDist;
  std::vector<double> xDist;
  std::vector<double> zDist;
  std::vector<double> yDist;
  std::vector<double> massDist;

  double x_max = 200;
  int Npart = 1 << 10;

  auto fillVec = [Npart, &gen](std::vector<double>& vec, double min, double max) -> void {
    for (int i = 0; i < Npart; ++i) {
      std::uniform_real_distribution<double> dist(min, max);
      vec.push_back(dist(gen));
    }
  };

  fillVec(pxDist, 10., 100.);
  fillVec(xDist, -100., 100.);
  fillVec(yDist, -100., 100.);
  fillVec(zDist, -300., 300.);
  fillVec(massDist, 10., 100.);
  auto const t = std::uniform_real_distribution<double>(-2., 2.)(gen);
  int NIter = 1000000;

  Timer timer;
  std::vector<GoodParticle> good_particles;
  initializeAoS(good_particles, pxDist, xDist, yDist, zDist, massDist, Npart);

  calculateAoS(good_particles, t , x_max, NIter);
  std::cout << "AoS Elapsed time: " << timer.elapsed() << " ms " << "\n";

  timer.reset();
  ParticleSoA soa_particles(Npart);
  initializeSoA(soa_particles, pxDist, xDist, yDist, zDist, massDist, Npart);

  calculateSoA(soa_particles.x_, soa_particles.px_, soa_particles.mass_, soa_particles.hit_x_, t, x_max, Npart,  NIter);
  std::cout << "SoA Elapsed time: " << timer.elapsed() << " ms " << "\n";

  return 0;
}


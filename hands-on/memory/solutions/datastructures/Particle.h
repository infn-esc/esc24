#ifndef GOOD_PARTICLE
#define GOOD_PARTICLE
#include <vector>
#include <string> 

// I follow the general rule: put members with higher alignment (alignof()) first.
// the name of the particle is not really useful for computations, I expect to not use it very much
// --> I put the name of the particle at the end, keeping the other members closer to each other
// This struct has a sizeof(96) > 64 bytes (cache line size), I'd like to have all the useful members in the same cache line
struct GoodParticle {
  double x_, y_, z_;
  double px_, py_, pz_;
  bool hit_x_, hit_y_, hit_z_;
  float mass_;
  float energy_;
  int id_;
  std::string name_; 
};


//create the SoA version of Particle. To avoid further allocations, we resize all the "columns" first (in the constructor)
struct  ParticleSoA{
    ParticleSoA(int N) {
      x_.resize(N);
      y_.resize(N);
      z_.resize(N);
      px_.resize(N);
      py_.resize(N);
      pz_.resize(N);
      hit_x_.resize(N);
      hit_y_.resize(N);
      hit_z_.resize(N);
      mass_.resize(N);
      energy_.resize(N);
      id_.resize(N);
      name_.resize(N);
  
    }  

//SoA columns
    std::vector<double> x_, y_, z_;
    std::vector<double> px_,py_, pz_;
    std::vector<bool> hit_x_, hit_y_, hit_z_; 
    std::vector<float> mass_; 
    std::vector<float> energy_; 
    std::vector<int> id_; 
    std::vector<std::string> name_; 
};
#endif

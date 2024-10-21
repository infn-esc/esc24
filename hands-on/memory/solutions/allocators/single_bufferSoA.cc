#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <vector>

// get cache line size (or in any case the value we need to have destructive
// interference in false sharing)
// constexpr size_t CACHE_ALIGNMENT =
// std::hardware_destructive_interference_size;
constexpr std::size_t CACHE_ALIGNMENT = std::hardware_destructive_interference_size;

// I want to use a smart pointer together with the operatore "aligned" new --> I
// have to provide a custom deleter to the unique pointer, so let's create it
struct AlignedDeleter {
  void operator()(std::byte *ptr) {
    std::cout << "Correctly Deleting ptr in unique ptr\n";
    ::operator delete[](ptr, std::align_val_t{CACHE_ALIGNMENT});
  }
};

// create a nickname to my new unique_ptr
using AlignedUniquePtr = std::unique_ptr<std::byte[], AlignedDeleter>;

class alignas(CACHE_ALIGNMENT) ParticleSoA {
public:
  explicit ParticleSoA(size_t numParticles) : size(numParticles) {
    // c alculate the total buffer size needed for all particle information
    bufferSize =
        numParticles * (7 * sizeof(double) + 4 * sizeof(bool) +
                        1 * sizeof(float) + sizeof(int) + 256 * sizeof(char));

    // allocate a single buffer of memory (array of std::byte for example)
    // aligned to 64 let's use "aligned" new and we can try to wrap it in a
    // unique pointer to have automatic free of the memory
    buffer =
        AlignedUniquePtr(static_cast<std::byte *>(::operator new[](
                             bufferSize, std::align_val_t{CACHE_ALIGNMENT})),
                         AlignedDeleter());

    // initialize the buffer to 0 for clarity
    std::memset(buffer.get(), 0, bufferSize);

    // get the initial position of the pointer
    // and assign it to a "new" pointer the will keep moving to assign the
    // position of all the SoA columns
    std::byte *currentPtr = buffer.get();

    // set each SoA column
    x = reinterpret_cast<double *>(currentPtr);
    currentPtr += numParticles * sizeof(double);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    y = reinterpret_cast<double *>(currentPtr);
    currentPtr += numParticles * sizeof(double);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    z = reinterpret_cast<double *>(currentPtr);
    currentPtr += numParticles * sizeof(double);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    px = reinterpret_cast<double *>(currentPtr);
    currentPtr += numParticles * sizeof(double);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    py = reinterpret_cast<double *>(currentPtr);
    currentPtr += numParticles * sizeof(double);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    pz = reinterpret_cast<double *>(currentPtr);
    currentPtr += numParticles * sizeof(double);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    hit_x = reinterpret_cast<bool *>(currentPtr);
    currentPtr += numParticles * sizeof(bool);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    hit_y = reinterpret_cast<bool *>(currentPtr);
    currentPtr += numParticles * sizeof(bool);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    hit_z = reinterpret_cast<bool *>(currentPtr);
    currentPtr += numParticles * sizeof(bool);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    hit_E = reinterpret_cast<bool *>(currentPtr);
    currentPtr += numParticles * sizeof(bool);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    mass = reinterpret_cast<double *>(currentPtr);
    currentPtr += numParticles * sizeof(double);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    energy = reinterpret_cast<float *>(currentPtr);
    currentPtr += numParticles * sizeof(float);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    id = reinterpret_cast<int *>(currentPtr);
    currentPtr += numParticles * sizeof(int);
    currentPtr = alignPointer(currentPtr, CACHE_ALIGNMENT);

    // note: for the std::string you need to create the object!
    // we can use new(ptr) to allocate the object in the correct memory address
    name = reinterpret_cast<std::string *>(currentPtr);
    for (size_t i = 0; i < numParticles; ++i) {
      new (&name[i]) std::string(); // Placement new to construct std::string
    }

    // Print the addresses of each member
    std::cout << " x " << x << "\n";
    std::cout << " y " << y << "\n";
    std::cout << " z " << z << "\n";
    std::cout << " px " << px << "\n";
    std::cout << " py " << py << "\n";
    std::cout << " pz " << pz << "\n";
    std::cout << " hit_x " << hit_x << "\n";
    std::cout << " hit_y " << hit_y << "\n";
    std::cout << " hit_z " << hit_z << "\n";
    std::cout << " hit_E " << hit_E << "\n";
    std::cout << " mass " << mass << "\n";
    std::cout << " energy " << energy << "\n";
    std::cout << " id " << id << "\n";
    std::cout << " name " << name << "\n";
  }

  // Destructor to manually free the memory and destroy std::string objects
  ~ParticleSoA() {
    std::cout << "Calling ParticleSoA destructor and destroying std::string "
              << std::endl;
    for (size_t i = 0; i < size; ++i) {
      name[i].~basic_string(); // Manually destroy each std::string
    }
  }

  // Pointers to various arrays
  double *x;
  double *y;
  double *z;
  double *px;
  double *py;
  double *pz;
  bool *hit_x;
  bool *hit_y;
  bool *hit_z;
  bool *hit_E;
  double *mass;
  float *energy;
  int *id;
  std::string *name;

private:
  size_t size;
  size_t bufferSize;
  AlignedUniquePtr buffer;

  // utility to align pointer with correct padding
  inline std::byte *alignPointer(std::byte *ptr, size_t alignment) {
    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    auto alignedAddr = (addr + alignment - 1) / alignment *
                       alignment; // find the closest aligned address
    return reinterpret_cast<std::byte *>(alignedAddr);
  }
};

void initializeSoA(ParticleSoA &particles, const std::vector<double> &pxDist,
                   const std::vector<double> &xDist,
                   const std::vector<double> &yDist,
                   const std::vector<double> &zDist,
                   const std::vector<double> &massDist, int Npart) {
  for (int i = 0; i < Npart; ++i) {
    particles.id[i] = i;
    particles.px[i] = pxDist[i];
    particles.py[i] = pxDist[i];
    particles.pz[i] = pxDist[i];
    particles.x[i] = xDist[i];
    particles.y[i] = yDist[i];
    particles.z[i] = zDist[i];
    particles.mass[i] = massDist[i];
    particles.name[i] = "Particle" + std::to_string(i);
    particles.energy[i] = 0.;
  }
}

int main() {
  int N = 169;
  ParticleSoA p(N);
  std::random_device rd;
  std::mt19937 gen(1);

  // Create random vectors
  std::vector<double> pxDist;
  std::vector<double> xDist;
  std::vector<double> yDist;
  std::vector<double> zDist;
  std::vector<double> massDist;

  auto fillVec = [N, &gen](std::vector<double> &vec, double min,
                           double max) -> void {
    for (int i = 0; i < N; ++i) {
      std::uniform_real_distribution<double> dist(min, max);
      vec.push_back(dist(gen));
    }
  };

  fillVec(pxDist, 10., 100.);
  fillVec(xDist, -100., 100.);
  fillVec(yDist, -100., 100.);
  fillVec(zDist, -300., 300.);
  fillVec(massDist, 10., 100.);

  // Fill the SoA
  initializeSoA(p, pxDist, xDist, yDist, zDist, massDist, N);
  for (int i = 0; i < N; ++i) {
    std::cout << " Name  " << p.name[i] << " id " << p.id[i] << " " << p.x[i]
              << " " << p.y[i] << " " << p.z[i] << " " << p.px[i] << " "
              << p.py[i] << " " << p.pz[i] << " " << p.mass[i] << " "
              << p.energy[i] << " " << p.hit_x[i] << " " << p.hit_y[i] << " "
              << p.hit_z[i] << "\n";
  }
}

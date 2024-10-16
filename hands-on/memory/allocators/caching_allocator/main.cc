#include "CachingAllocator.h"
#include "ParticleSoA.h"
#include "ParticleSoAVec.h"
#include <chrono>
#include <random>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

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

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());

  // Define the range for random numbers
  std::uniform_int_distribution<int> distribution(5000, 10000);

  const int NIter = 10;
  CachingAllocator allocator(true);
  Timer timer;

  for (int it = 0; it < NIter; ++it) {
    const int N = distribution(gen);
    ParticleSoA particles;
    particles.x = static_cast<double *>(allocator.allocate(sizeof(double) * N));
    particles.y = static_cast<double *>(allocator.allocate(sizeof(double) * N));
    particles.z = static_cast<double *>(allocator.allocate(sizeof(double) * N));
    // Fill vectors x and y
    for (int i = 0; i < N; ++i) {
      particles.x[i] = static_cast<double>(i);
      particles.y[i] = static_cast<double>(i * 2);
    }

    // Add vectors x and y to z
    for (int i = 0; i < N; ++i) {
      particles.z[i] = particles.x[i] + particles.y[i];
    }

    // Check the result for correctness
    for (int i = 0; i < N; ++i) {
      if (particles.z[i] != particles.x[i] + particles.y[i]) {
        std::cerr << "Result is incorrect!" << std::endl;
        return 1; // Return an error code
      }
    }

    allocator.deallocate(particles.z);
    allocator.deallocate(particles.y);
    allocator.deallocate(particles.x);
  }

  allocator.free();

  std::cout << "Elapsed time Caching Allocator: " << timer.elapsed() << " ms "
            << std::endl;

  timer.reset();

  for (int it = 0; it < NIter; ++it) {
    int NPart = distribution(gen);
    const int N = NPart;
    ParticleSoAVec particlesVec;
    particlesVec.z.resize(N);
    for (int i = 0; i < N; ++i) {
      particlesVec.x.push_back(static_cast<double>(i));
      particlesVec.y.push_back(static_cast<double>(i));
    }

    // Add vectors x and y to z
    for (int i = 0; i < N; ++i) {
      particlesVec.z[i] = particlesVec.x[i] + particlesVec.y[i];
    }
    // Check the result for correctness
    for (int i = 0; i < N; ++i) {
      if (particlesVec.z[i] != particlesVec.x[i] + particlesVec.y[i]) {
        std::cerr << "Result is incorrect!" << std::endl;
        return 1; // Return an error code
      }
    }
  }

  std::cout << "Elapsed time Without Caching Allocator: " << timer.elapsed()
            << " ms " << std::endl;

  return 0;
}

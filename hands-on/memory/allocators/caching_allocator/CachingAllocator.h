#include <algorithm>
#include <cmath>
#include <iostream>
#include <new>
#include <vector>

// some constants
#define MAX_BINS 7
#define MIN_BINS 3
#define GROWTH 8

class CachingAllocator {
public:
  CachingAllocator(bool debug = false) : debug_(debug) {
    initialiseBins();
    precalculateBinSizes();
  };

  void *allocate(std::size_t size) {
    if (debug_)
      printf("Asking for %lld Bytes\n", static_cast<long long>(size));
  }

  void deallocate(void *ptr) { return; }
  // Function to release all allocated memory
  void free() {}

private:
  struct MemoryBlock { // this class is used to represent blocks we are going to
                       // allocate
    MemoryBlock() = delete; // I remove the default constructor because I want
                            // to create a block only when I know the size
    MemoryBlock(std::size_t allocSize) {
      ptr = (char *)std::aligned_alloc(std::size_t(64), allocSize);
      size = allocSize;
    }

    char *ptr;
    std::size_t size;
  };
  bool debug_;

  std::vector<std::vector<MemoryBlock>> live_blocks;   // currently in used
  std::vector<std::vector<MemoryBlock>> cached_blocks; // currently cached
  std::vector<std::size_t> bin_sizes;

  void initialiseBins() {
    live_blocks.resize(MAX_BINS);
    cached_blocks.resize(MAX_BINS);
  }

  void precalculateBinSizes() {
    bin_sizes.reserve(MAX_BINS);
    for (std::size_t i = 0; i <= MAX_BINS; ++i) {
      bin_sizes.push_back(static_cast<std::size_t>(std::pow(GROWTH, i)));
    }
  }
};

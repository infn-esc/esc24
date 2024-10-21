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

  // method allocate
  void *allocate(std::size_t size) {
    // note std::move is used to avoid copying
    if (debug_)
      printf("Asking for %lld Bytes\n", static_cast<long long>(size));

    // use method to find the bin index for the requested allocation
    std::size_t binIndex = findBin(size);
    if (binIndex <= MAX_BINS) { // if the binIndex is below the MAX_BINS then we
                                // can proceed with the caching logic
      if (!cached_blocks[binIndex].empty()) { // let's check if we have
                                              // something in our cached storage
        const MemoryBlock cachedMemoryBlock =
            std::move(cached_blocks[binIndex]
                          .back()); // take the memory block (the last one
                                    // allocated/returned to that bin)
        cached_blocks[binIndex].pop_back(); // remember to remove it
        void *allocatedMemory =
            cachedMemoryBlock.ptr; // take the ptr of the allocated block
        live_blocks[binIndex].push_back(std::move(
            cachedMemoryBlock)); // put the block in the live block container
        if (debug_)
          printf("Reusing block %p of size %lld Bytes\n", cachedMemoryBlock.ptr,
                 static_cast<long long>(size));
        return allocatedMemory; // return the allocated block to the user!;
      } else {                  // otherwise, let's create a new block!
        live_blocks[binIndex].emplace_back(bin_sizes[binIndex]);
        if (debug_)
          printf(
              "No cached block found, creating new one %p of size %lld Bytes, "
              "binIndex %d\n",
              live_blocks[binIndex].back().ptr,
              static_cast<long long>(live_blocks[binIndex].back().size),
              static_cast<int>(binIndex));
        return live_blocks[binIndex]
            .back()
            .ptr; // and now return the new block to the user!
      }
    } else {
      // if requested allocxation is to big --> return bad_alloc
      throw std::bad_alloc();
    }
  }

  // deal with deallocation
  void deallocate(void *ptr) {
    // need to find the block in the live blocks
    for (size_t i = 0; i < live_blocks.size(); ++i) {
      auto &bin = live_blocks[i];
      auto it =
          std::find_if(bin.begin(), bin.end(), [ptr](const MemoryBlock &block) {
            return block.ptr == ptr;
          });
      if (it != bin.end()) {
        if (debug_)
          printf("Deallocating block %p of size %lld to bin %lld\n", ptr,
                 static_cast<long long>(it->size), static_cast<long long>(i));
        // now that I found the block in the live block let's transfer it to the
        // cached blocks
        const MemoryBlock block = std::move(*it);
        bin.erase(it); // remove it from live blocks
        std::size_t binIndex = findBin(block.size);
        cached_blocks[binIndex].push_back(std::move(block));
        return;
      }
    }
    if (debug_)
      printf("Unable to deallocate block %p\n", ptr);
  }
  // Function to release all allocated memory
  void free() {
    // loop over all the live blocks and cache block and free the memory!
    for (auto &bin : live_blocks) {

      for (auto &block : bin) {
        if (debug_) {
          printf("Live block %p getting released\n", block.ptr);
        }
        std::free(block.ptr);
      }
    }
    for (auto &bin : cached_blocks) {
      for (auto &block : bin) {
        if (debug_) {
          printf("Cahced block %p getting released\n", block.ptr);
        }
        std::free(block.ptr);
      }
    }
    // Clear the vectors
    live_blocks.clear();
    cached_blocks.clear();
  }

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

  std::size_t findBin(std::size_t size) {
    // Find the appropriate bin index for the given size, rounding up
    return static_cast<std::size_t>(
        std::ceil(std::log(size) / std::log(GROWTH)));
  }
};

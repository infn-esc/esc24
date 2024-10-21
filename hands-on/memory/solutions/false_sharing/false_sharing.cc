#include <chrono>
#include <iostream>
#include <mutex>
#include <stdlib.h>
#include <thread>
#include <vector>
#include <new>

constexpr size_t numThreads = 2;
constexpr long long unsigned int iterations = 1UL << 30;
constexpr std::size_t cache_line_size = std::hardware_destructive_interference_size;

template <typename TResolution> class Timer {
public:
  Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}

  void reset() { start_time_ = std::chrono::high_resolution_clock::now(); }

  double elapsed() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, TResolution> elapsed_time =
        end_time - start_time_;
    return elapsed_time.count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_time_;
};

// Mutex for output
std::mutex printMutex;

struct Counter {
  int value = 0;
};

// fix this one!
struct alignas(cache_line_size) GoodCounter {
  int value = 0;
};

// Let's wrap it under another struct, to emulate a real datastructure
template <typename TCounter> struct Data {
  TCounter counters[numThreads];

  // Function for counting using the Padded datastructure
  void counting(size_t index) {
    for (size_t i = 0; i < iterations; ++i) {
      counters[index].value++;
    }
#ifdef DEBUG
    // compile with -DDEBUG
    {
      std::lock_guard<std::mutex> lock(printMutex);
      std::cout << "Thread " << index << " modifying counters[" << index
                << "] at address: " << &(counters[index]) << std::endl;
    }
#endif
  }
};

int main() {
  Data<Counter> data;               // Create object
  std::vector<std::thread> threads; // Define vector of threads

  // Let's also measure the time spent
  Timer<std::milli> timer;
  // Creating the threads for counting
  for (size_t i = 0; i < numThreads; ++i) {
    threads.emplace_back([&data, i]() {
      data.counting(i);
    }); // Pass the index and data reference
  }

  // Joining threads, waiting for all the threads to finish
  for (auto &thread : threads) {
    thread.join();
  }
  std::cout << "Time taken: " << timer.elapsed() << " milliseconds"
            << std::endl;

  // Let's print the final results and the time it took
  for (size_t i = 0; i < numThreads; ++i) {
    std::cout << "Counter[" << i << "] = " << data.counters[i].value
              << std::endl;
  }

  Data<GoodCounter> good_data; // Create object
  threads.clear();             // Define vector of threads

  // Let's also measure the time spent
  timer.reset();
  // Creating the threads for counting
  for (size_t i = 0; i < numThreads; ++i) {
    threads.emplace_back([&good_data, i]() {
      good_data.counting(i);
    }); // Pass the index and data reference
  }

  // Joining threads, waiting for all the threads to finish
  for (auto &thread : threads) {
    thread.join();
  }
  std::cout << "Time taken: " << timer.elapsed() << " milliseconds"
            << std::endl;

  // Let's print the final results and the time it took
  for (size_t i = 0; i < numThreads; ++i) {
    std::cout << "Counter[" << i << "] = " << data.counters[i].value
              << std::endl;
  }

  return 0;
}


#ifndef __OOBLECK_TIMING_UTILS_H__
#define __OOBLECK_TIMING_UTILS_H__

#include <chrono>
#include <iostream>

#define DEBUG
#ifdef DEBUG
#define PRINT(x) std::cout << "[DEBUG]: " << x << std::endl
#else
#define PRINT(x)
#endif

namespace oobleck {
  extern bool is_timing_starts;
  extern std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

  void inline start_timing() {
    is_timing_starts = true;
    start_time = std::chrono::high_resolution_clock::now();
  }

  void inline end_timing(const std::string& message) {
    if (!is_timing_starts) {
      std::cerr << "Timing is not started yet." << std::endl;
      return;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << message << " took " << duration.count() << " ms." << std::endl;
    is_timing_starts = false;
  }
}
#endif // __OOBLECK_TIMING_UTILS_H__
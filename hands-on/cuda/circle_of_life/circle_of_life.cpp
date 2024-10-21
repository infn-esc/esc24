// Original Author: Felice Pantaleo (CERN), 2024

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>
#include <chrono>

#include "gif.h"

// Compile-time variable to control saving grids
constexpr bool SAVE_GRIDS = false; // Set to true to enable GIF output

void print_help() {
  std::cout << "Prey-Predator Simulation with Custom Rules\n\n";
  std::cout << "Usage: game_of_life [OPTIONS]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --help                         Display this help message\n";
  std::cout << "  --seed <value>                 Set the random seed\n";
  std::cout << "  --weights <empty> <predator> <prey> Set the integer weights "
               "for cell states\n";
  std::cout
      << "  --width <value>                Set the grid width (default: 200)\n";
  std::cout << "  --height <value>               Set the grid height (default: "
               "200)\n";
  std::cout << "  --verify <file>                Verify the grid against a "
               "reference file\n";
  std::cout << "\n";
  std::cout << "Simulation Rules:\n";
  std::cout << "- An empty cell becomes a prey if there are more than two "
               "preys surrounding it.\n";
  std::cout
      << "- A prey cell becomes empty if there is a single predator "
         "surrounding it and its level is higher than prey's level minus 10.\n";
  std::cout << "- A prey cell becomes a predator with level equal to max "
               "predator level + 1 if there are more than two predators and "
               "its level is smaller than the sum of the levels of the "
               "predators surrounding it.\n";
  std::cout << "- A prey cell becomes empty if there are no empty spaces "
               "surrounding it.\n";
  std::cout << "- A prey cell's level is increased by one if it survives "
               "starvation.\n";
  std::cout << "- A predator cell becomes empty if there are no preys "
               "surrounding it, or if all preys have levels higher than or "
               "equal to the predator's level.\n";
  std::cout << "\n";
}

enum class CellState : char { Empty = 0, Predator = 1, Prey = 2 };

struct Cell {
  CellState state;
  uint8_t level; // Level (1-255 for colored cells, 0 for empty)
};

using Grid = std::vector<std::vector<Cell>>;

Grid initialize_grid(size_t width, size_t height, int weight_empty,
                     int weight_predator, int weight_prey, std::mt19937 &gen) {
  Grid grid(height, std::vector<Cell>(width));

  // Fix for narrowing conversion warnings
  std::vector<double> weights = {static_cast<double>(weight_empty),
                                 static_cast<double>(weight_predator),
                                 static_cast<double>(weight_prey)};
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  for (auto &row : grid)
    for (auto &cell : row) {
      cell.state = static_cast<CellState>(dist(gen));
      if (cell.state == CellState::Predator || cell.state == CellState::Prey)
        cell.level = 50; // Initialize level to 50 for colored cells
      else
        cell.level = 0; // Empty cells have level 0
    }

  return grid;
}

struct NeighborData {
  std::vector<uint8_t> predator_levels;
  std::vector<uint8_t> prey_levels;
  uint8_t max_predator_level = 0;
  uint8_t max_prey_level = 0;
  int sum_predator_levels = 0;
  int sum_prey_levels = 0;
  int empty_neighbors = 0;
};

NeighborData gather_neighbor_data(const Grid &grid, int x, int y) {
  NeighborData data;
  int height = grid.size();
  int width = grid[0].size();

  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0)
        continue;
      int nx = (x + dx + width) % width;
      int ny = (y + dy + height) % height;
      const Cell &neighbor = grid[ny][nx];
      if (neighbor.state == CellState::Predator) {
        data.predator_levels.push_back(neighbor.level);
        data.max_predator_level =
            std::max(data.max_predator_level, neighbor.level);
        data.sum_predator_levels += neighbor.level;
      } else if (neighbor.state == CellState::Prey) {
        data.prey_levels.push_back(neighbor.level);
        data.max_prey_level = std::max(data.max_prey_level, neighbor.level);
        data.sum_prey_levels += neighbor.level;
      } else if (neighbor.state == CellState::Empty) {
        data.empty_neighbors++;
      }
    }
  return data;
}

void update_grid_sequential(const Grid &current_grid, Grid &new_grid) {
  size_t height = current_grid.size();
  size_t width = current_grid[0].size();

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      const Cell &current_cell = current_grid[y][x];
      Cell &new_cell = new_grid[y][x];

      NeighborData neighbors = gather_neighbor_data(current_grid, x, y);

      if (current_cell.state == CellState::Empty) {
        // Empty cell becomes Prey if more than two Preys surround it
        if (neighbors.prey_levels.size() >= 2) {
          new_cell.state = CellState::Prey;
          uint8_t max_prey_level = neighbors.max_prey_level;
          new_cell.level = (max_prey_level < 255) ? max_prey_level + 1 : 255;
        } else {
          // Remains Empty
          new_cell.state = CellState::Empty;
          new_cell.level = 0;
        }
      } else if (current_cell.state == CellState::Prey) {
        bool action_taken = false;

        // Prey becomes Empty if single Predator with higher level
        if (neighbors.predator_levels.size() == 1) {
          uint8_t predator_level = neighbors.predator_levels[0];
          uint8_t prey_level_minus_10 =
              (current_cell.level >= 10) ? current_cell.level - 10 : 0;
          if (predator_level > prey_level_minus_10) {
            new_cell.state = CellState::Empty;
            new_cell.level = 0;
            action_taken = true;
          }
        }

        // Prey becomes Empty if too many Preys surrounding it
        if (neighbors.prey_levels.size() > 2) {
          new_cell.state = CellState::Empty;
          new_cell.level = 0;
          action_taken = true;
        }

        // Prey becomes Predator under certain conditions
        if (!action_taken && neighbors.predator_levels.size() > 1 &&
            current_cell.level < neighbors.sum_predator_levels) {
          new_cell.state = CellState::Predator;
          uint8_t max_level =
              std::max(neighbors.max_predator_level, neighbors.max_prey_level);
          new_cell.level = (max_level < 255) ? max_level + 1 : 255;
          action_taken = true;
        }

        // Prey becomes Empty if no Empty neighbors
        if (!action_taken && (neighbors.empty_neighbors == 0 or
                              neighbors.prey_levels.size() > 3)) {
          new_cell.state = CellState::Empty;
          new_cell.level = 0;
          action_taken = true;
        }

        // Prey survives
        if (!action_taken) {
          new_cell.state = CellState::Prey;
          if (neighbors.prey_levels.size() < 3) {
            new_cell.level = static_cast<int>(current_cell.level + 1) <= 255
                                 ? current_cell.level + 1
                                 : 255;
          } else {
            new_cell.level = current_cell.level;
          }
        }
      } else if (current_cell.state == CellState::Predator) {
        // Predator dies if no Preys or all Preys have higher or equal levels
        if (neighbors.prey_levels.size() == 0) {
          // Predator dies
          new_cell.state = CellState::Empty;
          new_cell.level = 0;
        } else {
          bool all_prey_higher = true;
          for (uint8_t prey_level : neighbors.prey_levels) {
            if (current_cell.level >= prey_level) {
              all_prey_higher = false;
              break;
            }
          }

          if (all_prey_higher) {
            // Predator dies
            new_cell.state = CellState::Empty;
            new_cell.level = 0;
          } else {
            // Predator survives
            new_cell.state = CellState::Predator;
            new_cell.level = static_cast<int>(current_cell.level + 1) <= 255
                                 ? current_cell.level + 1
                                 : 255;
            ;
          }
        }
      }
    }
  }
}

void save_frame_as_gif(const Grid &grid, GifWriter &writer) {
  if constexpr (SAVE_GRIDS) {
    int width = grid[0].size();
    int height = grid.size();
    std::vector<uint8_t> image(4 * width * height, 255); // RGBA image

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        size_t idx = 4 * (y * width + x);
        const Cell &cell = grid[y][x];
        if (cell.state == CellState::Predator) {
          image[idx] = 0;              // R
          image[idx + 1] = 0;          // G
          image[idx + 2] = cell.level; // B
          image[idx + 3] = 255;        // A
        } else if (cell.state == CellState::Prey) {
          image[idx] = cell.level; // R
          image[idx + 1] = 0;      // G
          image[idx + 2] = 0;      // B
          image[idx + 3] = 255;    // A
        } else {
          image[idx] = 0;       // R
          image[idx + 1] = 255; // G
          image[idx + 2] = 0;   // B
          image[idx + 3] = 255; // A
        }
      }
    }
    // Set delay to 50 (hundredths of a second) for two iterations per second
    GifWriteFrame(&writer, image.data(), width, height, 50);
  }
}

void print_grid(const Grid &grid) {
  // Clear the screen
  std::cout << "\033[2J\033[1;1H";
  for (const auto &row : grid) {
    for (const auto &cell : row) {
      if (cell.state == CellState::Predator)
        std::cout << "\033[38;2;0;0;" << +cell.level
                  << "mO\033[0m"; // Blue with intensity
      else if (cell.state == CellState::Prey)
        std::cout << "\033[38;2;" << +cell.level
                  << ";0;0mO\033[0m"; // Red with intensity
      else
        std::cout << ' ';
    }
    std::cout << '\n';
  }
}

void save_grid_to_file(const Grid &grid, const std::string &filename) {
  std::ofstream ofs(filename);
  for (const auto &row : grid) {
    for (const auto &cell : row) {
      ofs << static_cast<int>(cell.state) << ' ' << static_cast<int>(cell.level)
          << ' ';
    }
    ofs << '\n';
  }
}

bool load_grid_from_file(Grid &grid, const std::string &filename) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "Error: Cannot open reference file " << filename << '\n';
    return false;
  }

  for (auto &row : grid) {
    for (auto &cell : row) {
      int state_int;
      int level_int;
      ifs >> state_int >> level_int;
      if (ifs.fail()) {
        std::cerr << "Error: Invalid data in reference file.\n";
        return false;
      }
      cell.state = static_cast<CellState>(state_int);
      cell.level = static_cast<uint8_t>(level_int);
    }
  }
  return true;
}

bool compare_grids(const Grid &grid1, const Grid &grid2) {
  size_t height = grid1.size();
  size_t width = grid1[0].size();
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x) {
      if (grid1[y][x].state != grid2[y][x].state)
        return false;
      if (grid1[y][x].level != grid2[y][x].level)
        return false;
    }
  return true;
}

int main(int argc, char *argv[]) {
  // Start with a grid 200x200
  size_t width = 200;
  size_t height = 200;
  unsigned int seed = 0; // Default seed
  bool seed_provided = false;
  int weight_empty = 5;
  int weight_predator = 1;
  int weight_prey = 1;
  std::string verify_filename;

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help") {
      print_help();
      return 0;
    } else if (arg == "--seed") {
      if (i + 1 < argc) {
        seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        seed_provided = true;
      } else {
        std::cerr << "Error: --seed option requires an argument.\n";
        return 1;
      }
    } else if (arg == "--weights") {
      if (i + 3 < argc) {
        weight_empty = std::stoi(argv[++i]);
        weight_predator = std::stoi(argv[++i]);
        weight_prey = std::stoi(argv[++i]);
        if (weight_empty < 0 || weight_predator < 0 || weight_prey < 0) {
          std::cerr << "Error: Weights cannot be negative.\n";
          return 1;
        }
        if (weight_empty == 0 && weight_predator == 0 && weight_prey == 0) {
          std::cerr << "Error: At least one weight must be positive.\n";
          return 1;
        }
      } else {
        std::cerr << "Error: --weights option requires three arguments.\n";
        return 1;
      }
    } else if (arg == "--width") {
      if (i + 1 < argc) {
        width = std::stoul(argv[++i]);
      } else {
        std::cerr << "Error: --width option requires an argument.\n";
        return 1;
      }
    } else if (arg == "--height") {
      if (i + 1 < argc) {
        height = std::stoul(argv[++i]);
      } else {
        std::cerr << "Error: --height option requires an argument.\n";
        return 1;
      }
    } else if (arg == "--verify") {
      if (i + 1 < argc) {
        verify_filename = argv[++i];
      } else {
        std::cerr << "Error: --verify option requires a filename.\n";
        return 1;
      }
    } else {
      std::cerr << "Invalid argument: " << arg
                << ". Use --help for usage information.\n";
      return 1;
    }
  }

  // Initialize random number generator
  if (!seed_provided) {
    seed = std::random_device{}();
  }
  std::mt19937 gen(seed);

  const size_t NUM_ITERATIONS = 500; // Total number of iterations
  Grid grid = initialize_grid(width, height, weight_empty, weight_predator,
                              weight_prey, gen);
  Grid new_grid = grid;

  // Generate reference filename
  std::string reference_filename =
      "reference_" + std::to_string(width) + "_" + std::to_string(height) +
      "_" + std::to_string(seed) + "_" + std::to_string(weight_empty) + "_" +
      std::to_string(weight_predator) + "_" + std::to_string(weight_prey) +
      ".txt";

  // Initialize GIF writer
  GifWriter writer = {};
  if constexpr (SAVE_GRIDS) {
    // Set delay to 50 (hundredths of a second) for two iterations per second
    if (!GifBegin(&writer, "simulation.gif", width, height, 50)) {
      std::cerr << "Error: Failed to initialize GIF writer.\n";
      return 1;
    }
  }

  // Simulation loop
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
    update_grid_sequential(grid, new_grid);
    save_frame_as_gif(grid, writer);
    std::swap(grid, new_grid);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time of sequential version: " << elapsed_seconds.count() << "s\n";

  if constexpr (SAVE_GRIDS) {
    GifEnd(&writer);
    std::cout << "Simulation saved as 'simulation.gif'.\n";
  }

  if (!verify_filename.empty()) {
    // Load the reference grid and compare after simulation
    Grid reference_grid(height, std::vector<Cell>(width));
    if (!load_grid_from_file(reference_grid, verify_filename)) {
      return 1;
    }
    if (compare_grids(grid, reference_grid)) {
      std::cout << "Verification successful: The grids match.\n";
    } else {
      std::cerr << "Verification failed: The grids do not match.\n";
      return 1;
    }
  } else {
    // Save the final grid to a reference file
    save_grid_to_file(grid, reference_filename);
    std::cout << "Reference grid saved to " << reference_filename << '\n';
  }

  return 0;
}

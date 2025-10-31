// cppimport

#include <chrono> // For seeding with high-resolution clock
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random> // For random number generation

namespace py = pybind11;
typedef py::array_t<uint8_t> Grid;
int const U = 1;
int const D = 2;
int const L = 3;
int const R = 4;
int const EMPTY = 5;
int const ERROR = 6;
int const CORNER = 7;

std::mt19937 engine(
    std::chrono::high_resolution_clock::now().time_since_epoch().count());
std::uniform_int_distribution<int> dist(0, 1);

Grid make_aztec_diamond(Grid grid) {
  int n = grid.shape()[0];
  // grid.mutable_at(0, 0) = CORNER;
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      if (row + col + 1 < n / 2) {
        grid.mutable_at(row, col) = CORNER;
        grid.mutable_at(n - row - 1, col) = CORNER;
        grid.mutable_at(row, n - col - 1) = CORNER;
        grid.mutable_at(n - row - 1, n - col - 1) = CORNER;
      }
    }
  }
  return grid;
}
void expand_grid(Grid grid, Grid expanded) {
  int n = grid.shape()[0];
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      if (grid.at(row, col) == U) {
        expanded.mutable_at(row, col + 1) = U;
      }
      if (grid.at(row, col) == D) {
        expanded.mutable_at(row + 2, col + 1) = D;
      }
      if (grid.at(row, col) == L) {
        expanded.mutable_at(row + 1, col) = L;
      }
      if (grid.at(row, col) == R) {
        expanded.mutable_at(row + 1, col + 2) = R;
      }
    }
  }
}
Grid remove_facing(Grid grid) {
  int n = grid.shape()[0];
  for (int row = 0; row < n - 1; ++row) {
    for (int col = 0; col < n - 1; ++col) {
      if (grid.at(row, col) == R && grid.at(row, col + 1) == L) {
        grid.mutable_at(row, col) = EMPTY;
        grid.mutable_at(row + 1, col) = EMPTY;
        grid.mutable_at(row, col + 1) = EMPTY;
        grid.mutable_at(row + 1, col + 1) = EMPTY;
      }
      if (grid.at(row, col) == D && grid.at(row + 1, col) == U) {
        grid.mutable_at(row, col) = EMPTY;
        grid.mutable_at(row + 1, col) = EMPTY;
        grid.mutable_at(row, col + 1) = EMPTY;
        grid.mutable_at(row + 1, col + 1) = EMPTY;
      }
    }
  }
  return grid;
}
Grid fill_empty_rand(Grid grid) {
  int n = grid.shape()[0];
  for (int row = 0; row < n - 1; ++row) {
    for (int col = 0; col < n - 1; ++col) {
      if (grid.at(row, col) == EMPTY) {
        if (dist(engine) % 2) {
          grid.mutable_at(row, col) = U;
          grid.mutable_at(row + 1, col) = D;
          grid.mutable_at(row, col + 1) = U;
          grid.mutable_at(row + 1, col + 1) = D;
        } else {
          grid.mutable_at(row, col) = L;
          grid.mutable_at(row + 1, col) = L;
          grid.mutable_at(row, col + 1) = R;
          grid.mutable_at(row + 1, col + 1) = R;
        }
      }
    }
  }
  return grid;
}

PYBIND11_MODULE(arctic_circle_compiled, m) {
  m.def("make_aztec_diamond", &make_aztec_diamond);
  m.def("expand_grid", &expand_grid);
  m.def("remove_facing", &remove_facing);
  m.def("fill_empty_rand", &fill_empty_rand);
}

/*
<%
cfg['extra_compile_args'] = ['-std=c++17', '-w', '-Ofast']
setup_pybind11(cfg)
%>
*/

// for (int i = 0; i < n / 2; ++i) {
//   for (int j = 0; j < n / 2; ++j) {
//     if (i + j < n / 2 - 1) {
//       grid.mutable_at(i, j) = CORNER;
//       grid.mutable_at(n - i - 1, j) = CORNER;
//       grid.mutable_at(i, n - j - 1) = CORNER;
//       grid.mutable_at(n - i - 1, n - j - 1) = CORNER;
//     }
//   }
// }

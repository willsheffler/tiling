// cppimport
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
typedef py::array_t<uint8_t> Grid;
int const U = 1;
int const D = 2;
int const L = 3;
int const R = 4;
int const EMPTY = 5;
int const ERROR = 6;
int const CORNER = 7;

Grid make_aztec_diamond(Grid grid) {
  int n = grid.shape()[0];
  if (n == 4) {
    grid.mutable_at(0, 0) = CORNER;
    grid.mutable_at(0, n - 1) = CORNER;
    grid.mutable_at(n - 1, 0) = CORNER;
    grid.mutable_at(n - 1, n - 1) = CORNER;
  }
  return grid;
}
Grid expand_grid(Grid grid, Grid expanded) {
  int n = grid.shape()[0];
  return expanded;
}
Grid remove_facing(Grid grid) {
  int n = grid.shape()[0];
  return grid;
}
Grid fill_empty_rand(Grid grid) {
  int n = grid.shape()[0];
  return grid;
}

PYBIND11_MODULE(arctic_circle_compiled, m) {
  // m.def("make_aztec_diamond", &make_aztec_diamond);
  // m.def("expand_grid", &expand_grid);
  // m.def("remove_facing", &remove_facing);
  // m.def("fill_empty_rand", &fill_empty_rand);
}

/*
<%
cfg['extra_compile_args'] = ['-std=c++17', '-w']
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

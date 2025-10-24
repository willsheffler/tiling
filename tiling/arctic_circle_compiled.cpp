// cppimport
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

int square(int x) { return x * x; }

py::array_t<uint8_t> make_aztec_diamond(py::array_t<uint8_t> grid) {
  return grid;
}
py::array_t<uint8_t> expand_grid(py::array_t<uint8_t> grid,
                                 py::array_t<uint8_t> expanded) {
  return expanded;
}
py::array_t<uint8_t> remove_facing(py::array_t<uint8_t> grid) { return grid; }
py::array_t<uint8_t> fill_empty_rand(py::array_t<uint8_t> grid) { return grid; }

PYBIND11_MODULE(arctic_circle_compiled, m) {
  // m.def("make_aztec_diamond", &make_aztec_diamond);
  // m.def("expand_grid", &expand_grid);
  // m.def("remove_facing", &remove_facing);
  // m.def("fill_empty_rand", &fill_empty_rand);
}

/*
<%
setup_pybind11(cfg)
%>
*/

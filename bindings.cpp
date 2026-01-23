#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Engine.h"

namespace py = pybind11;

PYBIND11_MODULE(engine_cpp, m) {
    m.def("run_backtest_from_json", &run_backtest_from_json,
          "Run backtest from JSON input and return JSON output");
}

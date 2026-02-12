#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Crucial para converter std::string e std::vector
#include "engine.h"

namespace py = pybind11;

PYBIND11_MODULE(engine_cpp, m) {
    m.doc() = "ART Engine Core";

    m.def("run", [](const std::string& json_input) {
        Engine engine;
        return engine.run(json_input);
    }, "Executa backtest via JSON string");
}
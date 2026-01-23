#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>

#include "Engine.h" // só o header com 'json run_engine(const json& data);'

namespace py = pybind11;
using json = nlohmann::json;

PYBIND11_MODULE(engine_cpp, m) {
    m.doc() = "Engine CPP module";

    m.def("run_engine", [](py::dict input){
        py::object json_module = py::module_::import("json");
        std::string json_str = py::str(json_module.attr("dumps")(input));
        json data = json::parse(json_str);

        json result = run_engine(data); // chama a função do Engine.cpp

        return json_module.attr("loads")(py::str(result.dump()));
    });
}

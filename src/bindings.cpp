#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Engine.h"
#include "Trade.h"
#include "trades_to_json.h"

namespace py = pybind11;

PYBIND11_MODULE(engine_cpp, m) {
    m.doc() = "High-performance backtest engine with C++ core";

    py::class_<Trade>(m, "Trade")
        .def_readonly("id", &Trade::id)
        .def_readonly("asset", &Trade::asset)
        .def_readonly("status", &Trade::status)
        .def_readonly("direction", &Trade::direction)
        .def_readonly("entry_price", &Trade::entry_price)
        .def_readonly("entry_datetime", &Trade::entry_datetime)
        .def_readonly("lot_size", &Trade::lot_size)
        .def_readonly("stop_loss", &Trade::stop_loss)
        .def_readonly("take_profit", &Trade::take_profit)
        .def_readonly("exit_price", &Trade::exit_price)
        .def_readonly("exit_datetime", &Trade::exit_datetime)
        .def_readonly("exit_reason", &Trade::exit_reason)
        .def_readonly("profit", &Trade::profit)
        .def_readonly("profit_r", &Trade::profit_r);

    m.def("run", [](const std::string& json_input) {
            auto trades = Engine::run(json_input);
            auto j = trades_to_json(trades);
            return j.dump();
        });
}

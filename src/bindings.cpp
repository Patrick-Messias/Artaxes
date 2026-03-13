#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <nlohmann/json.hpp>
#include "engine.h"

namespace py = pybind11;
using json = nlohmann::json;

// --- Conversor: Python -> JSON ---
static json py_to_json(const py::object& obj) {
    if (obj.is_none())                   return nullptr;
    if (py::isinstance<py::bool_>(obj))   return obj.cast<bool>();
    if (py::isinstance<py::int_>(obj))    return obj.cast<int64_t>();
    if (py::isinstance<py::float_>(obj))  return obj.cast<double>();
    if (py::isinstance<py::str>(obj))     return obj.cast<std::string>();

    if (py::isinstance<py::dict>(obj)) {
        json j = json::object();
        for (auto item : obj.cast<py::dict>()) {
            j[item.first.cast<std::string>()] = py_to_json(item.second.cast<py::object>());
        }
        return j;
    }

    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        json j = json::array();
        for (auto el : obj.cast<py::sequence>()) {
            j.push_back(py_to_json(el.cast<py::object>()));
        }
        return j;
    }

    try { return obj.cast<double>(); } catch (...) {}
    return nullptr;
}

// --- Conversor: JSON -> Python ---
static py::object json_to_py(const json& j) {
    switch (j.type()) {
        case json::value_t::null:             return py::none();
        case json::value_t::boolean:          return py::bool_(j.get<bool>());
        case json::value_t::number_integer:
        case json::value_t::number_unsigned:  return py::int_(j.get<int64_t>());
        case json::value_t::number_float:     return py::float_(j.get<double>());
        case json::value_t::string:           return py::str(j.get<std::string>());
        case json::value_t::array: {
            py::list lst;
            for (const auto& el : j) lst.append(json_to_py(el));
            return lst;
        }
        case json::value_t::object: {
            py::dict d;
            for (const auto& [k, v] : j.items()) d[py::str(k)] = json_to_py(v);
            return d;
        }
        default: return py::none();
    }
}

// --- Conversor de Resultados para Python ---
static py::dict engine_result_to_pydict(const EngineResult& res) {
    py::list all_sims;
    for (const auto& trades_json : res.simulations) {
        py::list sim_trades;
        for (const auto& t : trades_json) sim_trades.append(json_to_py(t));
        all_sims.append(sim_trades);
    }

    py::list wfm;
    // CORREÇÃO: res.wfm_data já contém nlohmann::json vindo do Operation::run
    for (const auto& row_json : res.wfm_data) {
        wfm.append(json_to_py(row_json));
    }

    py::dict out;
    out["simulations"] = all_sims;
    out["wfm_data"]    = wfm;
    return out;
}

// --- Wrapper Principal (Zero-Copy Bridge) ---
static py::dict execute_wrapper(
    const std::string& header,
    const std::unordered_map<std::string, py::array_t<double, py::array::c_style>>& ohlc_arrays,
    const py::array_t<int64_t, py::array::c_style>& datetime_int,
    const std::unordered_map<std::string, py::array_t<double, py::array::c_style>>& indicators_pool,
    const std::unordered_map<std::string, py::array_t<uint8_t, py::array::c_style>>& shared_signal_arrays,
    const py::list& sim_params_py,
    const py::object& exec_settings_py
) {
    std::unordered_map<std::string, const double*> ohlc_ptrs;
    for (const auto& [k, arr] : ohlc_arrays) ohlc_ptrs[k] = arr.data();

    std::unordered_map<std::string, const double*> ind_ptrs;
    for (const auto& [k, arr] : indicators_pool) ind_ptrs[k] = arr.data();

    std::unordered_map<std::string, const uint8_t*> shared_sig_ptrs;
    for (const auto& [k, arr] : shared_signal_arrays) shared_sig_ptrs[k] = arr.data();

    const size_t n_bars = (size_t)datetime_int.size();
    std::vector<int64_t> dt_vec(datetime_int.data(), datetime_int.data() + n_bars);

    std::vector<SimParams> sims;
    sims.reserve((size_t)py::len(sim_params_py));

    for (auto sim_obj : sim_params_py) {
        py::dict sim = sim_obj.cast<py::dict>();
        SimParams sp;
        sp.id = sim["id"].cast<std::string>();
        sp.params = py_to_json(sim["params"].cast<py::object>());

        if (sim.contains("signal_arrays")) {
            for (auto item : sim["signal_arrays"].cast<py::dict>()) {
                auto arr = item.second.cast<py::array_t<uint8_t, py::array::c_style>>();
                sp.signal_array_bufs[item.first.cast<std::string>()] =
                    std::vector<uint8_t>(arr.data(), arr.data() + arr.size());
            }
        }

        if (sim.contains("signal_refs")) {
            for (auto item : sim["signal_refs"].cast<py::dict>()) {
                sp.signal_refs[item.first.cast<std::string>()] = item.second.cast<std::string>();
            }
        }
        sims.push_back(std::move(sp));
    }

    json exec_settings = py_to_json(exec_settings_py);

    EngineResult res = Engine::execute(
        header, ohlc_ptrs, n_bars, dt_vec, ind_ptrs,
        shared_sig_ptrs, sims, exec_settings
    );

    return engine_result_to_pydict(res);
}

PYBIND11_MODULE(engine_cpp, m) {
    m.doc() = "ART Engine Core — Zero-Copy Bridge";
    m.def("execute", &execute_wrapper,
        py::arg("header"),
        py::arg("ohlc_arrays"),
        py::arg("datetime_int"),
        py::arg("indicators_pool"),
        py::arg("shared_signal_arrays"),
        py::arg("sim_params"),
        py::arg("exec_settings")
    );
}
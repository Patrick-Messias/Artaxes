#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <nlohmann/json.hpp>
#include "engine.h"
#include "Trade.h"

namespace py = pybind11;
using json = nlohmann::json;

// ── Conversores leves ─────────────────────────────────────────────────────────

static json py_to_json(const py::object& obj) {
    if (obj.is_none())                    return nullptr;
    if (py::isinstance<py::bool_>(obj))   return obj.cast<bool>();
    if (py::isinstance<py::int_>(obj))    return obj.cast<int64_t>();
    if (py::isinstance<py::float_>(obj))  return obj.cast<double>();
    if (py::isinstance<py::str>(obj))     return obj.cast<std::string>();
    if (py::isinstance<py::dict>(obj)) {
        json j = json::object();
        for (auto& item : obj.cast<py::dict>())
            j[item.first.cast<std::string>()] = py_to_json(item.second.cast<py::object>());
        return j;
    }
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        json j = json::array();
        for (const auto el : obj.cast<py::sequence>())
            j.push_back(py_to_json(el.cast<py::object>()));
        return j;
    }
    try { return obj.cast<double>(); } catch (...) {}
    return nullptr;
}

// Trade → py::dict direto, sem nlohmann intermediário
static py::dict trade_to_pydict(const Trade& t) {
    py::dict d;
    d["id"]             = py::str(t.id);
    d["asset"]          = py::str(t.asset);
    d["path"]           = py::str(t.path);
    d["status"]         = py::str(t.status);
    d["entry_datetime"] = py::str(t.entry_datetime);
    d["entry_price"]    = py::float_(t.entry_price);
    d["lot_size"]       = py::float_(t.lot_size);
    d["stop_loss"]      = t.stop_loss  > 0.0 ? py::object(py::float_(t.stop_loss))  : py::none();
    d["take_profit"]    = t.take_profit> 0.0 ? py::object(py::float_(t.take_profit)): py::none();
    d["mfe"]            = py::float_(t.mfe);
    d["mae"]            = py::float_(t.mae);
    d["bars_held"]      = py::int_(t.bars_held);
    if (t.closed) {
        d["exit_price"]    = py::float_(t.exit_price);
        d["exit_datetime"] = py::str(t.exit_datetime);
        d["exit_reason"]   = py::str(t.exit_reason);
        d["profit"]        = py::float_(t.profit);
        d["profit_r"]      = py::float_(t.profit_r);
    } else {
        d["exit_price"]    = py::none();
        d["exit_datetime"] = py::none();
        d["exit_reason"]   = py::none();
        d["profit"]        = py::none();
        d["profit_r"]      = py::none();
    }
    return d;
}

//DailyRow -> py::dict
static py::dict dailyrow_to_pydict(const DailyResult& d) {
    py::dict p;
    p["ts"]    = py::int_(d.ts);
    p["pnl"]   = py::float_(d.pnl);
    p["ps_id"] = py::int_(d.ps_id);
    return p;
}

static py::dict engine_result_to_pydict(const EngineResult& res) {
    py::list all_sims;
    for (const auto& sim_trades : res.simulations) {
        py::list sim_list;
        for (const auto& t : sim_trades)
            sim_list.append(trade_to_pydict(t));
        all_sims.append(sim_list);
    }
    py::list wfm;
    for (const auto& row : res.wfm_data)
        wfm.append(dailyrow_to_pydict(row));
    py::dict out;
    out["simulations"] = all_sims;
    out["wfm_data"]    = wfm;
    return out;
}

// ── Wrapper principal ─────────────────────────────────────────────────────────
static py::dict execute_wrapper(
    const std::string& header,
    const std::unordered_map<std::string, py::array_t<double,  py::array::c_style>>& ohlc_arrays,
    const py::array_t<int64_t, py::array::c_style>&                                   datetime_int,
    const std::unordered_map<std::string, py::array_t<double,  py::array::c_style>>& indicators_pool,
    const std::unordered_map<std::string, py::array_t<uint8_t, py::array::c_style>>& shared_signal_arrays,
    const py::list&   sim_params_py,
    const py::object& exec_settings_py
) {
    std::unordered_map<std::string, const double*>  ohlc_ptrs;
    for (const auto& [k, arr] : ohlc_arrays) ohlc_ptrs[k] = arr.data();

    std::unordered_map<std::string, const double*>  ind_ptrs;
    for (const auto& [k, arr] : indicators_pool) ind_ptrs[k] = arr.data();

    std::unordered_map<std::string, const uint8_t*> shared_sig_ptrs;
    for (const auto& [k, arr] : shared_signal_arrays) shared_sig_ptrs[k] = arr.data();

    const size_t n_bars = (size_t)datetime_int.size();
    std::vector<int64_t> dt_vec(datetime_int.data(), datetime_int.data() + n_bars);

    std::vector<SimParams> sims;
    sims.reserve((size_t)py::len(sim_params_py));
    for (const auto& sim_obj : sim_params_py) {
        py::dict sim = sim_obj.cast<py::dict>();
        SimParams sp;
        sp.id     = sim["id"].cast<std::string>();
        sp.params = py_to_json(sim["params"].cast<py::object>());
        if (sim.contains("signal_arrays")) {
            for (auto& item : sim["signal_arrays"].cast<py::dict>()) {
                auto arr = item.second.cast<py::array_t<uint8_t, py::array::c_style>>();
                sp.signal_array_bufs[item.first.cast<std::string>()] =
                    std::vector<uint8_t>(arr.data(), arr.data() + arr.size());
            }
        }
        if (sim.contains("signal_refs")) {
            for (auto& item : sim["signal_refs"].cast<py::dict>())
                sp.signal_refs[item.first.cast<std::string>()] =
                    item.second.cast<std::string>();
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
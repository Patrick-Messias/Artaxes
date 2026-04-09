#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <nlohmann/json.hpp>
#include "engine.h"
#include "Trade.h"
#include "Utils.h"

#include <iostream>

namespace py = pybind11;
using json = nlohmann::json;

// ── py_to_json ────────────────────────────────────────────────────────────────
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

// ── EngineResult → py::dict columnar ─────────────────────────────────────────
static py::dict engine_result_to_pydict(const EngineResult& res) {

    const size_t n_sims = res.simulations.size();
    size_t total_trades = 0;
    for (const auto& s : res.simulations) total_trades += s.size();
    const size_t n_wfm  = res.wfm_data.size();

    // ── Trades columnar ───────────────────────────────────────────────────────
    auto a_sim_offsets  = py::array_t<int32_t>(n_sims + 1);
    auto a_entry_price  = py::array_t<double>(total_trades);
    auto a_exit_price   = py::array_t<double>(total_trades);
    auto a_lot_size     = py::array_t<double>(total_trades);
    auto a_stop_loss    = py::array_t<double>(total_trades);
    auto a_take_profit  = py::array_t<double>(total_trades);
    auto a_profit       = py::array_t<double>(total_trades);
    auto a_profit_r     = py::array_t<double>(total_trades);
    //auto a_mfe          = py::array_t<double>(total_trades);
    //auto a_mae          = py::array_t<double>(total_trades);
    auto a_bars_held    = py::array_t<int32_t>(total_trades);
    auto a_closed       = py::array_t<uint8_t>(total_trades);

    py::list l_id(total_trades);
    py::list l_entry_dt(total_trades);
    py::list l_exit_dt(total_trades);
    py::list l_exit_reason(total_trades);
    py::list l_status(total_trades);

    auto* off           = a_sim_offsets.mutable_data();
    auto* p_entry_price = a_entry_price.mutable_data();
    auto* p_exit_price  = a_exit_price.mutable_data();
    auto* p_lot_size    = a_lot_size.mutable_data();
    auto* p_stop_loss   = a_stop_loss.mutable_data();
    auto* p_take_profit = a_take_profit.mutable_data();
    auto* p_profit      = a_profit.mutable_data();
    auto* p_profit_r    = a_profit_r.mutable_data();
    //auto* p_mfe         = a_mfe.mutable_data();
    //auto* p_mae         = a_mae.mutable_data();
    auto* p_bars_held   = a_bars_held.mutable_data();
    auto* p_closed      = a_closed.mutable_data();

    size_t idx = 0;
    for (size_t si = 0; si < n_sims; ++si) {
        off[si] = (int32_t)idx;
        for (const auto& t : res.simulations[si]) {
            p_entry_price[idx] = t.entry_price;
            p_exit_price[idx]  = t.exit_price;
            p_lot_size[idx]    = t.lot_size;
            p_stop_loss[idx]   = t.stop_loss;
            p_take_profit[idx] = t.take_profit;
            p_profit[idx]      = t.profit;
            p_profit_r[idx]    = t.profit_r;
            //p_mfe[idx]         = t.mfe;
            //p_mae[idx]         = t.mae;
            p_bars_held[idx]   = t.bars_held;
            p_closed[idx]      = t.closed ? 1 : 0;
            l_id[idx]          = py::str(t.id);
            l_entry_dt[idx]    = py::str(t.entry_datetime);
            l_exit_dt[idx]     = t.closed ? py::object(py::str(t.exit_datetime)) : py::none();
            l_exit_reason[idx] = t.closed ? py::object(py::str(t.exit_reason))   : py::none();
            l_status[idx]      = py::str(t.status);
            ++idx;
        }
    }
    off[n_sims] = (int32_t)idx;

    py::dict trades_col;
    trades_col["sim_offsets"]    = a_sim_offsets;
    trades_col["entry_price"]    = a_entry_price;
    trades_col["exit_price"]     = a_exit_price;
    trades_col["lot_size"]       = a_lot_size;
    trades_col["stop_loss"]      = a_stop_loss;
    trades_col["take_profit"]    = a_take_profit;
    trades_col["profit"]         = a_profit;
    trades_col["profit_r"]       = a_profit_r;
    //trades_col["mfe"]            = a_mfe;
    //trades_col["mae"]            = a_mae;
    trades_col["bars_held"]      = a_bars_held;
    trades_col["closed"]         = a_closed;
    trades_col["id"]             = l_id;
    trades_col["entry_datetime"] = l_entry_dt;
    trades_col["exit_datetime"]  = l_exit_dt;
    trades_col["exit_reason"]    = l_exit_reason;
    trades_col["status"]         = l_status;

    // ── WFM columnar ──────────────────────────────────────────────────────────
    //const size_t n_wfm = res.wfm_data.size();

    auto a_wfm_ts    = py::array_t<int64_t>(n_wfm);
    auto a_wfm_pnl   = py::array_t<double>(n_wfm);
    auto a_wfm_lot_size = py::array_t<double>(n_wfm);
    auto a_wfm_mae = py::array_t<double>(n_wfm);
    auto a_wfm_mfe = py::array_t<double>(n_wfm);
    //auto a_wfm_trade_id = py::array_t<str>(n_wfm);

    py::list l_wfm_trade_id(n_wfm);

    auto* p_ts    = a_wfm_ts.mutable_data();
    auto* p_pnl   = a_wfm_pnl.mutable_data();
    auto* p_wfm_lot_size = a_wfm_lot_size.mutable_data();
    auto* p_wfm_mae = a_wfm_mae.mutable_data();
    auto* p_wfm_mfe = a_wfm_mfe.mutable_data();
    //auto* p_trade_id = a_wfm_trade_id.mutable_data();

    for (size_t i = 0; i < n_wfm; ++i) {
        const auto& data = res.wfm_data[i];

        p_ts[i]    = res.wfm_data[i].ts;
        p_pnl[i]   = res.wfm_data[i].pnl;
        p_wfm_lot_size[i] = res.wfm_data[i].lot_size;
        p_wfm_mae[i] = res.wfm_data[i].mae;
        p_wfm_mfe[i] = res.wfm_data[i].mfe;
        
        //p_trade_id[i] = res.wfm_data[i].trade_id;
        l_wfm_trade_id[i] = py::cast(data.trade_id);
    }
    py::dict wfm_col;
    wfm_col["ts"]    = a_wfm_ts;
    wfm_col["pnl"]   = a_wfm_pnl;
    wfm_col["lot_size"] = a_wfm_lot_size;
    wfm_col["mae"] = a_wfm_mae;
    wfm_col["mfe"] = a_wfm_mfe;
    //wfm_col["trade_id"] = a_wfm_trade_id;
    wfm_col["trade_id"] = l_wfm_trade_id;
    
    py::dict out;
    out["trades_columnar"] = trades_col;
    out["wfm_columnar"]    = wfm_col;
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
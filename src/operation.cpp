#include "operation.h"
#include "backtest.h"
#include <execution>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>

using json = nlohmann::json;

EngineResult Operation::run(
    const std::string&                                    header,
    const std::unordered_map<std::string, const double*>& fast_pool,
    size_t                                                n_bars,
    const std::vector<int>&                               bar_dates,
    const std::vector<int>&                               bar_times,
    const std::vector<int>&                               bar_days,
    const json&                                           sim_params,
    const json&                                           exec_settings
) {
    std::vector<json> simulations;
    if (sim_params.is_array()) {
        simulations.reserve(sim_params.size());
        for (const auto& s : sim_params) simulations.push_back(s);
    }
    const size_t n_sims = simulations.size();

    // ── Pré-decodifica signal_data exclusivo de cada sim (fora do paralelo) ──
    using SigBuf = std::unordered_map<std::string, std::vector<double>>;
    std::vector<SigBuf> per_sim_sig_bufs(n_sims);
    for (size_t ps_id = 0; ps_id < n_sims; ++ps_id) {
        const json& sim = simulations[ps_id];
        if (sim.contains("signal_data") && sim["signal_data"].is_object()) {
            for (const auto& [key, val] : sim["signal_data"].items()) {
                try { per_sim_sig_bufs[ps_id][key] = val.get<std::vector<double>>(); }
                catch (...) {}
            }
        }
    }

    // ── Resultados indexados por ps_id — sem mutex ────────────────────────────
    std::vector<SimulationOutput> outputs(n_sims);

    std::vector<int> indexes((int)n_sims);
    std::iota(indexes.begin(), indexes.end(), 0);

    std::for_each(std::execution::par, indexes.begin(), indexes.end(), [&](int ps_id) {
        Backtest::SimView sim_view;
        sim_view.reserve(fast_pool.size() + per_sim_sig_bufs[ps_id].size());
        for (const auto& [key, ptr] : fast_pool)
            sim_view[key] = ptr;
        for (const auto& [key, vec] : per_sim_sig_bufs[ps_id])
            sim_view[key] = vec.data();

        outputs[ps_id] = Backtest::run_simulation(
            header, sim_view, n_bars,
            bar_dates, bar_times, bar_days,
            simulations[ps_id], exec_settings, ps_id
        );
    });

    // ── Monta EngineResult (C++ puro) ─────────────────────────────────────────
    EngineResult result;
    result.simulations.reserve(n_sims);
    for (size_t ps_id = 0; ps_id < n_sims; ++ps_id) {
        // trades → json array
        json trades_arr = json::array();
        for (const auto& t : outputs[ps_id].trades)
            trades_arr.push_back(trades_to_json({t})[0]);   // reutiliza helper de Utils.h
        result.simulations.push_back(std::move(trades_arr));

        // daily results → json objects
        for (const auto& dr : outputs[ps_id].daily_vec)
            result.wfm_data.push_back({{"ts", dr.ts}, {"pnl", dr.pnl}, {"id", dr.ps_id}});
    }
    return result;
}
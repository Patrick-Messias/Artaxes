#include "operation.h"
#include "backtest.h"
#include <execution>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>

using json = nlohmann::json;

EngineResult Operation::run(
    const std::string&                                      header,
    const std::unordered_map<std::string, const double*>&   fast_pool,
    size_t                                                  n_bars,
    const std::vector<int>&                                 bar_dates,
    const std::vector<int>&                                 bar_times,
    const std::vector<int>&                                 bar_days,
    const std::unordered_map<std::string, const uint8_t*>&  shared_signal_arrays,
    const std::vector<SimParams>&                           sim_params,
    const json&                                             exec_settings
) {
    const size_t n_sims = sim_params.size();

    std::vector<SimulationOutput> outputs(n_sims);
    std::vector<int> indexes((int)n_sims);
    std::iota(indexes.begin(), indexes.end(), 0);

    std::for_each(std::execution::par, indexes.begin(), indexes.end(), [&](int ps_id) {
        const SimParams& sp = sim_params[ps_id];

        // ── signal_view: shared + exclusivos desta sim ────────────────────────
        Backtest::SignalView signal_view;
        signal_view.reserve(shared_signal_arrays.size() + sp.signal_array_bufs.size());

        for (const auto& [k, ptr] : shared_signal_arrays)
            signal_view[k] = ptr;

        // Exclusivos sobrescrevem shared se houver conflito
        for (const auto& [k, buf] : sp.signal_array_bufs)
            signal_view[k] = buf.data();

        outputs[ps_id] = Backtest::run_simulation(
            header,
            fast_pool,
            signal_view,
            sp.signal_refs,
            n_bars,
            bar_dates, bar_times, bar_days,
            sp.params,
            exec_settings,
            ps_id
        );
    });

    EngineResult result;
    result.simulations.reserve(n_sims);
    for (size_t ps_id = 0; ps_id < n_sims; ++ps_id) {
        json trades_arr = json::array();
        for (const auto& t : outputs[ps_id].trades)
            trades_arr.push_back(trades_to_json({t})[0]);
        result.simulations.push_back(std::move(trades_arr));

        for (const auto& dr : outputs[ps_id].daily_vec)
            result.wfm_data.push_back({{"ts", dr.ts}, {"pnl", dr.pnl}, {"id", dr.ps_id}});
    }
    return result;
}
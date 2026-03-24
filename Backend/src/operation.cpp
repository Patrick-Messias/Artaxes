#include "operation.h"
#include "backtest.h"
#include <execution>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>
#include <chrono>

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

    auto t0 = std::chrono::high_resolution_clock::now();

    std::for_each(std::execution::par, indexes.begin(), indexes.end(), [&](int ps_id) {
        const SimParams& sp = sim_params[ps_id];

        Backtest::SignalView signal_view;
        signal_view.reserve(shared_signal_arrays.size() + sp.signal_array_bufs.size());
        for (const auto& [k, ptr] : shared_signal_arrays) signal_view[k] = ptr;
        for (const auto& [k, buf] : sp.signal_array_bufs)  signal_view[k] = buf.data();

        outputs[ps_id] = Backtest::run_simulation(
            header, fast_pool, signal_view, sp.signal_refs,
            n_bars, bar_dates, bar_times, bar_days,
            sp.params, exec_settings, ps_id
        );
    });

    EngineResult result;
    result.simulations.reserve(n_sims);

    auto t1 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < n_sims; ++i) {
        result.simulations.push_back(std::move(outputs[i].trades));
        for (const auto& dr : outputs[i].daily_vec)
            result.wfm_data.push_back(dr);  // cópia direta, sem json
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cerr << "   > [PERF] CPP-OP - parallel=" 
            << std::chrono::duration<double>(t1-t0).count() << "s | export="
            << std::chrono::duration<double>(t2-t1).count() << "s\n";

    return result;
}
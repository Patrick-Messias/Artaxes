#include "operation.h"
#include "backtest.h"
#include <execution>
#include <mutex>
#include <map>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iostream>

json Operation::run(const std::string& header, 
                    const std::map<std::string, std::vector<double>>& data,
                    const std::vector<std::string>& datetime,
                    const nlohmann::json& sim_params,
                    const nlohmann::json& exec_settings,
                    const nlohmann::json& indicators_pool,
                    const nlohmann::json& shared_signals) {

    std::vector<json> all_trades_results;
    std::vector<DailyResult> wfm_matrix;
    std::mutex mtx;

    std::vector<json> simulations;
    if (sim_params.is_array()) {
        for (const auto& s : sim_params) simulations.push_back(s);
    }

    std::vector<int> indexes(simulations.size());
    std::iota(indexes.begin(), indexes.end(), 0);

    std::for_each(std::execution::par, indexes.begin(), indexes.end(), [&](int ps_id) {
        const json& sim = simulations[ps_id];

        std::map<std::string, std::vector<double>> local_data = data;

        // 1. Indicators Pool — injeta apenas as colunas que esta simulation usa
        if (sim.contains("indicator_keys") && sim["indicator_keys"].is_array()) {
            for (const auto& pool_key_json : sim["indicator_keys"]) {
                std::string pk       = pool_key_json.get<std::string>();
                std::string col_name = pk.substr(pk.rfind("__") + 2);
                if (indicators_pool.contains(pk)) {
                    try { local_data[col_name] = indicators_pool[pk].get<std::vector<double>>(); }
                    catch (...) {}
                }
            }
        }

        // 2. Shared Signals
        if (shared_signals.is_object()) {
            for (const auto& [key, val] : shared_signals.items()) {
                try { local_data[key] = val.get<std::vector<double>>(); }
                catch (...) {}
            }
        }

        // 3. Exclusive Signal Data
        if (sim.contains("signal_data") && sim["signal_data"].is_object()) {
            for (const auto& [key, val] : sim["signal_data"].items()) {
                try { local_data[key] = val.get<std::vector<double>>(); }
                catch (...) {}
            }
        }

        // 4. Exclusive Indicator Data (legacy fallback, mantém compatibilidade)
        if (sim.contains("indicator_data") && sim["indicator_data"].is_object()) {
            for (const auto& [key, val] : sim["indicator_data"].items()) {
                try { local_data[key] = val.get<std::vector<double>>(); }
                catch (...) {}
            }
        }

        SimulationOutput sim_output = Backtest::run_simulation(
            header, local_data, datetime, sim, exec_settings, ps_id);

        std::lock_guard<std::mutex> lock(mtx);
        all_trades_results.push_back(sim_output.trades_json);
        wfm_matrix.insert(wfm_matrix.end(), sim_output.daily_vec.begin(), sim_output.daily_vec.end());
    });

    return {
        {"simulations", all_trades_results},
        {"wfm_data",    wfm_matrix}
    };
}
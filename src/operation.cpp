#include "operation.h"
#include "backtest.h"
#include <execution>
#include <mutex>
#include <map>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iostream>
#include <unordered_set>
#include <unordered_map>

json Operation::run(const std::string& header,
    const std::map<std::string, std::vector<double>>& data,
    const std::vector<std::string>& datetime,
    const nlohmann::json& sim_params,
    const nlohmann::json& exec_settings,
    const nlohmann::json& indicators_pool,
    const nlohmann::json& shared_signals
) {

    std::vector<json> all_trades_results;
    std::vector<DailyResult> wfm_matrix;
    std::mutex mtx;

    std::vector<json> simulations;
    if (sim_params.is_array()) {
        simulations.reserve(sim_params.size());
        for (const auto& s : sim_params) simulations.push_back(s);
    }

    // ||===================================================================|| Before parallel, convert json to SimView

    // Identifies which indicator_keys really are going to get used, avoids decoding entire pool
    std::unordered_set<std::string> required_indicator_keys;
    for (const auto& sim : simulations) {
        if (sim.contains("indicator_keys") && sim["indicator_keys"].is_array()) {
            for (const auto& k : sim["indicator_keys"]) {
                required_indicator_keys.insert(k.get<std::string>());
            }
        }
    }

    // Cache: pk -> vector<double> uses only necessary pk
    std::unordered_map<std::string, std::vector<double>> indicators_cache;
    indicators_cache.reserve(required_indicator_keys.size());

    for (const auto& pk : required_indicator_keys) {
        if (!indicators_pool.contains(pk)) continue;
        try { indicators_cache.emplace(pk, indicators_pool[pk].get<std::vector<double>>()); } 
        catch (...) {}
    }

    // Cache shared_signals, decodes only unce
    std::unordered_map<std::string, std::vector<double>> shared_cache;
    if (shared_signals.is_object()) {
        shared_cache.reserve(shared_signals.size());
        for (const auto& [key, val] : shared_signals.items()) {
            try { shared_cache.emplace(key, val.get<std::vector<double>>()); }
            catch (...) {}
        }
    }

    // Exclusive cache by sim: signal_data + indicator_data
    std::vector<std::unordered_map<std::string, std::vector<double>>> per_sim_cache(simulations.size());
    for (size_t ps_id = 0; ps_id < simulations.size(); ++ps_id) {
        const json& sim = simulations[ps_id];

        auto& out = per_sim_cache[ps_id];

        if (sim.contains("signal_data") && sim["signal_data"].is_object()) {
            for (const auto& [key, val] : sim["signal_data"].items()) {
                try { out[key] = val.get<std::vector<double>>(); }
                catch (...) {}
            }
        }

        if (sim.contains("indicator_data") && sim["indicator_data"].is_object()) {
            for (const auto& [key, val] : sim["indicator_data"].items()) {
                try { out[key] = val.get<std::vector<double>>(); }
                catch (...) {}
            }
        }
    }

    // ||===================================================================|| Parallel
    std::vector<int> indexes((int)simulations.size());
    std::iota(indexes.begin(), indexes.end(), 0);

    std::for_each(std::execution::par, indexes.begin(), indexes.end(), [&](int ps_id) {
        const json& sim = simulations[ps_id];

        Backtest::SimView sim_view;
        sim_view.reserve(64); // Reserve to reduce rehash

        // Indicators via indicator_key: col_name -> &vector<double>
        if (sim.contains("indicator_keys") && sim["indicator_keys"].is_array()) {
            for (const auto& pool_key_json : sim["indicator_keys"]) {
                std::string pk = pool_key_json.get<std::string>();
                auto it = indicators_cache.find(pk);
                if (it==indicators_cache.end()) continue;

                std::string col_name = pk.substr(pk.rfind("__") + 2);
                sim_view[col_name] = &it->second;
            }
        }

        // Shared Signals: key->&vector<double> 
        for (const auto& [key, vec] : shared_cache) {
            sim_view[key] = &vec;
        }

        // Exclusive data for this sim: key -> &vector<double> (do per_sim_cache)
        {
            auto& ex=per_sim_cache[ps_id];
            for (auto& [key, vec] : ex) {
                sim_view[key] = &vec;
            }
        }

        SimulationOutput sim_output = Backtest::run_simulation(
            header, data, sim_view, datetime, sim, exec_settings, ps_id
        );

        std::lock_guard<std::mutex> lock(mtx);
        all_trades_results.push_back(sim_output.trades_json);
        wfm_matrix.insert(wfm_matrix.end(), sim_output.daily_vec.begin(), sim_output.daily_vec.end());
    });

    return {
        {"simulations", all_trades_results},
        {"wfm_data",    wfm_matrix}
    };
}
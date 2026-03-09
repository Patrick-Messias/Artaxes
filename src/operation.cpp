#include "operation.h"
#include "backtest.h"
#include <execution>
#include <mutex>
#include <map>
#include <algorithm>
#include <vector>
#include <iostream>

json Operation::run(const std::string& header, 
                    const std::map<std::string, std::vector<double>>& data,
                    const std::vector<std::string>& datetime,
                    const nlohmann::json& sim_params,
                    const nlohmann::json& exec_settings,
                    const nlohmann::json& shared_inds
                    ) {

    // Lista para armazenar os resultados de todas as simulações
    std::vector<json> all_trades_results;
    std::vector<DailyResult> wfm_matrix;
    std::mutex mtx;

    // 1. Convertemos o nlohmann::json (sim_params) em um std::vector<json>
    //isso resolve o erro de "no instance of overloaded function std::for_each matches the argument list"
    //pois garante iteradores compatíveis com std::execution::par
    std::vector<json> simulations;
    if (sim_params.is_array()) {
        for (const auto& s : sim_params) simulations.push_back(s);
    }

    // 2. Processamento paralelo por simulação (Cada Param Set em uma Thread)
    std::vector<int> indexes(simulations.size());
    std::iota(indexes.begin(), indexes.end(), 0); // 0, 1, 2, ...

    std::for_each(std::execution::par, indexes.begin(), indexes.end(), [&](int ps_id) {
        // Recovers the simulation corresponding to this index
        const json& sim = simulations[ps_id];

        // Creates local copy of OHLC for this thread
        std::map<std::string, std::vector<double>> local_data = data; 

        // Injects shared indicators unce for all param_sets
        for (auto& [key, val] : shared_inds.items()) {
            local_data[key] = val.get<std::vector<double>>();
        }

        // Injects Signals
        if (sim.contains("signal_data") && sim["signal_data"].is_object()) {
            for (auto& [key, val] : sim["signal_data"].items()) {
                local_data[key] = val.get<std::vector<double>>();
            }
        }

        // Injects indicadots into local_data (if any)
        if (sim.contains("indicator_data") && sim["indicator_data"].is_object()) {
            for (auto& [key, val] : sim["indicator_data"].items()) {
                local_data[key] = val.get<std::vector<double>>();
            }
        }

        // Passes OHLC + Indicators to backtest engine
        SimulationOutput sim_output = Backtest::run_simulation(header, local_data, datetime, sim, exec_settings, ps_id);

        // Thread-Safe Consolidation
        std::lock_guard<std::mutex> lock(mtx);

        // Adds trades to JSON
        all_trades_results.push_back(sim_output.trades_json);

        wfm_matrix.insert(wfm_matrix.end(), sim_output.daily_vec.begin(), sim_output.daily_vec.end());
    });

    // Returns one big object
    json final_response = {
        {"simulations", all_trades_results},
        {"wfm_data", wfm_matrix}
    };
    return final_response;
}
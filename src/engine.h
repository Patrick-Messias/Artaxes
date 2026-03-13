#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Retorno C++ puro — convertido para py::dict em bindings.cpp
struct EngineResult {
    // [sim_idx] → list of trade dicts representados como nlohmann::json
    std::vector<nlohmann::json>  simulations;   // cada elemento = json array de trades
    std::vector<nlohmann::json>  wfm_data;      // cada elemento = {ts, pnl, id}
};

class Engine {
public:
    static EngineResult execute(
        const std::string&                                header,
        const std::unordered_map<std::string, const double*>& ohlc_arrays,
        size_t                                            n_bars,
        const std::vector<int64_t>&                       datetime_int,
        const std::unordered_map<std::string, const double*>& indicators_pool,
        const json&                                       sim_params,
        const json&                                       exec_settings
    );
};
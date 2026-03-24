#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <nlohmann/json.hpp>
#include "engine.h"
#include "backtest.h"

using json = nlohmann::json;

class Operation {
public:
    static EngineResult run(
        const std::string&                                      header,
        const std::unordered_map<std::string, const double*>&   fast_pool,
        size_t                                                  n_bars,
        const std::vector<int>&                                 bar_dates,
        const std::vector<int>&                                 bar_times,
        const std::vector<int>&                                 bar_days,
        const std::unordered_map<std::string, const uint8_t*>&  shared_signal_arrays,
        const std::vector<SimParams>&                           sim_params,
        const json&                                             exec_settings
    );
};
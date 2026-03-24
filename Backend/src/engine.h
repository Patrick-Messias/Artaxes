#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <nlohmann/json.hpp>
#include "Trade.h"
#include "Utils.h"

using json = nlohmann::json;

struct SimParams {
    std::string id;
    json        params;
    std::unordered_map<std::string, std::vector<uint8_t>> signal_array_bufs;
    std::unordered_map<std::string, std::string>          signal_refs;
};

struct EngineResult {
    std::vector<std::vector<Trade>> simulations;  // [sim_idx] → trades direto, sem json
    std::vector<DailyResult>               wfm_data;
};

class Engine {
public:
    static EngineResult execute(
        const std::string&                                      header,
        const std::unordered_map<std::string, const double*>&   ohlc_arrays,
        size_t                                                  n_bars,
        const std::vector<int64_t>&                             datetime_int,
        const std::unordered_map<std::string, const double*>&   indicators_pool,
        const std::unordered_map<std::string, const uint8_t*>&  shared_signal_arrays,
        const std::vector<SimParams>&                           sim_params,
        const json&                                             exec_settings
    );
};
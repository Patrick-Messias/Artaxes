#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <nlohmann/json.hpp>
#include "Trade.h"
#include "Utils.h"
#include "engine.h"   // SimParams

using json = nlohmann::json;

struct SimulationOutput {
    std::vector<Trade>       trades;
    std::vector<DailyResult> daily_vec;
};

class Backtest {
public:
    // fast_pool: ohlc + indicators + derived signal arrays (f64)
    // signal_view: entry/exit uint8 ponteiros (shared ou exclusivos)
    // signal_refs: {sig_name → col_name no fast_pool}
    using FastPool   = std::unordered_map<std::string, const double*>;
    using SignalView = std::unordered_map<std::string, const uint8_t*>;
    using SignalRefs = std::unordered_map<std::string, std::string>;

    static SimulationOutput run_simulation(
        const std::string&      header,
        const FastPool&         fast_pool,
        const SignalView&       signal_view,
        const SignalRefs&       signal_refs,
        size_t                  n_bars,
        const std::vector<int>& bar_dates,
        const std::vector<int>& bar_times,
        const std::vector<int>& bar_days,
        const json&             sim,
        const json&             exec_settings,
        int                     ps_id
    );
};
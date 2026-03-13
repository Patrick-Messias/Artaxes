#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "Trade.h"
#include "Utils.h"

// SimulationOutput usa apenas tipos C++ puros.
// A conversão para py::list/dict acontece em operation.cpp, não aqui.
struct SimulationOutput {
    std::vector<Trade>       trades;
    std::vector<DailyResult> daily_vec;
};

class Backtest {
public:
    using SimView = std::unordered_map<std::string, const double*>;

    static SimulationOutput run_simulation(
        const std::string&      header,
        const SimView&          sim_view,
        size_t                  n_bars,
        const std::vector<int>& bar_dates,
        const std::vector<int>& bar_times,
        const std::vector<int>& bar_days,
        const nlohmann::json&   sim,
        const nlohmann::json&   exec_settings,
        int                     ps_id
    );
};
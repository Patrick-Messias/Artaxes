#pragma once
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "Trade.h"

class Backtest {
public:
    static std::vector<Trade> run(const std::string& dataset_key,
                                  const nlohmann::json& dataset,
                                  const nlohmann::json& meta);
};

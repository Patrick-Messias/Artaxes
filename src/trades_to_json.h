#pragma once
#include <vector>
#include <nlohmann/json.hpp>
#include "Trade.h"

inline nlohmann::json trades_to_json(const std::vector<Trade>& trades) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& t : trades) {
        j.push_back({
            {"id", t.id},
            {"asset", t.asset},
            {"status", t.status},
            {"direction", t.direction},
            {"entry_price", t.entry_price},
            {"entry_datetime", t.entry_datetime},
            {"lot_size", t.lot_size},
            {"stop_loss", t.stop_loss ? nlohmann::json(*t.stop_loss) : nlohmann::json(nullptr)},
            {"take_profit", t.take_profit ? nlohmann::json(*t.take_profit) : nlohmann::json(nullptr)},
            {"exit_price", t.exit_price ? nlohmann::json(*t.exit_price) : nlohmann::json(nullptr)},
            {"exit_datetime", t.exit_datetime ? nlohmann::json(*t.exit_datetime) : nlohmann::json(nullptr)},
            {"exit_reason", t.exit_reason ? nlohmann::json(*t.exit_reason) : nlohmann::json(nullptr)},
            {"profit", t.profit ? nlohmann::json(*t.profit) : nlohmann::json(nullptr)},
            {"profit_r", t.profit_r ? nlohmann::json(*t.profit_r) : nlohmann::json(nullptr)}
        });
    }
    return j;
}

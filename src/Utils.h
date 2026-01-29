#pragma once
#include <vector>
#include <nlohmann/json.hpp>
#include "Trade.h"

using json = nlohmann::json;

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
            {"stop_loss", t.stop_loss ? json(*t.stop_loss) : json(nullptr)},
            {"take_profit", t.take_profit ? json(*t.take_profit) : json(nullptr)},
            {"exit_price", t.exit_price ? json(*t.exit_price) : json(nullptr)},
            {"exit_datetime", t.exit_datetime ? json(*t.exit_datetime) : json(nullptr)},
            {"exit_reason", t.exit_reason ? json(*t.exit_reason) : json(nullptr)},
            {"profit", t.profit ? json(*t.profit) : json(nullptr)},
            {"profit_r", t.profit_r ? json(*t.profit_r) : json(nullptr)}
        });
    }
    return j;
}

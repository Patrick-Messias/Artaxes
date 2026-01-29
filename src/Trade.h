#pragma once
#include <string>
#include <optional>

struct Trade {
    std::string id;
    std::string asset;
    std::string status;          // "open" | "closed"
    std::string direction;       // "long" | "short"
    double entry_price;
    std::string entry_datetime;
    double lot_size;
    std::optional<double> stop_loss;
    std::optional<double> take_profit;
    std::optional<double> exit_price;
    std::optional<std::string> exit_datetime;
    std::optional<std::string> exit_reason;
    std::optional<double> profit;     // %
    std::optional<double> profit_r;   // money
};

#pragma once
#include <string>
#include <optional>

// Depois remover entry_datetime, exit_datetime, já que esses são redundantes, talvez até max_fav/adv se performance ficar ruim

struct Trade {
    std::string id;
    std::string asset;
    std::string path;
    std::string status;          // "open" | "closed"
    std::string entry_datetime;
    std::optional<double> entry_price;
    std::optional<double> position_value; // entry_price * lot_size 
    std::optional<double> lot_size;
    std::optional<double> trail_stop;
    std::optional<double> comission;
    std::optional<double> pnl_gross;
    std::optional<double> stop_loss;
    std::optional<double> take_profit;

    std::optional<double> max_fav_price;
    std::optional<double> max_adv_price;

    std::optional<double> prev_day_price;

    std::optional<double> mfe;
    std::optional<double> mae;
    std::optional<double> exit_price;
    std::optional<std::string> exit_datetime;
    std::optional<std::string> exit_reason;
    std::optional<double> profit;     // %
    std::optional<double> profit_r;   // money
    std::optional<int> bars_held=0;
};

#pragma once
#include <string>
#include <optional>

struct Trade {
    std::string trade_id;
    std::string asset;
    std::string path;
    std::string status;
    char   entry_datetime[20] = {};   // "YYYYMMDD HHMMSS\0" — sem heap alloc
    double entry_price    = 0.0;
    double position_value = 0.0;
    double lot_size       = 0.0;
    double stop_loss      = 0.0;
    double take_profit    = 0.0;
    double max_fav_price  = 0.0;
    double max_adv_price  = 0.0;
    double daily_pnl_accum = 0.0; 
    //double prev_day_price = 0.0;
    double mfe            = 0.0; // Final Maximum Favorable Excursion
    double mae            = 0.0; // Final Maximum Adverse Excursion
    double exit_price     = 0.0;
    char   exit_datetime[20] = {};
    std::string exit_reason;
    double profit   = 0.0;
    double profit_r = 0.0;
    int    bars_held = 0;
    bool   closed   = false;
};
#include "Backtest.h"
#include "Trade.h"
#include "Utils.h"
#include <iostream>
#include <random>
#include <sstream>
#include <algorithm>
#include <iomanip>

using json = nlohmann::json;

double calculate_lot_size(double price_entry, bool is_long_trade);

static std::string generate_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::stringstream ss;
    ss << std::hex << rng();
    return ss.str();
}

// Used to get timeEI/EF/TF from exec_settings as numbers
double get_value_safe(const json& node, const json& params) {
    if (node.is_number()) return node.get<double>();
    if (node.is_object() && node.value("type", "") == "param") {
        std::string p_name = node.value("name", "");
        if (params.contains(p_name) && params[p_name].is_number())
            return params[p_name].get<double>();
    }
    return 0.0;
}

// Checks if a limit/stop order has immediatly been closed by SL or TP 
std::string check_limit_instant_exit(bool is_long, double limit, double sl, double tp,
                                     double open, double high, double low, double close) {
    bool bullish = (close > open);
    bool hit_tp = (is_long ? (tp > 0 && high >= tp) : (tp > 0 && low  <= tp));
    bool hit_sl = (is_long ? (sl > 0 && low  <= sl) : (sl > 0 && high >= sl));

    if (bullish) {
        if (limit > open) {
            if (is_long) return hit_tp ? "TP" : "";
            else         return hit_sl ? "SL" : "";
        } else {
            if (is_long) {
                if (hit_tp && hit_sl) return "SL";
                return hit_sl ? "SL" : (hit_tp ? "TP" : "");
            } else {
                if (hit_tp && hit_sl) return "TP";
                return hit_tp ? "TP" : (hit_sl ? "SL" : "");
            }
        }
    } else {
        if (limit < open) {
            if (is_long) return hit_sl ? "SL" : "";
            else         return hit_tp ? "TP" : "";
        } else {
            if (is_long) {
                if (hit_tp && hit_sl) return "TP";
                return hit_tp ? "TP" : (hit_sl ? "SL" : "");
            } else {
                if (hit_tp && hit_sl) return "SL";
                return hit_sl ? "SL" : (hit_tp ? "TP" : "");
            }
        }
    }
    return "";
}



SimulationOutput Backtest::run_simulation(const std::string& header,
                                          const std::map<std::string, std::vector<double>>& data,
                                          const std::vector<std::string>& datetime,
                                          const nlohmann::json& sim,
                                          const nlohmann::json& exec_settings,
                                          int ps_id) {

    std::vector<Trade> trades;
    std::vector<Trade> active_trades;
    std::vector<Trade> pending_orders;
    std::vector<DailyResult> daily_results_matrix;
    double temp_cumulative_pnl = 0.0;

    auto get_vec_ptr = [&](const std::string& key) -> const double* {
        auto it = data.find(key);
        if (it != data.end() && !it->second.empty()) return it->second.data();
        return nullptr;
    };

    try {
        json params = sim.value("params", json::object());

        // ── OHLC ──────────────────────────────────────────────────────────────
        const double* open  = get_vec_ptr("open");
        const double* high  = get_vec_ptr("high");
        const double* low   = get_vec_ptr("low");
        const double* close = get_vec_ptr("close");
        const size_t  n_bars = datetime.size();

        // ── SIGNAL COLUMNS (from Python) ───────────────────────────────────────
        // Bool signals — 1.0 = true, 0.0 = false
        const double* sig_entry_long  = get_vec_ptr("entry_long");
        const double* sig_entry_short = get_vec_ptr("entry_long");
        const double* exit_tf_long   = get_vec_ptr("exit_tf_long");
        const double* exit_tf_short  = get_vec_ptr("exit_tf_short");

        // SL/TP — distance from entry price (positive float, 0 = not set)
        // Python computes: atr * sl_perc, or any expression → column
        const double* exit_sl_long  = get_vec_ptr("exit_sl_long");
        const double* exit_sl_short = get_vec_ptr("exit_sl_short");
        const double* exit_tp_long  = get_vec_ptr("exit_tp_long");
        const double* exti_tp_short = get_vec_ptr("exti_tp_short");

        // Limit order — absolute target price (0 = not set → falls back to market)
        // Python computes: df['low'].shift(1), or any expression → column
        const double* limit_long  = get_vec_ptr("limit_long");
        const double* limit_short = get_vec_ptr("limit_short");

        // Break-even trigger — distance from entry that triggers moving SL to entry
        // be_trigger_long  > 0 → move SL when price rises   >= entry + dist
        // be_trigger_short > 0 → move SL when price drops   <= entry - dist
        const double* be_pos_val_long  = get_vec_ptr("be_pos_val_long");
        const double* be_pos_val_short = get_vec_ptr("be_pos_val_short");

        // W I P
        const double* be_neg_val_long  = get_vec_ptr("be_neg_val_long");
        const double* be_neg_val_short = get_vec_ptr("be_neg_val_short");

        // ── EXECUTION SETTINGS ────────────────────────────────────────────────
        int backtest_start_idx = params.value("backtest_start_idx", 1);
        int limit_order_expiry = params.value("limit_order_exclusion_after_period", 1);
        double limit_order_perc_treshold = params.value("limit_order_perc_treshold_for_order_diff", 1.0);
        bool limit_can_enter_at_market_if_gap    = params.value("limit_can_enter_at_market_if_gap", false);
        bool limit_opposite_order_closes_pending = params.value("limit_opposite_order_closes_pending", true);

        int nb_long  = params.value("exit_nb_long",  0);
        int nb_short = params.value("exit_nb_short", 0);
        int exit_nb_only_if_pnl_is = params.value("exit_nb_only_if_pnl_is", 0);

        auto strat_num_pos = exec_settings.value("strat_num_pos", json::array({1,1}));
        int max_long  = strat_num_pos[0].get<int>();
        int max_short = strat_num_pos[1].get<int>();

        auto strat_max_num_pos_per_day = exec_settings.value("strat_max_num_pos_per_day", json::array({1,1}));
        int max_long_trades_per_day  = strat_max_num_pos_per_day[0].get<int>();
        int max_short_trades_per_day = strat_max_num_pos_per_day[1].get<int>();
        if (max_long_trades_per_day  == -1) max_long_trades_per_day  = 999999;
        if (max_short_trades_per_day == -1) max_short_trades_per_day = 999999;
        int day_trades_long = 0, day_trades_short = 0;

        bool hedge_enabled = exec_settings.value("hedge", false);
        double offset      = exec_settings.value("offset", 0.0);
        bool is_daytrade   = exec_settings.value("day_trade", false);

        std::string order_type = exec_settings.at("order_type").get<std::string>();
        std::string limit_order_base_ref = exec_settings.at("limit_order_base_calc_ref_price").get<std::string>();

        size_t last_underscore = header.find_last_of('_');
        std::string asset_base_name = (last_underscore != std::string::npos) ? header.substr(last_underscore + 1) : header;
        std::string trade_path = header + "_" + sim["id"].get<std::string>();

        std::vector<int> close_days = exec_settings.value("day_of_week_close_and_stop_trade", std::vector<int>{});

        int timeEI = exec_settings.contains("timeEI") && !exec_settings["timeEI"].is_null()
            ? (int)get_value_safe(exec_settings["timeEI"], params) : 0;
        int timeEF = exec_settings.contains("timeEF") && !exec_settings["timeEF"].is_null()
            ? (int)get_value_safe(exec_settings["timeEF"], params) : 1440;
        int timeTF = exec_settings.contains("timeTF") && !exec_settings["timeTF"].is_null()
            ? (int)get_value_safe(exec_settings["timeTF"], params) : 1440;


























        bool print_logs = exec_settings.value("print_logs", false);
        int max_num_logs_per_backtest = 50;
        int num_logs_counter = backtest_start_idx;


















        

        // ── PRE-COMPUTE DATES ─────────────────────────────────────────────────
        std::vector<int> bar_dates(n_bars), bar_times(n_bars), bar_days(n_bars);
        for (size_t i = 0; i < n_bars; ++i) {
            const std::string& dt = datetime[i];
            bar_dates[i] = (dt[0]-'0')*10000000 + (dt[1]-'0')*1000000 + (dt[2]-'0')*100000 + (dt[3]-'0')*10000
                         + (dt[5]-'0')*1000      + (dt[6]-'0')*100
                         + (dt[8]-'0')*10        + (dt[9]-'0');
            bar_times[i] = ((dt[11]-'0')*10 + (dt[12]-'0')) * 10000
                         + ((dt[14]-'0')*10 + (dt[15]-'0')) * 100
                         + ((dt[17]-'0')*10 + (dt[18]-'0'));
            std::tm tm = {};
            std::istringstream ss(dt.substr(0, 10));
            ss >> std::get_time(&tm, "%Y-%m-%d");
            tm.tm_isdst = -1;
            std::mktime(&tm);
            bar_days[i] = tm.tm_wday;
        }


        

        // ── MAIN LOOP ─────────────────────────────────────────────────────────
        for (size_t i = backtest_start_idx; i < n_bars; ++i) {
            int  currentTime      = extract_minutes(datetime[i]);
            bool is_last_bar      = (i == n_bars - 1);
            bool day_switched     = (!is_last_bar && bar_dates[i+1] != bar_dates[i]);
            bool daytrade_time_final = (currentTime >= timeTF);
            if (day_switched) { day_trades_long = 0; day_trades_short = 0; }

            // Read signals for this bar
            bool signal_long  = sig_entry_long  && sig_entry_long[i]  > 0.5;
            bool signal_short = sig_entry_short && sig_entry_short[i] > 0.5;
            bool do_exit_long  = exit_tf_long  && exit_tf_long[i]   > 0.5;
            bool do_exit_short = exit_tf_short && exit_tf_short[i]  > 0.5;


            // ── 1. EXIT LOGIC ─────────────────────────────────────────────────
            // Exit at Open[i] based on signal from previous bar
            auto it = active_trades.begin();
            while (it != active_trades.end()) {
                bool is_long = (it->lot_size.value_or(0.0) > 0.0);
                bool closed = false;
                std::string reason = "";

                // TF — Timeframe/Signal exit
                if (is_long ? do_exit_long : do_exit_short) {
                    reason = "TF"; closed = true;
                }

                // NB — Number of bars exit
                if (!closed) {
                    int target_nb = is_long ? nb_long : nb_short;
                    if (target_nb > 0 && it->bars_held >= target_nb) {
                        bool can_exit = false;
                        if (exit_nb_only_if_pnl_is == 0) {
                            can_exit = true;
                        } else {
                            double pnl = (open[i] - *it->entry_price) / *it->entry_price * (is_long ? 1.0 : -1.0);
                            if (exit_nb_only_if_pnl_is > 0 && pnl > 0) can_exit = true;
                            else if (exit_nb_only_if_pnl_is < 0 && pnl < 0) can_exit = true;
                        }
                        if (can_exit) { reason = "NB"; closed = true; }
                    }
                }

                // DT — Daytrade forced close
                if (!closed && is_daytrade && (daytrade_time_final || day_switched || is_last_bar)) {
                    reason = "DT"; closed = true;
                }

                // WC — Weekday close
                if (!closed && !close_days.empty()) {
                    int today = bar_days[i];
                    if (std::find(close_days.begin(), close_days.end(), today) != close_days.end()) {
                        reason = "WC"; closed = true;
                    }
                }

                // EF — End of file
                if (!closed && is_last_bar) {
                    reason = "EF"; closed = true;
                }

                // BE — Break-even: move SL to entry when price reaches trigger distance
                if (!closed) {
                    double entry_p   = *it->entry_price;
                    double current_p = open[i];

                    const double* be_col = is_long ? be_pos_val_long : be_pos_val_short;
                    if (be_col && it->stop_loss.value_or(0.0) != entry_p) {
                        double be_dist = be_col[i];
                        if (be_dist > 0) {
                            bool triggered = is_long
                                ? (current_p >= entry_p + be_dist)
                                : (current_p <= entry_p - be_dist);
                            if (triggered) {
                                // Only move SL in the favorable direction
                                if (is_long ? (current_p > entry_p) : (current_p < entry_p))
                                    it->stop_loss = entry_p;
                            }
                        }
                    }
                }

                if (closed) {
                    it->exit_price    = open[i];
                    it->exit_datetime = datetime[i];
                    it->exit_reason   = reason;
                    it->status        = "closed";
                    it->mae = *it->max_adv_price;
                    it->mfe = *it->max_fav_price;

                    double entry = *it->entry_price;
                    double exit  = *it->exit_price;
                    double pnl   = ((exit - entry) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
                    it->profit = pnl;
                    temp_cumulative_pnl += pnl;

                    if (print_logs)
                        std::cout << "[EXIT  ] " << (is_long ? "L" : "S") << " | " << reason
                                  << " " << datetime[i] << " | In: " << entry << " | Out: " << exit
                                  << " | PnL: " << pnl << "% | Acc: " << temp_cumulative_pnl << "%" << std::endl;

                    double prev_p    = it->prev_day_price.value_or(entry);
                    double daily_var = ((exit - prev_p) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
                    daily_results_matrix.push_back({format_datetime_to_int(datetime[i]), daily_var, ps_id});

                    trades.push_back(*it);
                    it = active_trades.erase(it);
                } else { ++it; }
            }


            // ── 2. PENDING ORDERS ─────────────────────────────────────────────
            auto p_it = pending_orders.begin();
            while (p_it != pending_orders.end()) {
                bool is_long = (p_it->lot_size.value_or(0.0) > 0.0);
                double entry = *p_it->entry_price;
                bool triggered = false, expired = false;

                // Expiration
                if (p_it->bars_held.value_or(0) >= limit_order_expiry) expired = true;
                else if (((daytrade_time_final || day_switched) && is_daytrade) || is_last_bar) expired = true;

                if (expired) {
                    if (print_logs)
                        std::cout << "[LIM-EXPIRED] " << (is_long ? "L" : "S") << " | " << datetime[i]
                                  << " | Price: " << entry << " | Held: " << p_it->bars_held.value() << " bars" << std::endl;
                    p_it = pending_orders.erase(p_it);
                    continue;
                }

                // Execution check (pos_type stored temporarily in exit_reason)
                std::string pos_type = p_it->exit_reason.value_or("");
                if (is_long) {
                    if      (pos_type == "L_BELOW" && open[i] <= entry) { triggered = true; entry = open[i]; }
                    else if (pos_type == "L_BELOW" && low[i]  <  entry)   triggered = true;
                    else if (pos_type == "L_ABOVE" && open[i] >= entry) { triggered = true; entry = open[i]; }
                    else if (pos_type == "L_ABOVE" && high[i] >  entry)   triggered = true;
                } else {
                    if      (pos_type == "S_ABOVE" && open[i] >= entry) { triggered = true; entry = open[i]; }
                    else if (pos_type == "S_ABOVE" && high[i] >  entry)   triggered = true;
                    else if (pos_type == "S_BELOW" && open[i] <= entry) { triggered = true; entry = open[i]; }
                    else if (pos_type == "S_BELOW" && low[i]  <  entry)   triggered = true;
                }

                if (triggered) {
                    p_it->status = "open";
                    p_it->entry_datetime = datetime[i];
                    p_it->entry_price = entry;

                    // Recalculate SL/TP at execution bar using signal columns
                    double sl_dist = (is_long ? exit_sl_long  : exit_sl_short)  ? (is_long ? exit_sl_long[i]  : exit_sl_short[i])  : 0.0;
                    double tp_dist = (is_long ? exit_tp_long  : exti_tp_short)  ? (is_long ? exit_tp_long[i]  : exti_tp_short[i])  : 0.0;

                    if (sl_dist > 0) p_it->stop_loss   = is_long ? (entry - sl_dist) : (entry + sl_dist);
                    if (tp_dist > 0) p_it->take_profit = is_long ? (entry + tp_dist) : (entry - tp_dist);

                    double curr_sl = p_it->stop_loss.value_or(0.0);
                    double curr_tp = p_it->take_profit.value_or(0.0);
                    std::string instant_exit = check_limit_instant_exit(
                        is_long, entry, curr_sl, curr_tp, open[i], high[i], low[i], close[i]);

                    if (!instant_exit.empty()) {
                        double exit_px = (instant_exit == "TP") ? curr_tp : curr_sl;
                        p_it->exit_price    = exit_px;
                        p_it->exit_reason   = instant_exit;
                        p_it->exit_datetime = datetime[i];
                        p_it->status        = "closed";

                        double pnl = ((exit_px - entry) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
                        p_it->profit = pnl;
                        temp_cumulative_pnl += pnl;

                        if (print_logs)
                            std::cout << "[LIM-INSTANT] " << (is_long ? "L" : "S") << " | " << instant_exit
                                      << " " << datetime[i] << " | PnL: " << pnl << "%" << std::endl;

                        double daily_var = ((exit_px - entry) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
                        daily_results_matrix.push_back({format_datetime_to_int(datetime[i]), daily_var, ps_id});

                        trades.push_back(std::move(*p_it));
                        p_it = pending_orders.erase(p_it);
                    } else {
                        if (print_logs)
                            std::cout << "[LIM-EXEC] " << (is_long ? "L" : "S") << " | " << datetime[i]
                                      << " | Price: " << entry
                                      << " | SL: " << curr_sl << " | TP: " << curr_tp << std::endl;

                        daily_results_matrix.push_back({format_datetime_to_int(datetime[i]), 0.0, ps_id});
                        p_it->prev_day_price = entry;

                        active_trades.push_back(std::move(*p_it));
                        p_it = pending_orders.erase(p_it);
                    }
                    is_long ? ++day_trades_long : ++day_trades_short;
                } else {
                    p_it->bars_held = p_it->bars_held.value_or(0) + 1;
                    ++p_it;
                }
            }


            // ── 3. NEW ENTRY ──────────────────────────────────────────────────
            bool day_is_blocked = false;
            if (!close_days.empty()) {
                int today = bar_days[i];
                day_is_blocked = std::find(close_days.begin(), close_days.end(), today) != close_days.end();
            }
            
            if (!day_is_blocked && is_daytrade) {
                if (currentTime < timeEI || currentTime > timeEF) day_is_blocked = true;
            }

            if (!day_is_blocked && (signal_long || signal_short)) {
                int current_longs = 0, current_shorts = 0;
                for (const auto& t : active_trades) {
                    double lot = t.lot_size.value_or(0.0);
                    if (lot > 0) current_longs++; else if (lot < 0) current_shorts++;
                }
                for (const auto& p : pending_orders) {
                    double lot = p.lot_size.value_or(0.0);
                    if (lot > 0) current_longs++; else if (lot < 0) current_shorts++;
                }

                auto execute_entry_internal = [&](bool is_long) {
                    Trade t;
                    t.id     = generate_id();
                    t.asset  = asset_base_name;
                    t.path   = trade_path;

                    double target_price = 0.0;
                    bool gap_too_big = false, already_hit = false;

                    if (order_type == "market") {
                        // Market order — always enters at open[i]
                        target_price = open[i];
                    } else {
                        // Limit order — target price comes from signal column
                        // Python is responsible for computing the absolute price:
                        //   limit_long  = df['low'].shift(1)   (buy below yesterday's low)
                        //   limit_short = df['high'].shift(1)  (sell above yesterday's high)
                        //   or any other expression
                        const double* lim_col = is_long ? limit_long : limit_short;
                        if (lim_col && lim_col[i] > 0.0) {
                            target_price = lim_col[i];
                        } else {
                            // Fallback: no limit column provided → treat as market
                            target_price = open[i];
                        }

                        // Gap handling: is target realistically reachable from open?
                        if (is_long) {
                            std::string side = (target_price > open[i]) ? "L_ABOVE" : "L_BELOW";
                            t.exit_reason = side; // temp storage
                            double diff = (side == "L_ABOVE")
                                ? (target_price - open[i]) / open[i]
                                : (open[i] - target_price) / open[i];
                            if (limit_order_base_ref != "open" && open[i] <= target_price) already_hit = true;
                            if (diff > limit_order_perc_treshold) gap_too_big = true;
                        } else {
                            std::string side = (target_price > open[i]) ? "S_ABOVE" : "S_BELOW";
                            t.exit_reason = side;
                            double diff = (side == "S_BELOW")
                                ? (open[i] - target_price) / open[i]
                                : (target_price - open[i]) / open[i];
                            if (limit_order_base_ref != "open" && open[i] >= target_price) already_hit = true;
                            if (diff > limit_order_perc_treshold) gap_too_big = true;
                        }

                        if (gap_too_big) {
                            if (print_logs)
                                std::cout << "[SKIP-GAP] " << (is_long ? "L" : "S")
                                          << " | " << datetime[i] << " | Gap too big" << std::endl;
                            return;
                        }
                    }

                    bool execute_now = (order_type == "market" || (already_hit && limit_can_enter_at_market_if_gap));
                    double final_entry = execute_now ? open[i] : target_price;

                    t.entry_price    = final_entry;
                    t.status         = execute_now ? "open" : "pending";
                    t.entry_datetime = datetime[i];
                    t.bars_held      = 0;
                    t.lot_size       = calculate_lot_size(final_entry, is_long);
                    t.max_fav_price  = final_entry;
                    t.max_adv_price  = final_entry;

                    // SL/TP distances read directly from signal columns
                    double sl_dist = (is_long ? exit_sl_long  : exit_sl_short)  ? (is_long ? exit_sl_long[i]  : exit_sl_short[i])  : 0.0;
                    double tp_dist = (is_long ? exit_tp_long  : exti_tp_short)  ? (is_long ? exit_tp_long[i]  : exti_tp_short[i])  : 0.0;

                    double sl = 0.0, tp = 0.0;
                    if (sl_dist > 0) { sl = is_long ? (final_entry - sl_dist) : (final_entry + sl_dist); t.stop_loss   = sl; }
                    if (tp_dist > 0) { tp = is_long ? (final_entry + tp_dist) : (final_entry - tp_dist); t.take_profit = tp; }

                    if (execute_now) {
                        daily_results_matrix.push_back({format_datetime_to_int(datetime[i]), 0.0, ps_id});
                        t.prev_day_price = final_entry;
                        active_trades.push_back(std::move(t));
                        is_long ? ++day_trades_long : ++day_trades_short;

                        if (print_logs)
                            std::cout << "[EN-MKT] " << (is_long ? "L" : "S") << " | " << datetime[i]
                                      << " | Price: " << std::fixed << std::setprecision(5) << final_entry
                                      << " | SL: " << sl << " | TP: " << tp
                                      << " | Open pos: " << (is_long ? current_longs + 1 : current_shorts + 1) << std::endl;
                    } else {
                        pending_orders.push_back(std::move(t));

                        if (print_logs)
                            std::cout << "[EN-LIM] " << (is_long ? "L" : "S") << " | " << datetime[i]
                                      << " | Target: " << std::fixed << std::setprecision(5) << final_entry
                                      << " | SL: " << sl << " | TP: " << tp
                                      << " | Open pos: " << (is_long ? current_longs + 1 : current_shorts + 1) << std::endl;
                    }
                };

                // Cancel opposite pending orders if configured
                auto cancel_opposite = [&](bool is_long_signal) {
                    if (!limit_opposite_order_closes_pending) return;
                    auto oit = pending_orders.begin();
                    while (oit != pending_orders.end()) {
                        double lot = oit->lot_size.value_or(0.0);
                        bool is_opposite = is_long_signal ? (lot < 0) : (lot > 0);
                        if (is_opposite) {
                            if (print_logs)
                                std::cout << "[CANCEL-OPP] " << datetime[i]
                                          << " | Price: " << *oit->entry_price << std::endl;
                            oit = pending_orders.erase(oit);
                        } else { ++oit; }
                    }
                };

                if (signal_long) {
                    cancel_opposite(true);
                    bool hedge_ok = hedge_enabled || (current_shorts == 0);
                    if (current_longs < max_long && day_trades_long < max_long_trades_per_day && hedge_ok)
                        execute_entry_internal(true);
                }
                if (signal_short) {
                    cancel_opposite(false);
                    bool hedge_ok = hedge_enabled || (current_longs == 0);
                    if (current_shorts < max_short && day_trades_short < max_short_trades_per_day && hedge_ok)
                        execute_entry_internal(false);
                }
            }


            // ── 4. INTRA-CANDLE SL/TP ────────────────────────────────────────
            it = active_trades.begin();
            while (it != active_trades.end()) {
                double sl = it->stop_loss.value_or(0.0);
                double tp = it->take_profit.value_or(0.0);
                bool is_long = (it->lot_size.value_or(0.0) > 0.0);
                bool closed = false;
                std::string reason = "";

                if (is_long) {
                    if (sl > 0.0 && low[i]  <= sl) { it->exit_price = sl; reason = "SL"; closed = true; }
                    else if (tp > 0.0 && high[i] >= tp) { it->exit_price = tp; reason = "TP"; closed = true; }
                } else {
                    if (sl > 0.0 && high[i] >= sl) { it->exit_price = sl; reason = "SL"; closed = true; }
                    else if (tp > 0.0 && low[i]  <= tp) { it->exit_price = tp; reason = "TP"; closed = true; }
                }

                if (closed) {
                    double entry = *it->entry_price;
                    double exit  = *it->exit_price;
                    it->exit_datetime = datetime[i];
                    it->exit_reason   = reason;
                    it->status        = "closed";
                    it->mae = *it->max_adv_price;
                    it->mfe = *it->max_fav_price;

                    double pnl = ((exit - entry) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
                    it->profit = pnl;
                    temp_cumulative_pnl += pnl;

                    if (print_logs)
                        std::cout << "[SL/TP ] " << (is_long ? "L" : "S") << " | " << reason
                                  << " " << datetime[i] << " | PnL: " << pnl
                                  << "% | Acc: " << temp_cumulative_pnl << "%" << std::endl;

                    double prev_p    = it->prev_day_price.value_or(entry);
                    double daily_var = ((exit - prev_p) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
                    daily_results_matrix.push_back({format_datetime_to_int(datetime[i]), daily_var, ps_id});

                    trades.push_back(*it);
                    it = active_trades.erase(it);
                } else {
                    if (is_long) {
                        if (high[i] > *it->max_fav_price) it->max_fav_price = high[i];
                        if (low[i]  < *it->max_adv_price) it->max_adv_price = low[i];
                    } else {
                        if (low[i]  < *it->max_fav_price) it->max_fav_price = low[i];
                        if (high[i] > *it->max_adv_price) it->max_adv_price = high[i];
                    }
                    it->bars_held = it->bars_held.value_or(0) + 1;
                    ++it;
                }
            }


            // ── 5. DAILY PnL UPDATE ───────────────────────────────────────────
            if (day_switched || daytrade_time_final || is_last_bar) {
                for (auto& trade : active_trades) {
                    double entry    = *trade.entry_price;
                    double current_p = close[i];
                    bool is_long    = (trade.lot_size.value_or(0.0) > 0);

                    double prev_p    = trade.prev_day_price.value_or(entry);
                    double daily_var = ((current_p - prev_p) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
                    daily_results_matrix.push_back({format_datetime_to_int(datetime[i]), daily_var, ps_id});
                    trade.prev_day_price = current_p;
                }
            }
        

        }
    } catch (const std::exception& e) {
        std::cerr << "[C++ Backtest Data Init Error]: " << e.what() << std::endl;
    }

    if (exec_settings.value("print_logs", false))
        std::cout << "[FINISH] Total Trades: " << trades.size()
                  << " | PnL: " << temp_cumulative_pnl << "%" << std::endl;

    std::string trade_pnl_resolution = exec_settings.value("trade_pnl_resolution", "daily");
    SimulationOutput output;
    output.trades_json = trades_to_json(trades, trade_pnl_resolution);
    output.daily_vec   = daily_results_matrix;
    return output;
}

double calculate_lot_size(double price_entry, bool is_long_trade) {
    return is_long_trade ? 1.0 : -1.0;
}






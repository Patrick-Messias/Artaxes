#include "backtest.h"
#include <unordered_set>
#include "Trade.h"
#include "Utils.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <cstdio>

using json = nlohmann::json;

double calculate_lot_size(double, bool is_long) { return is_long ? 1.0 : -1.0; }

static std::string generate_id() {
    // thread_local: sem contenção entre threads paralelas, sem heap alloc
    thread_local std::mt19937_64 rng{std::random_device{}()};
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)rng());
    return std::string(buf, 16);
}

double get_value_safe(const json& node, const json& params) {
    if (node.is_number()) return node.get<double>();
    if (node.is_object() && node.value("type","") == "param") {
        std::string n = node.value("name","");
        if (params.contains(n) && params[n].is_number()) return params[n].get<double>();
    }
    return 0.0;
}

static std::string check_instant_exit(bool is_long, double,
                                      double sl, double tp,
                                      double o, double h, double l, double c) {
    bool bullish = (c >= o);
    bool hit_sl  = is_long ? (sl > 0 && l <= sl) : (sl > 0 && h >= sl);
    bool hit_tp  = is_long ? (tp > 0 && h >= tp) : (tp > 0 && l <= tp);
    if (!hit_sl && !hit_tp) return "";
    if ( hit_sl && !hit_tp) return "SL";
    if ( hit_tp && !hit_sl) return "TP";
    return bullish ? (is_long ? "SL" : "TP") : (is_long ? "TP" : "SL");
}

SimulationOutput Backtest::run_simulation(
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
) {
    std::vector<Trade>       trades;
    std::vector<Trade>       active_trades;
    std::vector<Trade>       pending_orders;
    std::vector<DailyResult> daily_results_matrix;
    double temp_cumulative_pnl = 0.0;

    trades.reserve(512);
    daily_results_matrix.reserve(n_bars);  // upper bound: 1 entry per bar

    // Preço: OHLC, indicators, derived signal arrays (f64 no fast_pool)
    auto get_price_ptr = [&](const std::string& key) -> const double* {
        auto it = fast_pool.find(key);
        return (it != fast_pool.end()) ? it->second : nullptr;
    };
    // Sinal de preço via ref: resolve o nome da coluna e busca no fast_pool
    auto get_ref_ptr = [&](const std::string& sig_name) -> const double* {
        auto rit = signal_refs.find(sig_name);
        if (rit == signal_refs.end()) return nullptr;
        return get_price_ptr(rit->second);
    };
    // Binário: entry/exit uint8 no signal_view
    auto get_signal_ptr = [&](const std::string& key) -> const uint8_t* {
        auto it = signal_view.find(key);
        return (it != signal_view.end()) ? it->second : nullptr;
    };

    // bar_times[i] = HHMMSS → minutos
    auto bar_minutes = [&](size_t i) -> int {
        int t = bar_times[i];
        return (t / 10000) * 60 + (t % 10000) / 100;
    };

    try {
        const json& params = sim;  // sim já é sp.params directamente

        const double* open  = get_price_ptr("open");
        const double* high  = get_price_ptr("high");
        const double* low   = get_price_ptr("low");
        const double* close = get_price_ptr("close");

        const uint8_t* sig_entry_long    = get_signal_ptr("entry_long");
        const uint8_t* sig_entry_short   = get_signal_ptr("entry_short");
        const uint8_t* sig_exit_long     = get_signal_ptr("exit_long");
        const uint8_t* sig_exit_short    = get_signal_ptr("exit_short");
        const double*  sig_limit_long    = get_ref_ptr("limit_long");
        const double*  sig_limit_short   = get_ref_ptr("limit_short");
        const double*  sig_sl_price_long = get_ref_ptr("sl_price_long");
        const double*  sig_sl_price_short= get_ref_ptr("sl_price_short");
        const double*  sig_sl_long       = get_ref_ptr("sl_long");
        const double*  sig_sl_short      = get_ref_ptr("sl_short");
        const double*  sig_tp_price_long = get_ref_ptr("tp_price_long");
        const double*  sig_tp_price_short= get_ref_ptr("tp_price_short");
        const double*  sig_tp_long       = get_ref_ptr("tp_long");
        const double*  sig_tp_short      = get_ref_ptr("tp_short");
        const double*  sig_trail_long    = get_ref_ptr("trail_long");
        const double*  sig_trail_short   = get_ref_ptr("trail_short");
        const double*  sig_be_long       = get_ref_ptr("be_trigger_long");
        const double*  sig_be_short      = get_ref_ptr("be_trigger_short");

        auto resolve_sl = [&](bool is_long, double fill, int idx) -> double {
            const double* p = is_long ? sig_sl_price_long : sig_sl_price_short;
            if (p && p[idx] > 0.0) return p[idx];
            const double* d = is_long ? sig_sl_long : sig_sl_short;
            if (d && d[idx] > 0.0) return is_long ? fill - d[idx] : fill + d[idx];
            return 0.0;
        };
        auto resolve_tp = [&](bool is_long, double fill, int idx) -> double {
            const double* p = is_long ? sig_tp_price_long : sig_tp_price_short;
            if (p && p[idx] > 0.0) return p[idx];
            const double* d = is_long ? sig_tp_long : sig_tp_short;
            if (d && d[idx] > 0.0) return is_long ? fill + d[idx] : fill - d[idx];
            return 0.0;
        };

        bool   print_logs          = exec_settings.value("print_logs", false);
        double slippage            = exec_settings.value("slippage",   0.0);
        double commission_rate     = exec_settings.value("commission", 0.0);
        int    backtest_start_idx  = params.value("backtest_start_idx", 1);
        int    limit_order_expiry  = params.value("limit_order_exclusion_after_period", 1);
        double limit_perc_treshold = params.value("limit_order_perc_treshold_for_order_diff", 1.0);
        bool   gap_market_fallback = params.value("limit_can_enter_at_market_if_gap", false);
        bool   cancel_opp_pending  = params.value("limit_opposite_order_closes_pending", true);
        int    nb_long             = params.value("exit_nb_long",  0);
        int    nb_short            = params.value("exit_nb_short", 0);
        int    exit_nb_only_if_pnl = params.value("exit_nb_only_if_pnl_is", 0);

        auto snp = exec_settings.value("strat_num_pos", json::array({1,1}));
        int max_long  = snp[0].get<int>(), max_short = snp[1].get<int>();
        auto mpd = exec_settings.value("strat_max_num_pos_per_day", json::array({1,1}));
        int max_long_pd  = mpd[0].get<int>(); if (max_long_pd  == -1) max_long_pd  = 999999;
        int max_short_pd = mpd[1].get<int>(); if (max_short_pd == -1) max_short_pd = 999999;
        int day_trades_long = 0, day_trades_short = 0;

        bool hedge_enabled = exec_settings.value("hedge",     false);
        bool is_daytrade   = exec_settings.value("day_trade", false);
        std::string order_type     = exec_settings.at("order_type").get<std::string>();
        std::string limit_base_ref = exec_settings.at("limit_order_base_calc_ref_price").get<std::string>();

        size_t lus = header.find_last_of('_');
        std::string asset_name = (lus != std::string::npos) ? header.substr(lus+1) : header;
        std::string trade_path = header + "_ps" + std::to_string(ps_id);

        auto _cd_vec = exec_settings.value("day_of_week_close_and_stop_trade", std::vector<int>{});
        std::unordered_set<int> close_days(_cd_vec.begin(), _cd_vec.end());

        int timeEI = exec_settings.contains("timeEI") && !exec_settings["timeEI"].is_null()
                     ? (int)get_value_safe(exec_settings["timeEI"], params) : 0;
        int timeEF = exec_settings.contains("timeEF") && !exec_settings["timeEF"].is_null()
                     ? (int)get_value_safe(exec_settings["timeEF"], params) : 1440;
        int timeTF = exec_settings.contains("timeTF") && !exec_settings["timeTF"].is_null()
                     ? (int)get_value_safe(exec_settings["timeTF"], params) : 1440;

        // ── Helpers ───────────────────────────────────────────────────────────
        auto apply_exit_slip = [&](double price, const std::string& reason, bool is_long) {
            if (reason == "EF") return price;
            return is_long ? price - slippage : price + slippage;
        };

        // Formata datetime string a partir dos inteiros pré-computados
        auto make_dt_str = [&](size_t i) -> std::string {
            char buf[20];
            std::snprintf(buf, sizeof(buf), "%08d %06d", bar_dates[i], bar_times[i]);
            return std::string(buf);
        };

        auto close_trade = [&](Trade& t, double raw_exit, const std::string& reason,
                               size_t bar_idx, bool is_long) {
            double entry      = t.entry_price;
            double exit_price = apply_exit_slip(raw_exit, reason, is_long);
            double net_pnl    = ((exit_price - entry) / entry) * 100.0
                                * (is_long ? 1.0 : -1.0) - commission_rate * 100.0;
            t.exit_price    = exit_price;
            { auto _s = make_dt_str(bar_idx); std::memcpy(t.exit_datetime, _s.c_str(), std::min(_s.size()+1, sizeof(t.exit_datetime))); }
            t.exit_reason   = reason;
            t.status        = "closed";
            t.closed        = true;
            t.mae           = t.max_adv_price;
            t.mfe           = t.max_fav_price;
            t.profit        = net_pnl;
            temp_cumulative_pnl += net_pnl;
            if (print_logs)
                std::cout << "[EXIT] " << (is_long?"L":"S") << " " << reason
                          << " | In: " << std::fixed << std::setprecision(5) << entry
                          << " Out: " << exit_price
                          << " Net: " << std::setprecision(4) << net_pnl << "%" << std::endl;
            double prev_p = t.prev_day_price;
            double dv = ((exit_price - prev_p) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
            daily_results_matrix.push_back({format_datetime_to_int_from_parts(bar_dates[bar_idx], bar_times[bar_idx]), dv, ps_id});
        };

        auto open_trade = [&](Trade& t, double raw_fill, size_t idx, bool is_long) {
            double fill = is_long ? raw_fill + slippage : raw_fill - slippage;
            t.entry_price    = fill;
            t.status         = "open";
            auto _s = make_dt_str(idx);
            std::memcpy(t.entry_datetime, _s.c_str(), std::min(_s.size()+1, sizeof(t.entry_datetime)));
            t.bars_held      = 0;
            t.lot_size       = calculate_lot_size(fill, is_long);
            t.max_fav_price  = fill;
            t.max_adv_price  = fill;
            t.prev_day_price = fill;
            double sl = resolve_sl(is_long, fill, (int)idx);
            double tp = resolve_tp(is_long, fill, (int)idx);
            if (sl > 0.0) t.stop_loss   = sl;
            if (tp > 0.0) t.take_profit = tp;
            const double* trail = is_long ? sig_trail_long : sig_trail_short;
            if (trail && trail[idx] > 0.0) {
                double tsl = is_long ? fill - trail[idx] : fill + trail[idx];
                double csl = t.stop_loss;
                if (is_long ? (tsl > csl) : (tsl < csl || csl == 0.0)) t.stop_loss = tsl;
            }
        };

        auto update_trailing = [&](Trade& t, bool is_long, double ref, size_t idx) {
            const double* trail = is_long ? sig_trail_long : sig_trail_short;
            if (!trail || trail[idx] <= 0.0) return;
            double new_sl = is_long ? ref - trail[idx] : ref + trail[idx];
            double cur_sl = is_long ? t.stop_loss
                                    : (t.stop_loss > 0.0 ? t.stop_loss : std::numeric_limits<double>::max());
            if (is_long ? (new_sl > cur_sl) : (new_sl < cur_sl)) t.stop_loss = new_sl;
        };

        // ═════════════════════════════════════════════════════════════════════
        // MAIN LOOP
        // ═════════════════════════════════════════════════════════════════════

        for (size_t i = (size_t)backtest_start_idx; i < n_bars; ++i) {
            int  currentTime  = bar_minutes(i);
            bool is_last_bar  = (i == n_bars - 1);
            bool day_switched = (!is_last_bar && bar_dates[i+1] != bar_dates[i]);
            bool dt_final     = (currentTime >= timeTF);
            if (day_switched) { day_trades_long = 0; day_trades_short = 0; }

            bool signal_long   = sig_entry_long  && sig_entry_long[i]  != 0;
            bool signal_short  = sig_entry_short && sig_entry_short[i] != 0;
            bool do_exit_long  = sig_exit_long   && sig_exit_long[i]   != 0;
            bool do_exit_short = sig_exit_short  && sig_exit_short[i]  != 0;

            // ── 1. EXIT LOGIC ─────────────────────────────────────────────────
            {
                auto it = active_trades.begin();
                while (it != active_trades.end()) {
                    bool is_long = (it->lot_size > 0.0);
                    std::string reason;
                    if      (is_long ? do_exit_long : do_exit_short)               reason = "TF";
                    else if ([&]() {
                        int nb = is_long ? nb_long : nb_short;
                        if (nb <= 0 || it->bars_held < nb) return false;
                        if (exit_nb_only_if_pnl == 0) return true;
                        double p = (open[i] - it->entry_price) / it->entry_price * (is_long?1:-1);
                        return (exit_nb_only_if_pnl > 0) ? p > 0 : p < 0;
                    }())                                                            reason = "NB";
                    else if (is_daytrade && (dt_final||day_switched||is_last_bar)) reason = "DT";
                    else if (!close_days.empty() &&
                             close_days.count(bar_days[i]))           reason = "WC";
                    else if (is_last_bar)                                          reason = "EF";

                    // BE trigger
                    const double* be = is_long ? sig_be_long : sig_be_short;
                    if (be && be[i] > 0.0) {
                        double ep = it->entry_price;
                        if (is_long ? (open[i] >= ep + be[i]) : (open[i] <= ep - be[i])) {
                            double csl = it->stop_loss;
                            if (is_long ? (csl < ep) : (csl > ep || csl == 0.0))
                                it->stop_loss = ep;
                        }
                    }
                    update_trailing(*it, is_long, open[i], i);

                    if (!reason.empty()) {
                        close_trade(*it, open[i], reason, i, is_long);
                        trades.push_back(*it);
                        it = active_trades.erase(it);
                    } else { ++it; }
                }
            }

            // ── 2. PENDING ORDERS ─────────────────────────────────────────────
            {
                auto p_it = pending_orders.begin();
                while (p_it != pending_orders.end()) {
                    bool   is_long = (p_it->lot_size > 0.0);
                    double target  = p_it->entry_price;
                    bool triggered = false;
                    double fill    = target;

                    bool expired =
                        (p_it->bars_held >= limit_order_expiry) ||
                        (((dt_final||day_switched) && is_daytrade) || is_last_bar);
                    if (expired) { p_it = pending_orders.erase(p_it); continue; }

                    std::string pos_type = p_it->exit_reason;
                    if (is_long) {
                        if      (pos_type=="L_BELOW" && open[i] <= target) { triggered=true; fill=open[i]; }
                        else if (pos_type=="L_BELOW" && low[i]  <  target)   triggered=true;
                        else if (pos_type=="L_ABOVE" && open[i] >= target) { triggered=true; fill=open[i]; }
                        else if (pos_type=="L_ABOVE" && high[i] >= target)   triggered=true;
                    } else {
                        if      (pos_type=="S_ABOVE" && open[i] >= target) { triggered=true; fill=open[i]; }
                        else if (pos_type=="S_ABOVE" && high[i] >= target)   triggered=true;
                        else if (pos_type=="S_BELOW" && open[i] <= target) { triggered=true; fill=open[i]; }
                        else if (pos_type=="S_BELOW" && low[i]  <  target)   triggered=true;
                    }

                    if (triggered) {
                        open_trade(*p_it, fill, i, is_long);
                        double sl = p_it->stop_loss;
                        double tp = p_it->take_profit;
                        std::string inst = check_instant_exit(is_long, p_it->entry_price,
                                                              sl, tp, open[i], high[i], low[i], close[i]);
                        if (!inst.empty()) {
                            close_trade(*p_it, (inst=="TP"?tp:sl), inst, i, is_long);
                            trades.push_back(std::move(*p_it));
                        } else {
                            daily_results_matrix.push_back({format_datetime_to_int_from_parts(bar_dates[i], bar_times[i]), 0.0, ps_id});
                            active_trades.push_back(std::move(*p_it));
                        }
                        p_it = pending_orders.erase(p_it);
                        is_long ? ++day_trades_long : ++day_trades_short;
                    } else {
                        p_it->bars_held = p_it->bars_held + 1;
                        ++p_it;
                    }
                }
            }

            // ── 3. NEW ENTRIES ────────────────────────────────────────────────
            {
                bool day_blocked = (!close_days.empty() &&
                    close_days.count(bar_days[i]));
                if (!day_blocked && is_daytrade)
                    day_blocked = (currentTime < timeEI || currentTime > timeEF);

                if (!day_blocked && (signal_long || signal_short)) {
                    int cur_longs = 0, cur_shorts = 0;
                    for (const auto& t : active_trades)  { if(t.lot_size>0)++cur_longs; else if(t.lot_size<0)++cur_shorts; }
                    for (const auto& p : pending_orders) { if(p.lot_size>0)++cur_longs; else if(p.lot_size<0)++cur_shorts; }

                    auto execute_entry = [&](bool is_long) {
                        if (order_type == "market") {
                            Trade t; t.id=generate_id(); t.asset=asset_name; t.path=trade_path;
                            open_trade(t, open[i], i, is_long);
                            double sl=t.stop_loss, tp=t.take_profit;
                            std::string inst = check_instant_exit(is_long,t.entry_price,sl,tp,open[i],high[i],low[i],close[i]);
                            if (!inst.empty()) {
                                close_trade(t,(inst=="TP"?tp:sl),inst,i,is_long);
                                trades.push_back(std::move(t));
                            } else {
                                daily_results_matrix.push_back({format_datetime_to_int_from_parts(bar_dates[i],bar_times[i]),0.0,ps_id});
                                active_trades.push_back(std::move(t));
                            }
                            is_long ? ++day_trades_long : ++day_trades_short;
                        } else {
                            const double* lim = is_long ? sig_limit_long : sig_limit_short;
                            if (!lim || lim[i] <= 0.0) {
                                Trade t; t.id=generate_id(); t.asset=asset_name; t.path=trade_path;
                                open_trade(t,open[i],i,is_long);
                                daily_results_matrix.push_back({format_datetime_to_int_from_parts(bar_dates[i],bar_times[i]),0.0,ps_id});
                                active_trades.push_back(std::move(t));
                                is_long ? ++day_trades_long : ++day_trades_short;
                                return;
                            }
                            double target = lim[i];
                            std::string pos_type;
                            double diff = std::abs(open[i]-target)/open[i];
                            if (is_long) {
                                pos_type = (target <= open[i]) ? "L_BELOW" : "L_ABOVE";
                            } else {
                                pos_type = (target >= open[i]) ? "S_ABOVE" : "S_BELOW";
                            }
                            if (diff > limit_perc_treshold) return;
                            bool already_hit = (is_long && target<=open[i]) || (!is_long && target>=open[i]);
                            if (already_hit && gap_market_fallback) {
                                Trade t; t.id=generate_id(); t.asset=asset_name; t.path=trade_path;
                                open_trade(t,open[i],i,is_long);
                                daily_results_matrix.push_back({format_datetime_to_int_from_parts(bar_dates[i],bar_times[i]),0.0,ps_id});
                                active_trades.push_back(std::move(t));
                                is_long ? ++day_trades_long : ++day_trades_short;
                                return;
                            }
                            Trade t; t.id=generate_id(); t.asset=asset_name; t.path=trade_path;
                            t.entry_price=target; t.status="pending"; { auto _s=make_dt_str(i); std::memcpy(t.entry_datetime,_s.c_str(),std::min(_s.size()+1,sizeof(t.entry_datetime))); }
                            t.bars_held=0; t.lot_size=is_long?1.0:-1.0; t.exit_reason=pos_type;
                            pending_orders.push_back(std::move(t));
                        }
                    };

                    auto cancel_opposite = [&](bool long_sig) {
                        if (!cancel_opp_pending) return;
                        auto oit = pending_orders.begin();
                        while (oit != pending_orders.end()) {
                            double lot = oit->lot_size;
                            oit = (long_sig?(lot<0):(lot>0)) ? pending_orders.erase(oit) : ++oit;
                        }
                    };

                    if (signal_long  && cur_longs  < max_long  && day_trades_long  < max_long_pd  && (hedge_enabled||cur_shorts==0))
                        { cancel_opposite(true);  execute_entry(true);  }
                    if (signal_short && cur_shorts < max_short && day_trades_short < max_short_pd && (hedge_enabled||cur_longs==0))
                        { cancel_opposite(false); execute_entry(false); }
                }
            }

            // ── 4. INTRA-CANDLE SL/TP + TRAILING ─────────────────────────────
            {
                auto it = active_trades.begin();
                while (it != active_trades.end()) {
                    bool is_long = (it->lot_size > 0.0);
                    update_trailing(*it, is_long, is_long ? high[i] : low[i], i);
                    double sl=it->stop_loss, tp=it->take_profit;
                    bool hit_sl = is_long?(sl>0&&low[i]<=sl):(sl>0&&high[i]>=sl);
                    bool hit_tp = is_long?(tp>0&&high[i]>=tp):(tp>0&&low[i]<=tp);
                    if (hit_sl || hit_tp) {
                        std::string reason; double exit_px;
                        if (hit_sl && hit_tp) {
                            reason = check_instant_exit(is_long,it->entry_price,sl,tp,open[i],high[i],low[i],close[i]);
                            if (reason.empty()) reason="SL";
                            exit_px = (reason=="TP")?tp:sl;
                        } else { reason=hit_sl?"SL":"TP"; exit_px=hit_sl?sl:tp; }
                        close_trade(*it, exit_px, reason, i, is_long);
                        trades.push_back(*it);
                        it = active_trades.erase(it);
                    } else {
                        if (is_long) {
                            if (high[i]>it->max_fav_price) it->max_fav_price=high[i];
                            if (low[i] <it->max_adv_price) it->max_adv_price=low[i];
                        } else {
                            if (low[i] <it->max_fav_price) it->max_fav_price=low[i];
                            if (high[i]>it->max_adv_price) it->max_adv_price=high[i];
                        }
                        it->bars_held = it->bars_held+1;
                        ++it;
                    }
                }
            }

            // ── 5. DAILY PnL UPDATE ───────────────────────────────────────────
            if (day_switched || dt_final || is_last_bar) {
                for (auto& trade : active_trades) {
                    double entry  = trade.entry_price, curr_p = close[i];
                    bool is_long  = (trade.lot_size > 0);
                    double prev_p = trade.prev_day_price;
                    double dv = ((curr_p-prev_p)/entry)*100.0*(is_long?1.0:-1.0);
                    daily_results_matrix.push_back({format_datetime_to_int_from_parts(bar_dates[i],bar_times[i]),dv,ps_id});
                    trade.prev_day_price = curr_p;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Backtest Error]: " << e.what() << std::endl;
    }

    SimulationOutput output;
    output.trades   = std::move(trades);
    output.daily_vec = std::move(daily_results_matrix);
    return output;
}
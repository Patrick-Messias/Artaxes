#include "Backtest.h"
#include "Trade.h"
#include "Utils.h"
#include <iostream>
#include <random>
#include <sstream>
#include <algorithm>
#include <limits>
#include <iomanip>

using json = nlohmann::json;

double calculate_lot_size(double price_entry, bool is_long_trade);

static std::string generate_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::stringstream ss;
    ss << std::hex << rng();
    return ss.str();
}

// Mantido apenas para timeEI/EF/TF vindos de exec_settings
double get_value_safe(const json& node, const json& params) {
    if (node.is_number()) return node.get<double>();
    if (node.is_object() && node.value("type", "") == "param") {
        std::string p_name = node.value("name", "");
        if (params.contains(p_name) && params[p_name].is_number())
            return params[p_name].get<double>();
    }
    return 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// check_instant_exit
//
// Dado um trade aberto neste bar com preço `fill`, verifica se SL ou TP
// foram atingidos no mesmo bar e qual foi primeiro.
//
// Usa a direção do candle para inferir o caminho de preço:
//   Bullish → O → L → H → C   (a mínima é atingida antes da máxima)
//   Bearish → O → H → L → C   (a máxima é atingida antes da mínima)
//
// Equivalente ao comportamento do MT5 em "bar-based backtesting".
// ─────────────────────────────────────────────────────────────────────────────
static std::string check_instant_exit(bool is_long, double fill,
                                      double sl, double tp,
                                      double o, double h, double l, double c) {
    bool bullish = (c >= o); // doji = bullish (conservador)
    bool hit_sl = is_long ? (sl > 0 && l <= sl) : (sl > 0 && h >= sl);
    bool hit_tp = is_long ? (tp > 0 && h >= tp) : (tp > 0 && l <= tp);

    if (!hit_sl && !hit_tp) return "";
    if (hit_sl && !hit_tp)  return "SL";
    if (hit_tp && !hit_sl)  return "TP";

    // Ambos atingidos — desempata pela direção do candle
    if (bullish) {
        // Path O→L→H: para long, SL (embaixo) é verificado antes do TP (acima)
        return is_long ? "SL" : "TP";
    } else {
        // Path O→H→L: para long, TP (acima) é verificado antes do SL (embaixo)
        return is_long ? "TP" : "SL";
    }
}


SimulationOutput Backtest::run_simulation(const std::string& header,
                                          const std::map<std::string, std::vector<double>>& base_data,
                                          const std::map<std::string, std::vector<double>>& sim_data,
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
        auto it_sim = sim_data.find(key);
        if (it_sim != sim_data.end() && !it_sim->second.empty()) return it_sim->second.data();
        auto it_base = base_data.find(key);
        if (it_base != base_data.end() && !it_base->second.empty()) return it_base->second.data();
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

        // ── SIGNAL COLUMNS ────────────────────────────────────────────────────

        // Entrada/saída (bool: > 0.5 = true)
        const double* sig_entry_long  = get_vec_ptr("entry_long");
        const double* sig_entry_short = get_vec_ptr("entry_short");
        const double* sig_exit_long   = get_vec_ptr("exit_long");
        const double* sig_exit_short  = get_vec_ptr("exit_short");

        // Preço absoluto da ordem pendente
        // C++ infere o tipo (LIMIT vs STOP) pela posição relativa ao open:
        //   Long:  price < open → L_BELOW (Buy Limit)   price > open → L_ABOVE (Buy Stop)
        //   Short: price > open → S_ABOVE (Sell Limit)  price < open → S_BELOW (Sell Stop)
        const double* sig_limit_long  = get_vec_ptr("limit_long");
        const double* sig_limit_short = get_vec_ptr("limit_short");

        // SL — PRIORIDADE 1: preço absoluto (swing low/high, nível técnico)
        //      PRIORIDADE 2: distância (entry ± dist, ex: ATR * mult)
        // Diferença crítica em limit orders com gap:
        //   sl_price → mantém o nível exato independente do fill
        //   sl_dist  → recalcula a partir do fill real
        const double* sig_sl_price_long  = get_vec_ptr("sl_price_long");
        const double* sig_sl_price_short = get_vec_ptr("sl_price_short");
        const double* sig_sl_long        = get_vec_ptr("sl_long");
        const double* sig_sl_short       = get_vec_ptr("sl_short");

        // TP — mesma hierarquia
        const double* sig_tp_price_long  = get_vec_ptr("tp_price_long");
        const double* sig_tp_price_short = get_vec_ptr("tp_price_short");
        const double* sig_tp_long        = get_vec_ptr("tp_long");
        const double* sig_tp_short       = get_vec_ptr("tp_short");

        // Trailing stop — distância a partir da máxima/mínima favorável
        // trail_long[i]  → SL sobe para max(SL, high[i]  - trail_dist)
        // trail_short[i] → SL desce para min(SL, low[i]  + trail_dist)
        const double* sig_trail_long  = get_vec_ptr("trail_long");
        const double* sig_trail_short = get_vec_ptr("trail_short");

        // Break-even — distância que ativa movimento do SL para entry
        const double* sig_be_trigger_long  = get_vec_ptr("be_trigger_long");
        const double* sig_be_trigger_short = get_vec_ptr("be_trigger_short");

        // ── RESOLVERS SL/TP ───────────────────────────────────────────────────
        // Prioridade: preço absoluto > distância > sem nível (0)
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

        // ── EXECUTION SETTINGS ────────────────────────────────────────────────
        bool   print_logs       = exec_settings.value("print_logs", false);
        double slippage         = exec_settings.value("slippage",   0.0);   // preço unitário
        double commission_rate  = exec_settings.value("commission", 0.0);   // fração do entry

        int    backtest_start_idx  = params.value("backtest_start_idx", 1);
        int    limit_order_expiry  = params.value("limit_order_exclusion_after_period", 1);
        double limit_perc_treshold = params.value("limit_order_perc_treshold_for_order_diff", 1.0);
        bool   gap_market_fallback = params.value("limit_can_enter_at_market_if_gap", false);
        bool   cancel_opp_pending  = params.value("limit_opposite_order_closes_pending", true);

        int nb_long             = params.value("exit_nb_long",  0);
        int nb_short            = params.value("exit_nb_short", 0);
        int exit_nb_only_if_pnl = params.value("exit_nb_only_if_pnl_is", 0);

        auto strat_num_pos = exec_settings.value("strat_num_pos", json::array({1,1}));
        int max_long  = strat_num_pos[0].get<int>();
        int max_short = strat_num_pos[1].get<int>();

        auto max_per_day = exec_settings.value("strat_max_num_pos_per_day", json::array({1,1}));
        int max_long_pd  = max_per_day[0].get<int>(); if (max_long_pd  == -1) max_long_pd  = 999999;
        int max_short_pd = max_per_day[1].get<int>(); if (max_short_pd == -1) max_short_pd = 999999;
        int day_trades_long = 0, day_trades_short = 0;

        bool hedge_enabled = exec_settings.value("hedge", false);
        bool is_daytrade   = exec_settings.value("day_trade", false);

        std::string order_type       = exec_settings.at("order_type").get<std::string>();
        std::string limit_base_ref   = exec_settings.at("limit_order_base_calc_ref_price").get<std::string>();

        size_t last_us = header.find_last_of('_');
        std::string asset_name = (last_us != std::string::npos) ? header.substr(last_us + 1) : header;
        std::string trade_path = header + "_" + sim["id"].get<std::string>();

        std::vector<int> close_days = exec_settings.value("day_of_week_close_and_stop_trade", std::vector<int>{});

        int timeEI = exec_settings.contains("timeEI") && !exec_settings["timeEI"].is_null()
            ? (int)get_value_safe(exec_settings["timeEI"], params) : 0;
        int timeEF = exec_settings.contains("timeEF") && !exec_settings["timeEF"].is_null()
            ? (int)get_value_safe(exec_settings["timeEF"], params) : 1440;
        int timeTF = exec_settings.contains("timeTF") && !exec_settings["timeTF"].is_null()
            ? (int)get_value_safe(exec_settings["timeTF"], params) : 1440;

        // ── PRE-COMPUTE DATES ─────────────────────────────────────────────────
        std::vector<int> bar_dates(n_bars), bar_times(n_bars), bar_days(n_bars);
        for (size_t i = 0; i < n_bars; ++i) {
            const std::string& dt = datetime[i];
            bar_dates[i] = (dt[0]-'0')*10000000 + (dt[1]-'0')*1000000 + (dt[2]-'0')*100000 + (dt[3]-'0')*10000
                         + (dt[5]-'0')*1000      + (dt[6]-'0')*100
                         + (dt[8]-'0')*10        + (dt[9]-'0');
            bar_times[i] = ((dt[11]-'0')*10+(dt[12]-'0'))*10000
                         + ((dt[14]-'0')*10+(dt[15]-'0'))*100
                         + ((dt[17]-'0')*10+(dt[18]-'0'));
            std::tm tm = {};
            std::istringstream ss(dt.substr(0, 10));
            ss >> std::get_time(&tm, "%Y-%m-%d");
            tm.tm_isdst = -1; std::mktime(&tm);
            bar_days[i] = tm.tm_wday;
        }

        // ── HELPER: aplica slippage adverso por tipo de saída ─────────────────
        auto apply_exit_slip = [&](double price, const std::string& reason, bool is_long) -> double {
            if (reason == "EF") return price;          // fim do dado: sem slippage
            if (reason == "TF" || reason == "NB" ||
                reason == "DT" || reason == "WC")      // saídas a mercado
                return is_long ? price - slippage : price + slippage;
            if (reason == "SL")
                return is_long ? price - slippage : price + slippage;
            if (reason == "TP")
                return is_long ? price - slippage : price + slippage;
            return price;
        };

        // ── HELPER: close_trade (centraliza lógica de fechamento) ─────────────
        auto close_trade = [&](Trade& t, double raw_exit, const std::string& reason,
                               const std::string& dt_str, bool is_long) {
            double entry      = *t.entry_price;
            double exit_price = apply_exit_slip(raw_exit, reason, is_long);

            double gross_pnl = ((exit_price - entry) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
            double comm_pct  = commission_rate * 100.0;          // round-trip em %
            double net_pnl   = gross_pnl - comm_pct;

            t.exit_price    = exit_price;
            t.exit_datetime = dt_str;
            t.exit_reason   = reason;
            t.status        = "closed";
            t.mae           = *t.max_adv_price;
            t.mfe           = *t.max_fav_price;
            t.profit        = net_pnl;
            temp_cumulative_pnl += net_pnl;

            if (print_logs)
                std::cout << "[EXIT] " << (is_long ? "L" : "S") << " | " << reason
                          << " " << dt_str
                          << " | In: " << std::fixed << std::setprecision(5) << entry
                          << " | Out: " << exit_price
                          << " | Net: " << std::setprecision(4) << net_pnl
                          << "% | Acc: " << temp_cumulative_pnl << "%" << std::endl;

            double prev_p    = t.prev_day_price.value_or(entry);
            double daily_var = ((exit_price - prev_p) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
            daily_results_matrix.push_back({format_datetime_to_int(dt_str), daily_var, ps_id});
        };

        // ── HELPER: open_trade (fill + SL/TP + trailing init) ─────────────────
        // raw_fill: preço antes do slippage (open ou target)
        auto open_trade = [&](Trade& t, double raw_fill, int idx, bool is_long) {
            double fill = is_long ? raw_fill + slippage : raw_fill - slippage;

            t.entry_price    = fill;
            t.status         = "open";
            t.entry_datetime = datetime[idx];
            t.bars_held      = 0;
            t.lot_size       = calculate_lot_size(fill, is_long);
            t.max_fav_price  = fill;
            t.max_adv_price  = fill;
            t.prev_day_price = fill;

            double sl = resolve_sl(is_long, fill, idx);
            double tp = resolve_tp(is_long, fill, idx);
            if (sl > 0.0) t.stop_loss   = sl;
            if (tp > 0.0) t.take_profit = tp;

            // Trailing: inicializa no entry
            const double* trail = is_long ? sig_trail_long : sig_trail_short;
            if (trail && trail[idx] > 0.0) {
                // Não sobrescreve SL se trailing é mais frouxo que o SL atual
                double trail_sl = is_long ? fill - trail[idx] : fill + trail[idx];
                double curr_sl  = t.stop_loss.value_or(0.0);
                if (is_long ? (trail_sl > curr_sl) : (trail_sl < curr_sl || curr_sl == 0.0))
                    t.stop_loss = trail_sl;
            }
        };

        // ── HELPER: update_trailing ────────────────────────────────────────────
        // Atualiza o trailing stop usando o preço `ref` (open ou high/low intra-candle)
        auto update_trailing = [&](Trade& t, bool is_long, double ref, int idx) {
            const double* trail = is_long ? sig_trail_long : sig_trail_short;
            if (!trail || trail[idx] <= 0.0) return;
            double dist = trail[idx];
            if (is_long) {
                double new_sl = ref - dist;
                double cur_sl = t.stop_loss.value_or(0.0);
                if (new_sl > cur_sl) t.stop_loss = new_sl;
            } else {
                double new_sl = ref + dist;
                double cur_sl = t.stop_loss.value_or(std::numeric_limits<double>::max());
                if (new_sl < cur_sl) t.stop_loss = new_sl;
            }
        };


        // ═════════════════════════════════════════════════════════════════════
        // MAIN LOOP
        // ═════════════════════════════════════════════════════════════════════
        for (size_t i = backtest_start_idx; i < n_bars; ++i) {

            int  currentTime     = extract_minutes(datetime[i]);
            bool is_last_bar     = (i == n_bars - 1);
            bool day_switched    = (!is_last_bar && bar_dates[i+1] != bar_dates[i]);
            bool dt_final        = (currentTime >= timeTF);
            if (day_switched) { day_trades_long = 0; day_trades_short = 0; }

            bool signal_long   = sig_entry_long  && sig_entry_long[i]  > 0.5;
            bool signal_short  = sig_entry_short && sig_entry_short[i] > 0.5;
            bool do_exit_long  = sig_exit_long   && sig_exit_long[i]   > 0.5;
            bool do_exit_short = sig_exit_short  && sig_exit_short[i]  > 0.5;


            // ── 1. EXIT LOGIC ─────────────────────────────────────────────────
            // Fecha trades ativos no open[i] (sinal vem do bar anterior)
            {
                auto it = active_trades.begin();
                while (it != active_trades.end()) {
                    bool is_long = (it->lot_size.value_or(0.0) > 0.0);
                    std::string reason = "";

                    if      (is_long ? do_exit_long : do_exit_short)           reason = "TF";
                    else if ([&]() -> bool {
                        int nb = is_long ? nb_long : nb_short;
                        if (nb <= 0 || it->bars_held < nb) return false;
                        if (exit_nb_only_if_pnl == 0) return true;
                        double p = (open[i] - *it->entry_price) / *it->entry_price * (is_long ? 1 : -1);
                        return (exit_nb_only_if_pnl > 0) ? p > 0 : p < 0;
                    }())                                                        reason = "NB";
                    else if (is_daytrade && (dt_final||day_switched||is_last_bar)) reason = "DT";
                    else if (!close_days.empty() &&
                             std::find(close_days.begin(), close_days.end(), bar_days[i]) != close_days.end())
                                                                                reason = "WC";
                    else if (is_last_bar)                                       reason = "EF";

                    // BE trigger — verifica ANTES do trailing
                    {
                        const double* be = is_long ? sig_be_trigger_long : sig_be_trigger_short;
                        if (be && be[i] > 0.0) {
                            double entry_p = *it->entry_price;
                            double cur_p   = open[i];
                            bool triggered = is_long ? (cur_p >= entry_p + be[i])
                                                     : (cur_p <= entry_p - be[i]);
                            if (triggered) {
                                double cur_sl = it->stop_loss.value_or(0.0);
                                // Só move SL para entry se ainda está do lado errado
                                bool sl_not_yet_at_entry = is_long ? (cur_sl < entry_p)
                                                                    : (cur_sl > entry_p || cur_sl == 0.0);
                                if (sl_not_yet_at_entry) it->stop_loss = entry_p;
                            }
                        }
                    }

                    // Trailing update no open (será refinado com H/L intra-candle na seção 4)
                    update_trailing(*it, is_long, open[i], i);

                    if (!reason.empty()) {
                        close_trade(*it, open[i], reason, datetime[i], is_long);
                        trades.push_back(*it);
                        it = active_trades.erase(it);
                    } else { ++it; }
                }
            }


            // ── 2. PENDING ORDERS ─────────────────────────────────────────────
            {
                auto p_it = pending_orders.begin();
                while (p_it != pending_orders.end()) {
                    bool is_long  = (p_it->lot_size.value_or(0.0) > 0.0);
                    double target = *p_it->entry_price;
                    bool triggered = false, expired = false;
                    double fill = target; // será ajustado se gap

                    if (p_it->bars_held.value_or(0) >= limit_order_expiry) expired = true;
                    else if (((dt_final || day_switched) && is_daytrade) || is_last_bar) expired = true;

                    if (expired) {
                        if (print_logs)
                            std::cout << "[LIM-EXP] " << (is_long?"L":"S")
                                      << " | " << datetime[i]
                                      << " | Target: " << target << std::endl;
                        p_it = pending_orders.erase(p_it);
                        continue;
                    }

                    std::string pos_type = p_it->exit_reason.value_or("");
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
                        // Reconstroi o trade com fill real, slippage, SL/TP e trailing
                        open_trade(*p_it, fill, i, is_long);

                        double sl = p_it->stop_loss.value_or(0.0);
                        double tp = p_it->take_profit.value_or(0.0);

                        // Checa SL/TP instantâneo no bar de execução
                        std::string instant = check_instant_exit(
                            is_long, *p_it->entry_price, sl, tp,
                            open[i], high[i], low[i], close[i]);

                        if (!instant.empty()) {
                            double exit_px = (instant == "TP") ? tp : sl;
                            close_trade(*p_it, exit_px, instant, datetime[i], is_long);
                            if (print_logs)
                                std::cout << "[LIM-INST] " << (is_long?"L":"S") << " | " << instant
                                          << " " << datetime[i]
                                          << " | Fill: " << std::fixed << std::setprecision(5) << *p_it->entry_price
                                          << " | PnL: " << std::setprecision(4) << p_it->profit.value_or(0) << "%" << std::endl;
                            trades.push_back(std::move(*p_it));
                            p_it = pending_orders.erase(p_it);
                        } else {
                            daily_results_matrix.push_back({format_datetime_to_int(datetime[i]), 0.0, ps_id});
                            if (print_logs)
                                std::cout << "[LIM-EXEC] " << (is_long?"L":"S") << " | " << pos_type
                                          << " " << datetime[i]
                                          << " | Fill: " << std::fixed << std::setprecision(5) << *p_it->entry_price
                                          << " | SL: " << sl << " | TP: " << tp << std::endl;
                            active_trades.push_back(std::move(*p_it));
                            p_it = pending_orders.erase(p_it);
                        }
                        is_long ? ++day_trades_long : ++day_trades_short;
                    } else {
                        p_it->bars_held = p_it->bars_held.value_or(0) + 1;
                        ++p_it;
                    }
                }
            }


            // ── 3. NEW ENTRY ──────────────────────────────────────────────────
            bool day_blocked = false;
            if (!close_days.empty())
                day_blocked = std::find(close_days.begin(), close_days.end(), bar_days[i]) != close_days.end();
            if (!day_blocked && is_daytrade)
                if (currentTime < timeEI || currentTime > timeEF) day_blocked = true;

            if (!day_blocked && (signal_long || signal_short)) {
                int cur_longs = 0, cur_shorts = 0;
                for (const auto& t : active_trades)  { double l=t.lot_size.value_or(0); if(l>0) cur_longs++; else if(l<0) cur_shorts++; }
                for (const auto& p : pending_orders) { double l=p.lot_size.value_or(0); if(l>0) cur_longs++; else if(l<0) cur_shorts++; }

                auto execute_entry_internal = [&](bool is_long) {

                    if (order_type == "market") {
                        // ── MARKET ORDER ──────────────────────────────────────
                        Trade t;
                        t.id = generate_id(); t.asset = asset_name; t.path = trade_path;
                        open_trade(t, open[i], i, is_long);

                        double sl = t.stop_loss.value_or(0.0);
                        double tp = t.take_profit.value_or(0.0);

                        // Instant SL/TP check: vela enorme pode já fechar no mesmo bar
                        std::string instant = check_instant_exit(
                            is_long, *t.entry_price, sl, tp,
                            open[i], high[i], low[i], close[i]);

                        if (!instant.empty()) {
                            double exit_px = (instant == "TP") ? tp : sl;
                            close_trade(t, exit_px, instant, datetime[i], is_long);
                            if (print_logs)
                                std::cout << "[MKT-INST] " << (is_long?"L":"S") << " | " << instant
                                          << " " << datetime[i]
                                          << " | PnL: " << t.profit.value_or(0) << "%" << std::endl;
                            trades.push_back(std::move(t));
                        } else {
                            daily_results_matrix.push_back({format_datetime_to_int(datetime[i]), 0.0, ps_id});
                            if (print_logs)
                                std::cout << "[EN-MKT] " << (is_long?"L":"S") << " | " << datetime[i]
                                          << " | Fill: " << std::fixed << std::setprecision(5) << *t.entry_price
                                          << " | SL: " << sl << " | TP: " << tp
                                          << " | Pos: " << (is_long ? cur_longs+1 : cur_shorts+1) << std::endl;
                            active_trades.push_back(std::move(t));
                        }
                        is_long ? ++day_trades_long : ++day_trades_short;

                    } else {
                        // ── LIMIT / STOP PENDING ORDER ────────────────────────
                        const double* lim_col = is_long ? sig_limit_long : sig_limit_short;
                        if (!lim_col || lim_col[i] <= 0.0) {
                            // Sem coluna limit → fallback market
                            Trade t; t.id=generate_id(); t.asset=asset_name; t.path=trade_path;
                            open_trade(t, open[i], i, is_long);
                            daily_results_matrix.push_back({format_datetime_to_int(datetime[i]), 0.0, ps_id});
                            active_trades.push_back(std::move(t));
                            is_long ? ++day_trades_long : ++day_trades_short;
                            return;
                        }

                        double target = lim_col[i];
                        std::string pos_type;
                        bool gap_too_big = false, already_hit = false;

                        if (is_long) {
                            if (target <= open[i]) {
                                pos_type = "L_BELOW"; // BUY LIMIT
                                double diff = (open[i] - target) / open[i];
                                if (limit_base_ref != "open" && open[i] <= target) already_hit = true;
                                if (diff > limit_perc_treshold) gap_too_big = true;
                            } else {
                                pos_type = "L_ABOVE"; // BUY STOP (breakout)
                                double diff = (target - open[i]) / open[i];
                                if (diff > limit_perc_treshold) gap_too_big = true;
                            }
                        } else {
                            if (target >= open[i]) {
                                pos_type = "S_ABOVE"; // SELL LIMIT
                                double diff = (target - open[i]) / open[i];
                                if (limit_base_ref != "open" && open[i] >= target) already_hit = true;
                                if (diff > limit_perc_treshold) gap_too_big = true;
                            } else {
                                pos_type = "S_BELOW"; // SELL STOP (breakout)
                                double diff = (open[i] - target) / open[i];
                                if (diff > limit_perc_treshold) gap_too_big = true;
                            }
                        }

                        if (gap_too_big) {
                            if (print_logs)
                                std::cout << "[SKIP-GAP] " << (is_long?"L":"S")
                                          << " | " << pos_type << " " << datetime[i]
                                          << " | Target: " << target << " | Open: " << open[i] << std::endl;
                            return;
                        }

                        // Gap favorável + configurado para entrar a mercado
                        if (already_hit && gap_market_fallback) {
                            Trade t; t.id=generate_id(); t.asset=asset_name; t.path=trade_path;
                            open_trade(t, open[i], i, is_long);
                            daily_results_matrix.push_back({format_datetime_to_int(datetime[i]), 0.0, ps_id});
                            if (print_logs)
                                std::cout << "[EN-GAP-MKT] " << (is_long?"L":"S") << " | " << datetime[i]
                                          << " | Fill: " << open[i] << std::endl;
                            active_trades.push_back(std::move(t));
                            is_long ? ++day_trades_long : ++day_trades_short;
                            return;
                        }

                        // Coloca a ordem pendente
                        Trade t;
                        t.id          = generate_id();
                        t.asset       = asset_name;
                        t.path        = trade_path;
                        t.entry_price = target;
                        t.status      = "pending";
                        t.entry_datetime = datetime[i];
                        t.bars_held   = 0;
                        t.lot_size    = is_long ? 1.0 : -1.0; // placeholder; recalculado no fill
                        t.exit_reason = pos_type;               // L_BELOW/L_ABOVE/S_ABOVE/S_BELOW

                        if (print_logs)
                            std::cout << "[EN-" << pos_type << "] " << (is_long?"L":"S")
                                      << " | " << datetime[i]
                                      << " | Target: " << std::fixed << std::setprecision(5) << target
                                      << " | Open: " << open[i] << std::endl;
                        pending_orders.push_back(std::move(t));
                    }
                };

                auto cancel_opposite = [&](bool long_sig) {
                    if (!cancel_opp_pending) return;
                    auto oit = pending_orders.begin();
                    while (oit != pending_orders.end()) {
                        double lot = oit->lot_size.value_or(0.0);
                        if (long_sig ? (lot < 0) : (lot > 0)) {
                            if (print_logs)
                                std::cout << "[CANCEL-OPP] " << datetime[i]
                                          << " | Target: " << *oit->entry_price << std::endl;
                            oit = pending_orders.erase(oit);
                        } else ++oit;
                    }
                };

                if (signal_long) {
                    cancel_opposite(true);
                    if (cur_longs < max_long && day_trades_long < max_long_pd &&
                        (hedge_enabled || cur_shorts == 0))
                        execute_entry_internal(true);
                }
                if (signal_short) {
                    cancel_opposite(false);
                    if (cur_shorts < max_short && day_trades_short < max_short_pd &&
                        (hedge_enabled || cur_longs == 0))
                        execute_entry_internal(false);
                }
            }


            // ── 4. INTRA-CANDLE SL/TP + TRAILING ─────────────────────────────
            // Atualiza trailing com H/L real do bar e verifica SL/TP
            {
                auto it = active_trades.begin();
                while (it != active_trades.end()) {
                    bool is_long = (it->lot_size.value_or(0.0) > 0.0);

                    // Trailing com extremos do candle (mais preciso que apenas open)
                    update_trailing(*it, is_long, is_long ? high[i] : low[i], i);

                    double sl = it->stop_loss.value_or(0.0);
                    double tp = it->take_profit.value_or(0.0);

                    bool hit_sl = is_long ? (sl > 0 && low[i]  <= sl) : (sl > 0 && high[i] >= sl);
                    bool hit_tp = is_long ? (tp > 0 && high[i] >= tp) : (tp > 0 && low[i]  <= tp);

                    if (hit_sl || hit_tp) {
                        std::string reason;
                        double exit_px;
                        if (hit_sl && hit_tp) {
                            // Ambos no mesmo bar: desempata pelo path do candle
                            reason = check_instant_exit(is_long, *it->entry_price, sl, tp,
                                                        open[i], high[i], low[i], close[i]);
                            if (reason.empty()) reason = "SL"; // fallback
                            exit_px = (reason == "TP") ? tp : sl;
                        } else {
                            reason  = hit_sl ? "SL" : "TP";
                            exit_px = hit_sl ? sl : tp;
                        }
                        close_trade(*it, exit_px, reason, datetime[i], is_long);
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
            }


            // ── 5. DAILY PnL UPDATE ───────────────────────────────────────────
            if (day_switched || dt_final || is_last_bar) {
                for (auto& trade : active_trades) {
                    double entry    = *trade.entry_price;
                    double curr_p   = close[i];
                    bool is_long    = (trade.lot_size.value_or(0.0) > 0);
                    double prev_p   = trade.prev_day_price.value_or(entry);
                    double daily_v  = ((curr_p - prev_p) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
                    daily_results_matrix.push_back({format_datetime_to_int(datetime[i]), daily_v, ps_id});
                    trade.prev_day_price = curr_p;
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[C++ Backtest Error]: " << e.what() << std::endl;
    }

    if (exec_settings.value("print_logs", false))
        std::cout << "[FINISH] Trades: " << trades.size()
                  << " | Net PnL: " << temp_cumulative_pnl << "%" << std::endl;

    std::string resolution = exec_settings.value("trade_pnl_resolution", "daily");
    SimulationOutput output;
    output.trades_json = trades_to_json(trades, resolution);
    output.daily_vec   = daily_results_matrix;
    return output;
}

double calculate_lot_size(double price_entry, bool is_long_trade) {
    return is_long_trade ? 1.0 : -1.0;
}
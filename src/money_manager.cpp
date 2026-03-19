#include "money_manager.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

// ── pool_val ──────────────────────────────────────────────────────────────────
double MoneyManager::pool_val(
    const std::string&                                          ref,
    size_t                                                      bar_idx,
    const std::unordered_map<std::string, const double*>&       fast_pool)
{
    auto lb = ref.find('[');
    if (lb == std::string::npos) {
        auto it = fast_pool.find(ref);
        if (it == fast_pool.end() || !it->second) return 0.0;
        return it->second[bar_idx];
    }
    std::string col    = ref.substr(0, lb);
    int         offset = std::stoi(ref.substr(lb + 1, ref.find(']') - lb - 1));
    auto it = fast_pool.find(col);
    if (it == fast_pool.end() || !it->second) return 0.0;
    int idx = (int)bar_idx - offset;
    return (idx >= 0) ? it->second[idx] : 0.0;
}

// ── apply_capital_method ──────────────────────────────────────────────────────
// "fixed"          → capital constante
// "compound_fract" → capital + profit_usd × fract  (escalar)
double MoneyManager::apply_capital_method(
    const json&                                             mm_params,
    double                                                  price,
    size_t                                                  bar_idx,
    double                                                  cumulative_profit,
    const std::unordered_map<std::string, const double*>&   fast_pool)
{
    const double capital = mm_params.value("capital",        100000.0);
    const std::string cm = mm_params.value("capital_method", "fixed");

    if (cm == "fixed") return capital;

    // compound — usa fract escalar ou série barra a barra se disponível
    double fract = mm_params.value("compound_fract", 1.0);
    double series_val = pool_val("compound_fract", bar_idx, fast_pool);
    if (series_val > 0.0) fract = series_val;  // série substitui escalar se presente

    double profit_usd   = (cumulative_profit / 100.0) * capital;
    double capital_base = capital + profit_usd * fract;
    return std::max(capital_base, capital * 0.1);
}

// ── apply_lot_constraints ─────────────────────────────────────────────────────
// Aplica lot_min, lot_max e lot_step do asset como camada final
// Equivalente ao que MT5 faz antes de enviar a ordem
double MoneyManager::apply_lot_constraints(double lot, const json& mm_params)
{
    double lot_min  = mm_params.value("lot_min",  0.01);
    double lot_max  = mm_params.value("lot_max",  10000.0);
    double lot_step = mm_params.value("lot_step", lot_min);

    if (lot_step <= 0.0) lot_step = lot_min;

    // Arredonda para o lot_step mais próximo
    double stepped = std::round(lot / lot_step) * lot_step;

    // Aplica min/max
    return std::max(lot_min, std::min(stepped, lot_max));
}

// ── resolve_dist ──────────────────────────────────────────────────────────────
double MoneyManager::resolve_dist(
    const json&                                             mm_params,
    double                                                  price,
    size_t                                                  bar_idx,
    const std::unordered_map<std::string, const double*>&   fast_pool)
{
    if (mm_params.contains("dist_ref") && mm_params["dist_ref"].is_string()) {
        double val = pool_val(mm_params["dist_ref"].get<std::string>(), bar_idx, fast_pool);
        if (val > 0.0) return val;
    }
    if (mm_params.contains("dist_fixed") && mm_params["dist_fixed"].is_number()) {
        double val = mm_params["dist_fixed"].get<double>();
        if (val > 0.0) return val;
    }
    if (mm_params.contains("tick") && mm_params["tick"].is_number()) {
        double val = mm_params["tick"].get<double>();
        if (val > 0.0) return val;
    }
    return price * 0.001;
}

// ── calc_kelly ────────────────────────────────────────────────────────────────
double MoneyManager::calc_kelly(double capital, double price, double tick_fin_val,
                                const std::vector<double>& profits, const json& mm_params)
{
    int min_trades = mm_params.value("min_trades", 30);
    if ((int)profits.size() < min_trades) return 1.0;

    int    wins = 0, losses = 0;
    double sum_win = 0.0, sum_loss = 0.0;
    for (double p : profits) {
        if (p > 0.0) { ++wins;   sum_win  += p; }
        else         { ++losses; sum_loss += std::abs(p); }
    }
    if (wins == 0 || losses == 0) return 1.0;

    double win_rate = (double)wins / profits.size();
    double avg_win  = sum_win  / wins;
    double avg_loss = sum_loss / losses;
    double b        = avg_win / avg_loss;
    double kelly_f  = (win_rate * b - (1.0 - win_rate)) / b;
    if (kelly_f <= 0.0) return 1.0;

    kelly_f *= mm_params.value("kelly_weight", 0.25);
    return (capital * kelly_f) / (price * tick_fin_val);
}

// ── calc_var ──────────────────────────────────────────────────────────────────
double MoneyManager::calc_var(double capital, double price, double tick_fin_val,
                              const std::vector<double>& profits, const json& mm_params)
{
    int min_trades = mm_params.value("min_trades", 30);
    if ((int)profits.size() < min_trades) return 1.0;

    double confidence = mm_params.value("var_confidence", 0.95);
    std::vector<double> sorted = profits;
    std::sort(sorted.begin(), sorted.end());

    size_t idx = (size_t)std::floor((1.0 - confidence) * sorted.size());
    idx = std::min(idx, sorted.size() - 1);
    double var = sorted[idx];

    if (var >= 0.0) return 1.0;

    double risk_pct = mm_params.value("risk_pct", 0.01);
    double risk_usd = capital * risk_pct;
    return risk_usd / (std::abs(var) * tick_fin_val);
}

// ── calculate ─────────────────────────────────────────────────────────────────
LotResult MoneyManager::calculate(
    const json&                                             mm_params,
    double                                                  price,
    bool                                                    is_long,
    size_t                                                  bar_idx,
    const std::unordered_map<std::string, const double*>&   fast_pool,
    const std::vector<double>&                              trade_profits,
    double                                                  cumulative_profit)
{
    if (mm_params.is_null() || !mm_params.is_object())
        return { is_long ? 1.0 : -1.0, is_long };

    const std::string method = mm_params.value("method", "neutral");

    // Capital base ajustado pelo capital_method
    double capital      = apply_capital_method(mm_params, price, bar_idx, cumulative_profit, fast_pool);
    double tick_fin_val = mm_params.value("tick_fin_val", 1.0);
    double tick         = mm_params.value("tick",         0.01);
    double risk_pct     = mm_params.value("risk_pct",     0.01);

    double lot = 1.0;

    if (method == "neutral") {
        lot = 1.0;
    }
    else if (method == "fixed") {
        lot = mm_params.value("fixed_lot", 1.0);
    }
    else if (method == "risk_per_trade") {
        double dist       = resolve_dist(mm_params, price, bar_idx, fast_pool);
        double dist_ticks = dist / tick;
        if (dist_ticks > 0.0) {
            lot = (capital * risk_pct) / (dist_ticks * tick_fin_val);
        }
    }
    else if (method == "pct_capital") {
        // Sem clamp de risk — pct já controla diretamente o tamanho
        double pct = mm_params.value("pct", 0.02);
        if (price > 0.0 && tick_fin_val > 0.0)
            lot = (capital * pct) / (price * tick_fin_val);
    }
    else if (method == "kelly") {
        lot = calc_kelly(capital, price, tick_fin_val, trade_profits, mm_params);
    }
    else if (method == "var") {
        lot = calc_var(capital, price, tick_fin_val, trade_profits, mm_params);
    }
    else if (method == "signal") {
        std::string ref_key = is_long
            ? mm_params.value("ref_long",  "custom_lot_size_long")
            : mm_params.value("ref_short", "custom_lot_size_short");
        double val = pool_val(ref_key, bar_idx, fast_pool);
        lot = (val > 0.0) ? val : 1.0;
    }
    else {
        std::cerr << "[MoneyManager] Unknown method: " << method << " — using neutral\n";
        lot = 1.0;
    }

    // Aplica constraints do asset (lot_min, lot_max, lot_step) como camada final
    // Equivalente ao que MT5 faz antes de enviar a ordem
    lot = apply_lot_constraints(lot, mm_params);

    return { is_long ? lot : -lot, is_long };
}
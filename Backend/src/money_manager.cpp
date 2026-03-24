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
// "fixed"    → capital constante
// "compound" → capital + profit_usd × fract
//              fract: escalar mm["compound_fract"] ou série fast_pool["compound_fract"]
double MoneyManager::apply_capital_method(
    const json&                                             mm_params,
    size_t                                                  bar_idx,
    double                                                  cumulative_profit,
    const std::unordered_map<std::string, const double*>&   fast_pool)
{
    const double capital = mm_params.value("capital",        100000.0);
    const std::string cm = mm_params.value("capital_method", "fixed");

    if (cm == "fixed") return capital;

    // compound — fract escalar ou série (série substitui escalar se presente)
    double fract = mm_params.value("compound_fract", 1.0);
    double series_val = pool_val("compound_fract_series", bar_idx, fast_pool);
    if (series_val > 0.0) fract = series_val;

    // cumulative_profit em % → converte para $ usando capital inicial
    double profit_usd   = (cumulative_profit / 100.0) * capital;
    double capital_base = capital + profit_usd * fract;

    // Nunca desce abaixo de 10% do capital inicial
    return std::max(capital_base, capital * 0.1);
}

// ── resolve_dist ──────────────────────────────────────────────────────────────
// Prioridade: fast_pool["dist_ref"] → abs(entry - sl) → dist_fixed → tick
double MoneyManager::resolve_dist(
    const json&                                             mm_params,
    double                                                  price,
    double                                                  sl_price,
    size_t                                                  bar_idx,
    const std::unordered_map<std::string, const double*>&   fast_pool)
{
    // 1. Série calculada em Python (ATR, range, etc.)
    double pool_dist = pool_val("dist_ref", bar_idx, fast_pool);
    if (pool_dist > 0.0) return pool_dist;

    // 2. SL do trade — distância natural do risco definida pelo usuário
    if (sl_price > 0.0) {
        double sl_dist = std::abs(price - sl_price);
        if (sl_dist > 0.0) return sl_dist;
    }

    // 3. Valor fixo em pontos
    if (mm_params.contains("dist_fixed") && mm_params["dist_fixed"].is_number()) {
        double val = mm_params["dist_fixed"].get<double>();
        if (val > 0.0) return val;
    }

    // 4. Tick do asset — fallback mínimo
    if (mm_params.contains("tick") && mm_params["tick"].is_number()) {
        double val = mm_params["tick"].get<double>();
        if (val > 0.0) return val;
    }

    return price * 0.001;
}

// ── apply_lot_constraints ─────────────────────────────────────────────────────
// Aplica min_lot, max_lot e lot_step — equivalente ao que MT5 faz antes de enviar ordem
double MoneyManager::apply_lot_constraints(double lot, const json& mm_params)
{
    double min_lot  = mm_params.value("min_lot",  0.01);
    double max_lot  = mm_params.value("max_lot",  10000.0);
    double lot_step = mm_params.value("lot_step", min_lot);

    if (lot_step <= 0.0) lot_step = min_lot;

    double stepped = std::round(lot / lot_step) * lot_step;
    return std::max(min_lot, std::min(stepped, max_lot));
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
    return (capital * risk_pct) / (std::abs(var) * tick_fin_val);
}

// ── calculate ─────────────────────────────────────────────────────────────────
LotResult MoneyManager::calculate(
    const json&                                             mm_params,
    double                                                  price,
    bool                                                    is_long,
    double                                                  sl_price,
    size_t                                                  bar_idx,
    const std::unordered_map<std::string, const double*>&   fast_pool,
    const std::vector<double>&                              trade_profits,
    double                                                  cumulative_profit)
{
    if (mm_params.is_null() || !mm_params.is_object())
        return { is_long ? 1.0 : -1.0, is_long };

    const std::string method = mm_params.value("method", "neutral");

    double capital      = apply_capital_method(mm_params, bar_idx, cumulative_profit, fast_pool);
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
        double dist       = resolve_dist(mm_params, price, sl_price, bar_idx, fast_pool);
        double dist_ticks = dist / tick;
        if (dist_ticks > 0.0)
            lot = (capital * risk_pct) / (dist_ticks * tick_fin_val);
    }
    else if (method == "pct_capital") {
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

    // Constraints do asset — camada final (min_lot, max_lot, lot_step)
    lot = apply_lot_constraints(lot, mm_params);

    return { is_long ? lot : -lot, is_long };
}
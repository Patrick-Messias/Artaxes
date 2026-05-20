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
    MMContext                                               mm_context,
    AssetContext                                            asset_context,
    size_t                                                  bar_idx,
    double                                                  cumulative_profit,
    const std::unordered_map<std::string, const double*>&   fast_pool)
{
    double capital = mm_context.capital;
    const std::string capital_coin = asset_context.capital_coin;
    const std::string cm = mm_context.capital_method;
    double fract = mm_context.compound_fract.value_or(1.0); // fractal default is 1.0

    if (cm == "fixed") return capital;

    // compound — fract escalar ou série (série substitui escalar se presente)
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
    MMContext                                               mm_context, 
    AssetContext                                            asset_context,
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
    double dist_fix = mm_context.dist_fixed.value_or(0.0);
    if (dist_fix > 0) return dist_fix;

    // 4. Tick do asset — fallback mínimo
    double tick = asset_context.tick_size;
    if (tick > 0) return tick;

    return price * 0.001;
}

// ── apply_lot_constraints ─────────────────────────────────────────────────────
// Aplica min_lot, max_lot e lot_step — equivalente ao que MT5 faz antes de enviar ordem
double MoneyManager::apply_lot_constraints(AssetContext asset_context, double lot)
{
    double min_lot  = asset_context.min_lot; 
    double max_lot  = asset_context.lot_max; 
    double lot_step = asset_context.min_lot; 

    if (lot_step <= 0.0) lot_step = min_lot;

    double stepped = std::round(lot / lot_step) * lot_step;
    return std::max(min_lot, std::min(stepped, max_lot));
}

// ── calc_kelly ────────────────────────────────────────────────────────────────
double MoneyManager::calc_kelly(MMContext                                               mm_context,
                                double                                                  capital,
                                double                                                  price,
                                double                                                  tick_fin_val,
                                const std::vector<double>&                              profits)
{
    int min_trades = mm_context.min_trades.value_or(0);
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

    double kelly_weight = mm_context.kelly_weight.value_or(0.25);
    kelly_f *= kelly_weight;
    return (capital * kelly_f) / (price * tick_fin_val);
}

// ── calc_var ──────────────────────────────────────────────────────────────────
double MoneyManager::calc_var(MMContext                                               mm_context, 
                              double                                                  capital, 
                              double                                                  price, 
                              double                                                  tick_fin_val,
                              const std::vector<double>&                              profits)
{
    int min_trades = mm_context.min_trades.value_or(0);
    if ((int)profits.size() < min_trades) return 1.0;

    double confidence = mm_context.var_confidence.value_or(0.95); 
    std::vector<double> sorted = profits;
    std::sort(sorted.begin(), sorted.end());

    size_t idx = (size_t)std::floor((1.0 - confidence) * sorted.size());
    idx = std::min(idx, sorted.size() - 1);
    double var = sorted[idx];

    if (var >= 0.0) return 1.0;

    double risk_pct = mm_context.risk_pct.value_or(0.01); 
    return (capital * risk_pct) / (std::abs(var) * tick_fin_val);
}

// ── calculate ─────────────────────────────────────────────────────────────────
LotResult MoneyManager::calculate(
    MMContext                                               mm_context,
    AssetContext                                            asset_context,
    double                                                  price,
    bool                                                    is_long,
    double                                                  sl_price,
    size_t                                                  bar_idx,
    const std::unordered_map<std::string, const double*>&   fast_pool,
    const std::vector<double>&                              trade_profits,
    double                                                  cumulative_profit)
{
    const std::string method = mm_context.sizing_method;
    double min_lot = asset_context.min_lot;
    double contract_size = asset_context.contract_size;
    double leverage = asset_context.leverage;
    const std::string capital_coin = asset_context.capital_coin;

    double calculate_capital      = apply_capital_method(mm_context, asset_context, bar_idx, cumulative_profit, fast_pool);
    double tick_fin_val = asset_context.tick_val;
    double tick         = asset_context.tick_size;
    double risk_pct     = mm_context.risk_pct.value_or(0.01);

    double lot = 1.0;

    if (method == "neutral") {
        lot = min_lot;
    }
    else if (method == "fixed") {
        lot = mm_context.fixed_lot.value_or(1.0);
    }
    else if (method == "risk_per_trade") {
        double dist       = resolve_dist(mm_context, asset_context, price, sl_price, bar_idx, fast_pool);
        double dist_ticks = dist / tick;
        if (dist_ticks > 0.0)
            lot = (calculate_capital * risk_pct) / (dist_ticks * tick_fin_val);
    }
    else if (method == "pct_capital") {
        double pct = mm_context.pct.value_or(0.01);
        if (price > 0.0 && tick_fin_val > 0.0)
            lot = (calculate_capital * pct) / (price * tick_fin_val);
    }
    else if (method == "kelly") {
        lot = calc_kelly(mm_context, calculate_capital, price, tick_fin_val, trade_profits);
    }
    else if (method == "var") {
        lot = calc_var(mm_context, calculate_capital, price, tick_fin_val, trade_profits);
    }
    else if (method == "signal") {
        std::string ref_key = is_long
            ? mm_context.ref_long.value_or("custom_lot_size_long") 
            : mm_context.ref_short.value_or("custom_lot_size_short");
        double val = pool_val(ref_key, bar_idx, fast_pool);
        lot = (val > 0.0) ? val : 1.0;
    }
    else {
        std::cerr << "[MoneyManager] Unknown method: " << method << " — using neutral\n";
        lot = 1.0;
    }

    // Constraints do asset — camada final (min_lot, max_lot, lot_step)
    lot = apply_lot_constraints(asset_context, lot);

    double margin_req = (price * contract_size * std::abs(lot)) / leverage;
    double scale = (margin_req > 0) ? (calculate_capital / margin_req) : 1.0;

    // Applies lot side
    lot = is_long ? lot : -lot;

    return { lot, is_long, margin_req, scale };
}
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct LotResult {
    double lot_size;  // positivo=long, negativo=short
    bool   is_long;
};

// ── MoneyManager C++ ──────────────────────────────────────────────────────────
// sizing_method:
//   "neutral"        → lot = 1.0
//   "fixed"          → lot fixo em mm["fixed_lot"]
//   "risk_per_trade" → (capital × risk_pct) / (dist_ticks × tick_fin_val)
//   "pct_capital"    → (capital × pct) / (price × tick_fin_val)  — sem clamp
//   "kelly"          → kelly fraction usando trade_profits acumulados
//   "var"            → VaR usando trade_profits acumulados
//   "signal"         → lot vem do fast_pool via "custom_lot_size_long/short"
//
// capital_method:
//   "fixed"    → capital constante
//   "compound" → capital + (cumulative_profit% / 100 × capital) × compound_fract
//                compound_fract substituído por fast_pool["compound_fract"] se presente
//
// dist prioridade para risk_per_trade:
//   fast_pool["dist_ref"] → abs(entry - sl_price) → mm["dist_fixed"] → mm["tick"]
//
// lot constraints (camada final — equivalente MT5):
//   mm["min_lot"], mm["max_lot"], mm["lot_step"]

class MoneyManager {
public:
    static LotResult calculate(
        const json&                                             mm_params,
        double                                                  price,
        bool                                                    is_long,
        double                                                  sl_price,      // SL do trade (0 se não definido)
        size_t                                                  bar_idx,
        const std::unordered_map<std::string, const double*>&   fast_pool,
        const std::vector<double>&                              trade_profits,
        double                                                  cumulative_profit
    );

private:
    static double apply_capital_method(const json& mm_params, size_t bar_idx,
                                       double cumulative_profit,
                                       const std::unordered_map<std::string, const double*>& fast_pool);

    static double resolve_dist(const json& mm_params, double price, double sl_price,
                               size_t bar_idx,
                               const std::unordered_map<std::string, const double*>& fast_pool);

    static double apply_lot_constraints(double lot, const json& mm_params);

    static double pool_val(const std::string& ref, size_t bar_idx,
                           const std::unordered_map<std::string, const double*>& fast_pool);

    static double calc_kelly(double capital, double price, double tick_fin_val,
                             const std::vector<double>& profits, const json& mm_params);

    static double calc_var(double capital, double price, double tick_fin_val,
                           const std::vector<double>& profits, const json& mm_params);
};
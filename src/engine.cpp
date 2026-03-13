#include "engine.h"
#include "operation.h"
#include <iostream>

// Zeller's congruence — dia da semana sem mktime/string
// Retorna 0=Dom, 1=Seg … 6=Sáb
static int weekday_from_ymd(int y, int m, int d) {
    static const int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    if (m < 3) y--;
    return (y + y/4 - y/100 + y/400 + t[m-1] + d) % 7;
}

EngineResult Engine::execute(
    const std::string&                                header,
    const std::unordered_map<std::string, const double*>& ohlc_arrays,
    size_t                                            n_bars,
    const std::vector<int64_t>&                       datetime_int,
    const std::unordered_map<std::string, const double*>& indicators_pool,
    const json&                                       sim_params,
    const json&                                       exec_settings
) {
    try {
        // ── Pre-computa datas UMA VEZ para todas as simulações ────────────────
        std::vector<int> bar_dates(n_bars), bar_times(n_bars), bar_days(n_bars);
        for (size_t i = 0; i < n_bars; ++i) {
            int64_t val  = datetime_int[i];
            int64_t dpart = val / 1000000LL;   // YYYYMMDD
            int64_t tpart = val % 1000000LL;   // HHMMSS
            bar_dates[i] = (int)dpart;
            bar_times[i] = (int)tpart;
            int y = (int)(dpart / 10000);
            int m = (int)((dpart % 10000) / 100);
            int d = (int)(dpart % 100);
            bar_days[i]  = weekday_from_ymd(y, m, d);
        }

        // ── Monta fast_pool ───────────────────────────────────────────────────
        std::unordered_map<std::string, const double*> fast_pool;
        fast_pool.reserve(ohlc_arrays.size() + indicators_pool.size() + 32);

        for (const auto& [key, ptr] : ohlc_arrays)
            fast_pool[key] = ptr;

        for (const auto& [pool_key, ptr] : indicators_pool) {
            size_t pos = pool_key.rfind("__");
            std::string col = (pos == std::string::npos) ? pool_key : pool_key.substr(pos + 2);
            fast_pool[col] = ptr;
        }
        
        return Operation::run(header, fast_pool, n_bars,
                              bar_dates, bar_times, bar_days,
                              sim_params, exec_settings);

    } catch (const std::exception& e) {
        std::cerr << "[Engine::execute error]: " << e.what() << std::endl;
        return {};
    }
}
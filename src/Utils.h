#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include "Trade.h"

// ── trades_to_json ────────────────────────────────────────────────────────────
inline nlohmann::json trades_to_json(const std::vector<Trade>& trades,
                                     const std::string& /*resolution*/ = "daily") {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& t : trades) {
        j.push_back({
            {"id",             t.id},
            {"asset",          t.asset},
            {"path",           t.path},
            {"status",         t.status},
            {"entry_price",    t.entry_price},
            {"entry_datetime", t.entry_datetime},
            {"lot_size",       t.lot_size},
            {"stop_loss",      t.stop_loss   ? nlohmann::json(*t.stop_loss)   : nlohmann::json(nullptr)},
            {"take_profit",    t.take_profit ? nlohmann::json(*t.take_profit) : nlohmann::json(nullptr)},
            {"exit_price",     t.exit_price  ? nlohmann::json(*t.exit_price)  : nlohmann::json(nullptr)},
            {"exit_datetime",  t.exit_datetime ? nlohmann::json(*t.exit_datetime) : nlohmann::json(nullptr)},
            {"exit_reason",    t.exit_reason   ? nlohmann::json(*t.exit_reason)   : nlohmann::json(nullptr)},
            {"profit",         t.profit        ? nlohmann::json(*t.profit)        : nlohmann::json(nullptr)},
            {"profit_r",       t.profit_r      ? nlohmann::json(*t.profit_r)      : nlohmann::json(nullptr)},
            {"mfe",            t.mfe           ? nlohmann::json(*t.mfe)           : nlohmann::json(nullptr)},
            {"mae",            t.mae           ? nlohmann::json(*t.mae)           : nlohmann::json(nullptr)},
            {"bars_held",      t.bars_held     ? nlohmann::json(*t.bars_held)     : nlohmann::json(nullptr)},
        });
    }
    return j;
}

// ── DailyResult ───────────────────────────────────────────────────────────────
struct DailyResult {
    long long ts;     // YYYYMMDDHHMMSS como int64
    double    pnl;
    int       ps_id;
};

inline void to_json(nlohmann::json& j, const DailyResult& res) {
    j = nlohmann::json{{"ts", res.ts}, {"pnl", res.pnl}, {"id", res.ps_id}};
}
inline void from_json(const nlohmann::json& j, DailyResult& res) {
    j.at("ts").get_to(res.ts);
    j.at("pnl").get_to(res.pnl);
    j.at("id").get_to(res.ps_id);
}

// ── format_datetime_to_int ────────────────────────────────────────────────────
// Remove '-', ':' e ' ' manualmente para evitar dependência de <algorithm>
inline long long format_datetime_to_int(const std::string& dt_str) {
    std::string out;
    out.reserve(dt_str.size());
    for (char c : dt_str) {
        if (c != '-' && c != ':' && c != ' ')
            out += c;
    }
    try { return std::stoll(out); } catch (...) { return 0; }
}

// ── format_datetime_to_int_from_parts ────────────────────────────────────────
// bar_date = YYYYMMDD, bar_time = HHMMSS  (inteiros pré-computados)
inline long long format_datetime_to_int_from_parts(int bar_date, int bar_time) {
    return (long long)bar_date * 1000000LL + (long long)bar_time;
}

// ── extract_minutes ───────────────────────────────────────────────────────────
inline int extract_minutes(const std::string& dt) {
    if (dt.length() < 16) return -1;
    int hh = (dt[11] - '0') * 10 + (dt[12] - '0');
    int mm = (dt[14] - '0') * 10 + (dt[15] - '0');
    return hh * 60 + mm;
}

// ── get_day_of_week ───────────────────────────────────────────────────────────
inline int get_day_of_week(const std::string& dt_str) {
    std::tm tm = {};
    std::istringstream ss(dt_str);
    ss >> std::get_time(&tm, "%Y-%m-%d");
    if (ss.fail()) return -1;
    std::mktime(&tm);
    return tm.tm_wday;
}

// ── export_to_csv ─────────────────────────────────────────────────────────────
inline void export_to_csv(const std::string& filename,
                          const std::vector<std::string>& datetime,
                          const std::map<std::string, std::vector<double>>& data) {
    std::ofstream file(filename);
    file << "datetime";
    for (auto const& [col_name, _] : data) file << "," << col_name;
    file << "\n";
    for (size_t i = 0; i < datetime.size(); ++i) {
        file << datetime[i];
        for (auto const& [col_name, vec] : data)
            file << "," << (i < vec.size() ? std::to_string(vec[i]) : "NaN");
        file << "\n";
    }
}
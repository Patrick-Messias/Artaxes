#pragma once
#include <nlohmann/json.hpp>
using json = nlohmann::json;

json run_engine(const json& data);

#pragma once
#include <string>
#include <vector>
#include "Trade.h"

class Engine {
public:
    static std::vector<Trade> run(const std::string& payload_json);
};

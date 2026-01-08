#pragma once

#include <string>

struct Metadata {
    std::string name;
    std::string version;
};

Metadata load_metadata();

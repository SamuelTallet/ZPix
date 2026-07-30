#pragma once

#include <cstdint>
#include <string>

struct Metadata {
    std::string name;
    std::string version;
    uint16_t httpPort; // Zero if unreadable.
};

Metadata load_metadata();

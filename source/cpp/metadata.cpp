#include "metadata.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

static std::string read_file(const std::string& filename) {
    std::ifstream f("metadata/" + filename);
    if (!f.is_open()) return "";
    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string s = buffer.str();
    s.erase(s.find_last_not_of(" \n\r\t") + 1);
    return s;
}

static uint16_t read_port(const std::string& filename) {
    try {
        unsigned long port = std::stoul(read_file(filename));
        if (port < 1 || port > 65535) return 0;
        return static_cast<uint16_t>(port);
    } catch (const std::exception&) {
        return 0;
    }
}

Metadata load_metadata() {
    Metadata m;
    m.name = read_file("NAME");
    m.version = read_file("VERSION");
    m.httpPort = read_port("HTTP_PORT");
    return m;
}

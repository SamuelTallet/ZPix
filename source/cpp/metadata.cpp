#include "metadata.hpp"
#include <fstream>
#include <sstream>

static std::string read_file(const std::string& filename) {
    std::ifstream f("metadata/" + filename);
    if (!f.is_open()) return "";
    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string s = buffer.str();
    s.erase(s.find_last_not_of(" \n\r\t") + 1);
    return s;
}

Metadata load_metadata() {
    Metadata m;
    m.name = read_file("NAME");
    m.version = read_file("VERSION");
    return m;
}

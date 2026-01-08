#include "console.hpp"

Console::Console(const std::string& title) : _title(title) {
}

Console::~Console() {
}

void Console::hide() {
    HWND h = FindWindowA(NULL, _title.c_str());
    if (h) {
        ShowWindow(h, SW_HIDE);
    }
}

void Console::show() {
    HWND h = FindWindowA(NULL, _title.c_str());
    if (h) {
        ShowWindow(h, SW_SHOW);
    }
}

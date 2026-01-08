#pragma once

#include <windows.h>
#include <string>

class Console {
public:
    Console(const std::string& title);
    ~Console();

    void hide();
    void show();

private:
    std::string _title;
};

#pragma once

#include <thread>
#include <functional>
#include <cstdint>

class WatcherThread {
public:
    WatcherThread(uint16_t port, std::function<void()> onPortOpen);
    ~WatcherThread();

private:
    void run(std::stop_token stoken, uint16_t port, std::function<void()> onPortOpen);
    std::jthread worker;
};

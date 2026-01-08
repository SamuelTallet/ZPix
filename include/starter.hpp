#pragma once

#include <string>
#include <thread>
#include <functional>
#include "processes.hpp"

class StarterThread {
public:
    StarterThread(uint16_t port, JobObject& job, std::function<void()> on_exit);
    ~StarterThread();

private:
    void run(std::stop_token stoken, uint16_t port, JobObject& job, std::function<void()> on_exit);
    std::jthread worker;
};

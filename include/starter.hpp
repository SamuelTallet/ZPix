#pragma once

#include <string>
#include <thread>
#include <functional>
#include "processes.hpp"

class StarterThread {
public:
    StarterThread(JobObject& job, std::function<void()> on_exit);
    ~StarterThread();

private:
    void run(std::stop_token stoken, JobObject& job, std::function<void()> on_exit);
    std::jthread worker;
};

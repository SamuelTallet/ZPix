#include "starter.hpp"
#include <windows.h>
#include <vector>

void StarterThread::run(std::stop_token stoken, JobObject& job, std::function<void()> on_exit) {
    std::string cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File start.ps1";

    STARTUPINFOA si = { sizeof(si) };
    PROCESS_INFORMATION pi = { 0 };
    
    std::vector<char> cmdVec(cmd.begin(), cmd.end());
    cmdVec.push_back(0);

    cmdVec.push_back(0);

    if (CreateProcessA(NULL, cmdVec.data(), NULL, NULL, FALSE, CREATE_SUSPENDED | CREATE_NEW_CONSOLE, NULL, NULL, &si, &pi)) {
        job.add_process(pi.hProcess);
        ResumeThread(pi.hThread);
        
        HANDLE hStopEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
        std::stop_callback cb(stoken, [hStopEvent]() { SetEvent(hStopEvent); });

        HANDLE handles[] = { pi.hProcess, hStopEvent };

        if (WaitForMultipleObjects(2, handles, FALSE, INFINITE) == WAIT_OBJECT_0) {
            if (on_exit && !stoken.stop_requested()) on_exit();
        }

        CloseHandle(hStopEvent);

        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
}

StarterThread::StarterThread(JobObject& job, std::function<void()> on_exit) {
    worker = std::jthread([this, &job, on_exit](std::stop_token stoken) { this->run(stoken, job, on_exit); });
}

StarterThread::~StarterThread() {
}

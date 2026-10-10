#include "starter.hpp"
#include <windows.h>
#include <vector>

void StarterThread::run(std::stop_token stoken, JobObject& job, std::function<void()> on_exit) {
    // Full path since PowerShell folder may be missing from PATH.
    char systemDir[MAX_PATH];
    GetSystemDirectoryA(systemDir, MAX_PATH);
    std::string powershell = std::string(systemDir) + "\\WindowsPowerShell\\v1.0\\powershell.exe";
    std::string cmd = "\"" + powershell + "\" -NoProfile -ExecutionPolicy Bypass -File start.ps1";

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
    } else {
        // Without this, the launcher would wait forever with no window nor console.
        DWORD error = GetLastError();
        // Quit before showing error so this instance can't open a window meanwhile.
        if (on_exit) on_exit();
        char* reason = nullptr;
        FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, error, 0, reinterpret_cast<LPSTR>(&reason), 0, NULL);
        std::string message = "Failed to run PowerShell (error " + std::to_string(error) + ").\n" + (reason ? reason : "");
        LocalFree(reason);
        MessageBoxA(NULL, message.c_str(), "Initialization Error", MB_ICONERROR);
    }
}

StarterThread::StarterThread(JobObject& job, std::function<void()> on_exit) {
    worker = std::jthread([this, &job, on_exit](std::stop_token stoken) { this->run(stoken, job, on_exit); });
}

StarterThread::~StarterThread() {
}

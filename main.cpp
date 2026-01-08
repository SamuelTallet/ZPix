#include <windows.h>
#include <string>

#include "free_port.hpp"
#include "metadata.hpp"
#include "processes.hpp"
#include "starter.hpp"
#include "watcher.hpp"
#include "webview.hpp"
#include "console.hpp"


int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR, int nCmdShow) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    AttachConsole(ATTACH_PARENT_PROCESS);

    auto metadata = load_metadata();
    
    uint16_t port = 0;
    try {
        port = find_free_port();
    } catch (const std::exception& e) {
        MessageBoxA(NULL, e.what(), "Initialization Error", MB_ICONERROR);
        return 1;
    }

    std::wstring url = L"http://127.0.0.1:" + std::to_wstring(port);

    MSG msg;
    PeekMessageW(&msg, NULL, WM_USER, WM_USER, PM_NOREMOVE);

    JobObject job;
    DWORD mainThreadId = GetCurrentThreadId();

    StarterThread starter(port, job, [mainThreadId]() {
        PostThreadMessageW(mainThreadId, WM_QUIT, 0, 0);
    });

    std::string title = metadata.name + " " + metadata.version;
    std::wstring wName(title.begin(), title.end());

    Webview webview;
    HWND hwnd = webview.CreateWin(hInstance, wName);

    if (hwnd == NULL) {
        return 0;
    }

    WatcherThread watcher(port, [hwnd, &url, &title]() {
        Console console(title);
        console.hide();
        PostMessageW(hwnd, WM_APP_WEBVIEW_READY, 0, reinterpret_cast<LPARAM>(url.c_str()));
    });

    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    return 0;
}

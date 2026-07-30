// ZPix launcher for Windows.

#include <windows.h>
#include <string>

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

    if (metadata.httpPort == 0) {
        MessageBoxA(NULL, "Invalid or missing metadata/HTTP_PORT", "Initialization Error", MB_ICONERROR);
        return 1;
    }

    std::wstring url = L"http://127.0.0.1:" + std::to_wstring(metadata.httpPort);

    MSG msg;
    PeekMessageW(&msg, NULL, WM_USER, WM_USER, PM_NOREMOVE);

    JobObject job;
    DWORD mainThreadId = GetCurrentThreadId();

    StarterThread starter(job, [mainThreadId]() {
        PostThreadMessageW(mainThreadId, WM_QUIT, 0, 0);
    });

    std::string title = metadata.name + " " + metadata.version;
    std::wstring wName(title.begin(), title.end());

    Webview webview;
    HWND hwnd = webview.CreateWin(hInstance, wName);

    if (hwnd == NULL) {
        return 0;
    }

    WatcherThread watcher(metadata.httpPort, [hwnd, &url, &title]() {
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

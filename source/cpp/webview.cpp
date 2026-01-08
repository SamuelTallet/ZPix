#include "webview.hpp"
#include <dwmapi.h>

static Webview* s_instance = nullptr;

static void UpdateTitleBarTheme(HWND hwnd) {
    DWORD value = 1;
    DWORD size = sizeof(value);
    if (::RegGetValueW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize", L"AppsUseLightTheme", RRF_RT_REG_DWORD, nullptr, &value, &size) != ERROR_SUCCESS) {
        value = 0;
    }
    BOOL useImmersiveDarkMode = (value == 0);
    ::DwmSetWindowAttribute(hwnd, 20, &useImmersiveDarkMode, sizeof(useImmersiveDarkMode));
}

Webview::Webview() {
    s_instance = this;
}

Webview::~Webview() {
    if (webviewController) {
        webviewController->Close();
    }
    if (s_instance == this) {
        s_instance = nullptr;
    }
}

void Webview::Initialize(HWND parent, const std::wstring& url) {
    std::wstring userDataFolder = L"temp";
    CreateCoreWebView2EnvironmentWithOptions(nullptr, userDataFolder.c_str(), nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [this, parent, url](HRESULT result, ICoreWebView2Environment* env) -> HRESULT {
                if (FAILED(result)) {
                    MessageBoxW(parent, L"Failed to create WebView2 environment.", L"Error", MB_OK);
                    return result;
                }

                env->CreateCoreWebView2Controller(parent,
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [this, parent, url](HRESULT result, ICoreWebView2Controller* controller) -> HRESULT {
                            if (FAILED(result)) {
                                return result;
                            }

                            if (controller) {
                                webviewController = controller;
                                webviewController->get_CoreWebView2(&webviewWindow);
                            }
                            
                            RECT bounds;
                            GetClientRect(parent, &bounds);
                            Resize(bounds);
                            
                            if (webviewWindow) {
                                webviewWindow->add_WebMessageReceived(
                                    Callback<ICoreWebView2WebMessageReceivedEventHandler>(
                                        [this](ICoreWebView2* sender, ICoreWebView2WebMessageReceivedEventArgs* args) -> HRESULT {
                                            LPWSTR message;
                                            args->TryGetWebMessageAsString(&message);
                                            if (message) {
                                                std::wstring msg(message);
                                                if (msg.find(L"open_external:") == 0) {
                                                    std::wstring url = msg.substr(14);
                                                    OpenWithDefaultBrowser(url);
                                                }
                                                CoTaskMemFree(message);
                                            }
                                            return S_OK;
                                        }).Get(), nullptr);

                                webviewWindow->AddScriptToExecuteOnDocumentCreated(
                                    L"window.openWithDefaultBrowser = function(url) { window.chrome.webview.postMessage('open_external:' + url); };",
                                    nullptr);

                                webviewWindow->Navigate(url.c_str());
                            }

                            return S_OK;
                        }).Get());
                return S_OK;
            }).Get());
}

void Webview::Resize(const RECT& bounds) {
    if (webviewController) {
        webviewController->put_Bounds(bounds);
    }
}

void Webview::OpenWithDefaultBrowser(const std::wstring& url) {
    ShellExecuteW(nullptr, L"open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

HWND Webview::CreateWin(HINSTANCE hInstance, const std::wstring& title) {
    WNDCLASSW wc = { };
    wc.lpfnWndProc = Webview::WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = title.c_str();
    wc.hIcon = LoadIcon(hInstance, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        0,
        title.c_str(),
        title.c_str(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1600, 960,
        NULL,
        NULL,
        hInstance,
        NULL
    );



    UpdateTitleBarTheme(hwnd);

    return hwnd;
}

LRESULT CALLBACK Webview::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_APP_WEBVIEW_READY: {
        wchar_t* urlPtr = reinterpret_cast<wchar_t*>(lParam);
        if (urlPtr && s_instance) {
            ShowWindow(hwnd, SW_SHOW);
            UpdateWindow(hwnd);
            s_instance->Initialize(hwnd, urlPtr);
        }
        break;
    }
    case WM_SIZE: {
        RECT bounds;
        GetClientRect(hwnd, &bounds);
        if (s_instance) {
            s_instance->Resize(bounds);
        }
        break;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_SETTINGCHANGE:
        UpdateTitleBarTheme(hwnd);
        [[fallthrough]];
    default:
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
    return 0;
}

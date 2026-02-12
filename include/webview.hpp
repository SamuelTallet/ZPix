#pragma once

#include <windows.h>
#include <string>
#include <wrl.h>
#include <WebView2.h>
#define WM_APP_WEBVIEW_READY (WM_APP + 1)


using namespace Microsoft::WRL;

class Webview {
public:
    Webview();
    ~Webview();

    Webview(const Webview&) = delete;
    Webview& operator=(const Webview&) = delete;

    void Initialize(HWND parent, const std::wstring& url);
    void Resize(const RECT& bounds);
    void OpenWithDefaultBrowser(const std::wstring& url);
    std::wstring OpenNativeFileDialog(HWND parent);
    
    HWND CreateWin(HINSTANCE hInstance, const std::wstring& title);
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

private:
    ComPtr<ICoreWebView2Controller> webviewController;
    ComPtr<ICoreWebView2> webviewWindow;
};

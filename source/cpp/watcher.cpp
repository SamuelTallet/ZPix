#include "watcher.hpp"
#include <winsock2.h>
#include <chrono>

void WatcherThread::run(std::stop_token stoken, uint16_t port, std::function<void()> onPortOpen) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return;
    }

    while (!stoken.stop_requested()) {
        SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock == INVALID_SOCKET) break;

        sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        addr.sin_port = htons(port);

        if (connect(sock, (SOCKADDR*)&addr, sizeof(addr)) != SOCKET_ERROR) {
            closesocket(sock);
            WSACleanup();
            if (onPortOpen) onPortOpen();
            return;
        }

        closesocket(sock);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    WSACleanup();
}

WatcherThread::WatcherThread(uint16_t port, std::function<void()> onPortOpen) {
    worker = std::jthread([this, port, onPortOpen](std::stop_token stoken) { this->run(stoken, port, onPortOpen); });
}

WatcherThread::~WatcherThread() {
}

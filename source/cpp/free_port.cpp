#include "free_port.hpp"
#include <winsock2.h>
#include <random>
#include <stdexcept>

#pragma comment(lib, "ws2_32.lib")

uint16_t find_free_port() {
    WSADATA wsaData;
    int col = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (col != 0) {
        throw std::runtime_error("WSAStartup failed");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dist(42000, 65535);

    uint16_t port = 0;
    bool found = false;

    for (int i = 0; i < 36; ++i) {
        uint16_t candidate = dist(gen);
        SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock == INVALID_SOCKET) continue;

        sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        addr.sin_port = htons(candidate);

        if (bind(sock, (SOCKADDR*)&addr, sizeof(addr)) != SOCKET_ERROR) {
            found = true;
            port = candidate;
            closesocket(sock);
            break;
        }

        closesocket(sock);
    }

    WSACleanup();

    if (!found) {
        throw std::runtime_error("No free port found after 36 attempts");
    }

    return port;
}

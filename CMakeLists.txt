cmake_minimum_required(VERSION 4.1)

file(READ "metadata/SAFE_NAME" PROJECT_NAME)
file(READ "metadata/VERSION" PROJECT_VERSION)
file(READ "metadata/DESCRIPTION" DESCRIPTION)

set(DEVELOPER "Samuel Tallet")
set(COPYRIGHT "Copyright 2026, ${DEVELOPER}")

project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

include(resources/resources.cmake)

file(GLOB_RECURSE SOURCES "source/cpp/*.cpp")

add_executable(${PROJECT_NAME} WIN32 main.cpp ${SOURCES})
target_sources(${PROJECT_NAME} PRIVATE ${SHARED_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE include)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-windows")

find_package(unofficial-webview2 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE unofficial::webview2::webview2 ws2_32 dwmapi)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:${PROJECT_NAME}>"
        "${CMAKE_SOURCE_DIR}/${PROJECT_NAME}.exe"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin/WebView2Loader.dll"
        "${CMAKE_SOURCE_DIR}/WebView2Loader.dll"
)

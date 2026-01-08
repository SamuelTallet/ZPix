configure_file("${CMAKE_CURRENT_LIST_DIR}/version.rc.in" "resources/version.rc" @ONLY)
list(APPEND SHARED_SOURCES
    "resources/resources.rc"
    "${CMAKE_CURRENT_BINARY_DIR}/resources/version.rc")

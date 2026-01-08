#pragma once

#include <windows.h>

class JobObject {
public:
    JobObject();
    ~JobObject();

    JobObject(const JobObject&) = delete;
    JobObject& operator=(const JobObject&) = delete;

    void add_process(HANDLE process_handle);

private:
    HANDLE hJob;
};

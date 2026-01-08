#include "processes.hpp"
#include <stdexcept>

JobObject::JobObject() {
    hJob = CreateJobObject(NULL, NULL);
    if (hJob == NULL) {
        throw std::runtime_error("Failed to create Job Object");
    }

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli = { 0 };
    jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

    if (!SetInformationJobObject(hJob, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli))) {
        CloseHandle(hJob);
        throw std::runtime_error("Failed to set Job Object information");
    }
}

JobObject::~JobObject() {
    if (hJob) {
        CloseHandle(hJob);
    }
}

void JobObject::add_process(HANDLE process_handle) {
    AssignProcessToJobObject(hJob, process_handle);
}

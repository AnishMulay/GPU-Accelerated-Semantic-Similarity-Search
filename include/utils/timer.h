// include/utils/timer.h
#pragma once
#include <chrono>
#include <string>

namespace semantic_search {
namespace utils {

class Timer {
public:
    Timer(const std::string& name);
    ~Timer();
    
    void reset();
    double elapsedMilliseconds() const;
    double elapsedSeconds() const;
    
private:
    std::string m_name;
    std::chrono::high_resolution_clock::time_point m_startTime;
    bool m_running;
};

} // namespace utils
} // namespace semantic_search
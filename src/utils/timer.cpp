// src/utils/timer.cpp
#include "utils/timer.h"
#include <iostream>

namespace semantic_search {
namespace utils {

Timer::Timer(const std::string& name) 
    : m_name(name), m_running(true) {
    m_startTime = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    if (m_running) {
        std::cout << m_name << ": " << elapsedMilliseconds() << " ms" << std::endl;
    }
}

void Timer::reset() {
    m_startTime = std::chrono::high_resolution_clock::now();
    m_running = true;
}

double Timer::elapsedMilliseconds() const {
    auto endTime = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(endTime - m_startTime).count();
}

double Timer::elapsedSeconds() const {
    return elapsedMilliseconds() / 1000.0;
}

} // namespace utils
} // namespace semantic_search
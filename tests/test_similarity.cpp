// tests/test_similarity.cpp
#include <iostream>
#include "utils/timer.h"

using namespace semantic_search::utils;

int main() {
    Timer timer("Test Timer");
    // Simulate some work
    for (int i = 0; i < 100000000; i++) {}
    
    std::cout << "Test completed" << std::endl;
    return 0;
}

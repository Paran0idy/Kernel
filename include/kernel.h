#pragma once

#include <cuda.h>
#include <iostream>
#include <fstream>
#include <chrono>

const int warpSize = 32;

class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        return diff.count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

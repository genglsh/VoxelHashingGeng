//
// Created by gengshuai on 19-11-5.
//
#include <Utils.h>

namespace ark{

    void printTime(const system_clock::time_point &a,
                   const system_clock::time_point &b,
                   const std::string &note) {
        auto setTime = duration_cast<std::chrono::microseconds>(b - a).count();
        std::cout << note << (float)setTime * microseconds::period::num / microseconds::period::den
                  << "s" << std::endl;

    }
}
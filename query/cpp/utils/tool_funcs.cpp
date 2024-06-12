#include <iostream>
#include <unordered_set>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>
#include <cassert>

#include "tool_funcs.h"

Counter g_counter;

// std::random_device rd;
// std::mt19937 gen19937(rd());
long g_seed = 2000;
std::mt19937 g_gen19937(g_seed);

Timer::Timer() {
    this->Reset();
}

void Timer::Reset() {
    this->t_ = std::chrono::high_resolution_clock::now(); 
}

// elapse and reset
double Timer::Elapse() {
    double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now() - this->t_).count(); 
    this->Reset();
    duration *= 1e-9; // precision = 9
    return duration; 
}

std::ostream& operator<<(std::ostream& out, Timer& timer) {
    out << std::setprecision(7) << timer.Elapse();
    return out;
}


Counter::Counter() {
}

void Counter::Add(const unsigned short& k, const unsigned int& c) {
    this->count[k] += c;
}

void Counter::Sub(const unsigned short& k, const unsigned int& c) {
    this->count[k] -= c;
}

void Counter::Remove(const unsigned short& k) {
    this->count[k] = 0;
}

void Counter::Clear() {
    this->count.clear();
}

void Counter::Set(const unsigned short& k, const unsigned int& c) {
    this->count[k] = c;
}

unsigned int Counter::Get(const unsigned short& k) {
    return this->count[k];
}


// https://stackoverflow.com/questions/28287138/c-randomly-sample-k-numbers-from-range-0n-1-n-k-without-replacement
std::unordered_set<int> PickSet(int N, int k) {
    assert(k <= N || !(std::cerr << k << "," << N));
    std::unordered_set<int> elems;
    for (int r = N - k; r < N; ++r) {
        int v = std::uniform_int_distribution<>(1, r)(g_gen19937);

        // there are two cases.
        // v is not in candidates ==> add it
        // v is in candidates ==> well, r is definitely not, because
        // this is the first iteration in the loop that we could've
        // picked something that big.

        if (!elems.insert(v).second) {
            elems.insert(r);
        }   
    }
    return elems;
}


float L2(float x1, float y1, float x2, float y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

float L2Squared(float x1, float y1, float x2, float y2) {
    return pow(x2 - x1, 2) + pow(y2 - y1, 2);
}

// https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
std::string ExecCommand(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    // if (!pipe) throw std::runtime_error("popen() failed!");
    if (!pipe) return "";
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        // throw;
        return "";
    }
    pclose(pipe);
    return result;
}


long GetProcRAM(const char* command_name) {
    std::string str_command_name(command_name);
    std::string cmd = "ps -o rss,command ax --sort -rss | grep \"" + str_command_name + "\"";
    std::string str_mem = ExecCommand(cmd.c_str());
    str_mem = str_mem.erase(0, str_mem.find_first_not_of(' '));
    str_mem = str_mem.substr(0, str_mem.find(" "));
    return std::stol(str_mem) / 1024; // in MB
}
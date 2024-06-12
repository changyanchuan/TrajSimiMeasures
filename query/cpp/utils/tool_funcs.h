#ifndef _TOOL_FUNCS_H_
#define _TOOL_FUNCS_H_

#include <iostream>
#include <unordered_set>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <map>

class Timer {
public:
    Timer();

    void Reset();

    double Elapse();

    friend std::ostream& operator<<(std::ostream& out, Timer& timer);

private:
    std::chrono::high_resolution_clock::time_point t_;
};


class Counter;
extern Counter g_counter;
#define CNT_KNN_TRAJ_RETRIVED           11

class Counter {
public:    
    Counter();

    void Add(const unsigned short& k = 0, const unsigned int& c = 1);
    void Sub(const unsigned short& k = 0, const unsigned int& c = 1);
    void Remove(const unsigned short& k);
    void Clear();
    void Set(const unsigned short& k, const unsigned int& c);
    unsigned int Get(const unsigned short& k = 0);

private:
    std::map<unsigned short, unsigned int> count;
};


// extern std::random_device rd;
extern long g_seed;
extern std::mt19937 g_gen19937;

std::unordered_set<int> PickSet(int N, int k);


float L2(float x1, float y1, float x2, float y2);

float L2Squared(float x1, float y1, float x2, float y2);


template <typename T1, typename T2>
bool cmp_pair_1st(const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2) {
    return p1.first < p2.first;
}

template <typename T1, typename T2>
bool cmp_pair_1st_reverse(const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2) {
    return p1.first > p2.first;
}

template <typename T1, typename T2>
bool cmp_pair_2nd(const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2) {
    return p1.second < p2.second;
}

std::string ExecCommand(const char* cmd);

long GetProcRAM(const char* command_name);


#endif
#include <stdio.h>
#include <iostream>
#include <deque>
#include <queue>
#include <mutex>
#include <thread>
#include <lcm/lcm-cpp.hpp>
#include "../lcm_types/cpp/leg_control_data_lcmt.hpp"
#include "../lcm_types/cpp/microstrain_lcmt.hpp"
#include "../lcm_types/cpp/contact_ground_truth_t.hpp"


// mutex for critical section
std::mutex mtx;

std::queue<float *> cnnInputLegQueue;
std::queue<float *> cnnInputIMUQueue;
std::queue<int> cnnInputGtLabelQueue;

std::vector<std::vector<float>> cnnInputMatrix(150, std::vector<float>(54));
int dataRequire = 150;
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h> 
#include <Eigen/Dense>
#include <deque>
#include <queue>
#include <mutex>
#include <thread>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"


#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
// #include <Eigen/Dense>
#include <lcm/lcm-cpp.hpp>
#include "../lcm_types/cpp/leg_control_data_lcmt.hpp"
#include "../lcm_types/cpp/microstrain_lcmt.hpp"
#include "../lcm_types/cpp/contact_ground_truth_t.hpp"


// mutex for critical section
std::mutex mtx;

std::queue<float *> cnnInputLegQueue;
std::queue<float *> cnnInputIMUQueue;
std::queue<int> cnnInputGtLabelQueue;

const int input_h = 150;
const int input_w = 54;
std::vector<std::vector<float>> cnnInputMatrix(input_h, std::vector<float>(input_w));
float cnnInputMatrix_normalized[input_h][input_w];
std::vector<float> mean_vector(input_w, 0);
std::vector<float> std_vector(input_w, 0);
int dataRequire = 150;


int arg_c;
char** arg_v;

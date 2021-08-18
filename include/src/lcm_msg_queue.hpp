#ifndef lcm_msg_queue_h
#define lcm_msg_queue_h

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h> 
// #include <Eigen/Dense>
#include <deque>
#include <queue>
#include <mutex>
#include <thread>
#include <bitset>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

//! \brief contains several queues that will be used to transfer messages in the interface
struct lcmMsgQueues_t {
    std::mutex mtx;
    std::queue<float *> cnn_input_leg_queue;
    std::queue<float *> cnn_input_imu_queue;
    std::queue<int> cnn_input_gtlabel_queue;
    std::queue<int64_t> timestampe_queue;
    // std::queue<float *> cnnInputQueue;    
}


#endif
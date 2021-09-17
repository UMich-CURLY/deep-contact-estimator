#ifndef lcm_msg_queue_h
#define lcm_msg_queue_h

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h> 
// #include <Eigen/Dense>
#include <deque>
#include <queue>
#include <stack>
#include <mutex>
#include <thread>
#include <bitset>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "lcm_data_types.hpp"  

//! \brief contains several queues that will be used to transfer messages in the interface
struct LcmMsgQueues_t {
    std::mutex mtx;
    // Input queue:
    std::queue<std::shared_ptr<LcmLegStruct>> cnn_input_leg_queue;
    std::queue<std::shared_ptr<LcmIMUStruct>> cnn_input_imu_queue;
    // Synced queue:
    std::queue<std::shared_ptr<synced_proprioceptive_lcmt>> synced_msgs_queue;
    std::queue<int> cnn_input_gtlabel_queue;
    std::queue<double> timestamp_queue;

    // Stack for leg data to do synchronization in fast mode:
    std::stack<std::shared_ptr<LcmLegStruct>> cnn_input_leg_stack;    
};


#endif

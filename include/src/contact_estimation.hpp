#ifndef contact_estimation_h
#define contact_estimation_h


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

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "src/lcm_cnn_interface.hpp"
#include "communication/lcm_handler.hpp"
#include "utils/tensorrt_acc.hpp"

#include <lcm/lcm-cpp.hpp>
#include "../lcm_types/cpp/leg_control_data_lcmt.hpp"
#include "../lcm_types/cpp/microstrain_lcmt.hpp"
#include "../lcm_types/cpp/contact_t.hpp"
#include "../lcm_types/cpp/contact_ground_truth_t.hpp"
#include "../lcm_types/cpp/synced_proprioceptive_lcmt.hpp"

class ContactEstimation {
public:
    //!
    //! \brief Initialize necessary variables, such as the TensorRT Engine.
    //!
    ContactEstimation(const samplesCommon::Args &args, lcm::LCM* lcm_, std::mutex* mtx_, int debug_flag_, std::ofstream& myfile_, std::ofstream& myfile_leg_p_, LcmMsgQueues_t* lcm_msg_in_);

    //!
    //! \brief Destroy the class
    //!
    ~ContactEstimation();

    void makeInference(std::queue<float *>& cnnInputQueue, std::queue<float *>& new_data_queue);

	//!
    //! \brief Publish the output to "CNN_OUTPUT" channel
    //!
    void publishOutput(int output_idx);

private:
    int input_h; //!< The number of rows of the input matrix
    int input_w; //!< The number of columns of the input matrix
    TensorRTAccelerator sample; //!< sample contains the engine and other related parameters
    lcm::LCM* lcm;
    synced_proprioceptive_lcmt cnn_output;
    std::mutex* mtx;
    int debug_flag;
    std::ofstream& myfile;
    std::ofstream& myfile_leg_p;
    LcmMsgQueues_t* lcm_msg_in;
};

#endif

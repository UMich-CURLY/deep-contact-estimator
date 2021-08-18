#ifndef lcm_cnn_interface_h
#define lcm_cnn_interface_h

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

#include "src/lcm_handler.hpp"
#include "utils/tensorrt_acc.hpp"
#include "lcm_msg_queue.hpp"


// #include <Eigen/Dense>
#include <lcm/lcm-cpp.hpp>
#include "../lcm_types/cpp/leg_control_data_lcmt.hpp"
#include "../lcm_types/cpp/microstrain_lcmt.hpp"
#include "../lcm_types/cpp/contact_t.hpp"
#include "../lcm_types/cpp/contact_ground_truth_t.hpp"
#include "../lcm_types/cpp/synced_proprioceptive_lcmt.hpp"

// mutex for critical section
std::mutex mtx;
// queues that need to be shared between Handler and LcmCnnInterface:

struct lcmMsgQueues_t {
    std::queue<float *> cnn_input_leg_queue;
    std::queue<float *> cnn_input_imu_queue;
    std::queue<int> cnn_input_gtlabel_queue;
    std::queue<int64_t> timestampe_queue;
    // std::queue<float *> cnnInputQueue;    
}

//!
//! \brief The LcmCnnInterface class takes in pre-processed data from queues and
//! send input to a deserialized TensorRT Engine to make inference
//!
class LcmCnnInterface
{
public:
    //!
    //! \brief Initialize necessary variables, such as the TensorRT Engine.
    //!
    LcmCnnInterface(const samplesCommon::Args &args);
    
    //!
    //! \brief Destroy the class
    //!
    ~LcmCnnInterface();
    
    //!
    //! \brief Takes in preprocessed data from queues and build a 2D matrix
    //! with size of input_h x input_w
    //!
    void buildMatrix(std::queue<float *>& cnnInputQueue, std::queue<float *>& new_data_queue);

    //!
    //! \brief Normalize the matrix and change it into an 1D array, then make inference
    //!
    void normalizeAndInfer(std::queue<float *>& cnnInputQueue);
    
    //!
    //! \brief When the current input matrix is the first full dimension matrix we have, we need to
    //! run a full calculation for mean value and std. After that, we can use sliding
    //! window to update mean value and std.
    //!
    void runFullCalculation(std::queue<float *>& cnnInputQueue);

    //!
    //! \brief Use sliding window to find current mean and std and normalize the matrix.
    //!
    void runSlidingWindow(std::queue<float *>& cnnInputQueue);

private:
    int input_h; //!< The number of rows of the input matrix
    int input_w; //!< The number of columns of the input matrix
    std::vector<float> new_line; //!< The latest data
    std::vector<std::vector<float>> cnn_input_matrix; //!< input_matrix as a 2D matrix before normalization
    // float* cnn_input_matrix_normalized; //!< input_matrix as an 1D array after normalization
    std::vector<float> mean_vector; //!< mean value of each column
    std::vector<float> std_vector; //!< standard deviation of each column
    int data_require; //!< The number of data required to start the first inference
    // TensorRTAccelerator sample; //!< sample contains the engine and other related parameters
    std::vector<float> sum_of_rows; //!< the sum of elements in the same column;
    std::vector<float> sum_of_rows_square; //!< the sum of the square of elements in the same column;
    std::vector<float> previous_first_row; //!< save the value in previous row;
    bool is_first_full_matrix; //!< indicates whether the current matrix is the first full matrix

};

#endif
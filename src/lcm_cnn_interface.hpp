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
// #include <Eigen/Dense>
#include <lcm/lcm-cpp.hpp>
#include "../lcm_types/cpp/leg_control_data_lcmt.hpp"
#include "../lcm_types/cpp/microstrain_lcmt.hpp"
#include "../lcm_types/cpp/contact_t.hpp"
#include "../lcm_types/cpp/contact_ground_truth_t.hpp"

// mutex for critical section
std::mutex mtx;
// queues that need to be shared between Handler and LcmCnnInterface:
std::queue<float *> cnnInputLegQueue;
std::queue<float *> cnnInputIMUQueue;
std::queue<int> cnnInputGtLabelQueue;
// std::ofstream myfile;
std::string PROGRAM_PATH = "/media/jetson256g/code/LCM_CNN_INTERFACE/deep-contact-estimator/";

std::vector<float *> cnn_input_leg_vector;
int latest_idx = -1;
//!
//! \brief The Handler class takes in LCM messages from subscribed channels and process them
//! 
class Handler
{
public:
    ~Handler();

    //!
    //! \brief Receives messages from the "leg_control_data" channel and  
    //! stores them in an array, then put the array pointer into a queue
    //!
    void receiveLegControlMsg(const lcm::ReceiveBuffer* rbuf,
                                 const std::string& chan, 
                                 const leg_control_data_lcmt* msg);

    //!
    //! \brief Receives messages from the "microstrain" channel and  
    //! stores them in an array, then put the array pointer into a queue
    //!
    void receiveMicrostrainMsg(const lcm::ReceiveBuffer* rbuf,
                                 const std::string& chan, 
                                 const microstrain_lcmt* msg);

    //!
    //! \brief Receives messages from the "contact_ground_truth" channel and  
    //! stores it in an array, then put the array pointer into a queue
    //!
    void receiveContactGroundTruthMsg(const lcm::ReceiveBuffer* rbuf,
                                          const std::string& chan, 
                                          const contact_ground_truth_t* msg);

private:
    //! \brief Copy values from array2 to array1
    //! 
    //! \details Pass the begin pointer of array1 and array2 to copy values in array2 
    //! to array1 starting from the pointer
    //! 
    void arrayCopy(float array1 [], const float array2 [], int size);
};


//! \brief  The TensorRTAccelerator class implements the pre-trained CNN model
//!
//! \details It creates the network using an ONNX model. 
//!
class TensorRTAccelerator
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    //!
    //! \brief The constructor takes in parameters of the ONNX model
    //!
    TensorRTAccelerator(const samplesCommon::OnnxSampleParams& params);
        
    //!
    //! \brief Destroy the class
    //!
    ~TensorRTAccelerator();
    
    //!
    //! \brief Function builds the network engine from an ONNX model
    //!
    bool buildFromONNXModel();

    //!
    //! \brief Function builds the network engine from a serialized engine from disk
    //!
    bool buildFromSerializedEngine();
    
    //!
    //! \brief Runs the TensorRT inference engine for the given input, and publish the output
    //! to "CNN_OUTPUT" channel
    //!
    //! \details This function is the main execution function of the sample. It allocates the buffer,
    //!          sets inputs and executes the engine.
    //!
    bool inferAndPublish(float* cnnInputMatrix_normalized);

    //!
    //! \brief Serialize the TensorRT engine and save it to disk for later use. The engine can be
    //!  deserialized in buildFromSerialiezedEngine().
    //!
    bool serialize();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify
    lcm::LCM lcm;
    contact_t cnn_output;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    SampleUniquePtr<nvinfer1::IExecutionContext> context;

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    //! \return Whether the preprocess on the input is successful
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, const float* cnnInputMatrix_normalized);

    //!
    //! \brief Classifies digits and verify result
    //!
    //! \return The status of leg in, ranges from [0, 15]
    //!
    int getOutput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Publish the output to "CNN_OUTPUT" channel
    //!
    void publishOutput(int output_idx);
};

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
    void buildMatrix();

    //!
    //! \brief Normalize the matrix and change it into an 1D array, then make inference
    //!
    void normalizeAndInfer();

    //!
    //! \brief When the current input matrix is the first full dimension matrix we have, we need to
    //! run a full calculation for mean value and std. After that, we can use sliding
    //! window to update mean value and std.
    //!
    void runFullCalculation();

    //!
    //! \brief Use sliding window to find current mean and std and normalize the matrix.
    //!
    void runSlidingWindow();

private:
    int input_h; //!< The number of rows of the input matrix
    int input_w; //!< The number of columns of the input matrix
    std::vector<float> new_line; //!< The latest data
    std::vector<std::vector<float>> cnn_input_matrix; //!< input_matrix as a 2D matrix before normalization
    float* cnn_input_matrix_normalized; //!< input_matrix as an 1D array after normalization
    std::vector<float> mean_vector; //!< mean value of each column
    std::vector<float> std_vector; //!< standard deviation of each column
    int data_require; //!< The number of data required to start the first inference
    TensorRTAccelerator sample; //!< sample contains the engine and other related parameters
    std::vector<float> sum_of_rows; //!< the sum of elements in the same column;
    std::vector<float> sum_of_rows_square; //!< the sum of the square of elements in the same column;
    std::vector<float> previous_first_row; //!< save the value in previous row;
    bool is_first_full_matrix; //!< indicates whether the current matrix is the first full matrix
};

#endif

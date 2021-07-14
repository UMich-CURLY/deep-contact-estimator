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

std::queue<float *> cnnInputLegQueue;
std::queue<float *> cnnInputIMUQueue;
std::queue<int> cnnInputGtLabelQueue;

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
    void receive_leg_control_msg(const lcm::ReceiveBuffer* rbuf,
                                 const std::string& chan, 
                                 const leg_control_data_lcmt* msg);

    //!
    //! \brief Receives messages from the "microstrain" channel and  
    //! stores them in an array, then put the array pointer into a queue
    //!
    void receive_microstrain_msg(const lcm::ReceiveBuffer* rbuf,
                                 const std::string& chan, 
                                 const microstrain_lcmt* msg);

    //!
    //! \brief Receives messages from the "contact_ground_truth" channel and  
    //! stores it in an array, then put the array pointer into a queue
    //!
    void receive_contact_ground_truth_msg(const lcm::ReceiveBuffer* rbuf,
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


class MatrixBuilder
{
public:
    MatrixBuilder(const samplesCommon::Args &args);

    ~MatrixBuilder();

    void BuildMatrix();

    void SendCNNOutput();

private:
    int input_h;
    int input_w;
    std::vector<std::vector<float>> cnnInputMatrix;
    float* cnnInputMatrix_normalized;
    std::vector<float> mean_vector;
    std::vector<float> std_vector;
    int data_require;
    TensorRTAccelerator sample;
};

#endif
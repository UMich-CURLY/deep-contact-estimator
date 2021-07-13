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

class OnnxToTensorRT
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    //! \brief  The OnnxToTensorRT class implements the trained ONNX sample
    //!
    //! \details It creates the network using an ONNX model
    //!
    OnnxToTensorRT(const samplesCommon::OnnxSampleParams& params);

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(float* cnnInputMatrix_normalized);


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
    bool processInput(const samplesCommon::BufferManager& buffers, const float* cnnInputMatrix_normalized);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
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
    OnnxToTensorRT sample;
};

#endif
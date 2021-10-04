#ifndef tensorrt_acc_h
#define tensorrt_acc_h

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

#include "yaml-cpp/yaml.h"

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
    int infer(float* cnnInputMatrix_normalized);


    //!
    //! \brief Default infer function offered by TensorRT
    //!
    //! \details This function is the main execution function of the sample. It allocates the buffer,
    //!          sets inputs and executes the engine.
    //!
    bool infer();
    

    //!
    //! \brief Reads the input and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);
    
    //!
    //! \brief Classifies digits and verify result
    //!
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
    
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

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    SampleUniquePtr<nvinfer1::IExecutionContext> context;

    //!
    //! \brief Parses an ONNX model for Contact Estimtor and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

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


    // Configs:
    YAML::Node config_;
    std::string PROGRAM_PATH;
    std::string engine_save_path;
    std::string engine_load_path;

    // Input / Output size:
    int inputH;
    int inputW;
    int outputSize;
};

samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args &args);
#endif

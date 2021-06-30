#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <numeric>


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override{
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;


// Importing An ONNX Model Using the C++ Parser API:
nvinfer1::IBuilder * builder = createInferBuilder(gLogger);
const auto explicitBatch = 1U << static_cast<uint32_t>(nvInver1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  
nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

nvonnxparser::IParser* parser = 
nvonnxparser::createParser(*network, gLogger);

parser->parseFromFile(onnx_filename, 
ILogger::Severity::kWARNING);

// Build an engine -----------------------------------------------------------
builder->setMaxBatchSize(maxBatchSize);
IBuilderConfig* config = builder->createBuilderConfig();
config->setMaxWorkspaceSize(1 << 20);
ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

// Dispense with the network, builder, and parser if using one-----------------
parser->destroy();
network->destroy();
config->destroy();
builder->destroy();


// Serializing a model in C++--------------------------------------------------
// Run the builder as a prior offline step and then serialize:
IHostMemory *serializedModel = engine->serialize();
// store model to disk
// <…>
serializedModel->destroy();

// Performing inference in C++ ------------------------------------------------

//
IExecutionContext *context = engine->createExecutionContext();

//
int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

// Using these indices, set up a buffer array pointing to the input
// and output buffers on the GPU:
void* buffers[2];
buffers[inputIndex] = inputbuffer;
buffers[outputIndex] = outputBuffer;

// TensorRT execution is typically asynchronous, 
// so enqueue the kernels on a CUDA stream:
context->enqueue(batchSize, buffers, stream, nullptr);

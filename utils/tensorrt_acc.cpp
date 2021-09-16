#include "utils/tensorrt_acc.hpp"

const std::string gSampleName = "TensorRT.sample_onnx";

TensorRTAccelerator::TensorRTAccelerator(const samplesCommon::OnnxSampleParams &params)
    : mParams(params),
      mEngine(nullptr)
{
    char resolved_path[PATH_MAX];
    realpath("../", resolved_path);
    std::cout << resolved_path << std::endl;
    config_ = YAML::LoadFile(std::string(resolved_path) + "/config/interface.yaml");
    PROGRAM_PATH = config_["program_path"].as<std::string>();
}

TensorRTAccelerator::~TensorRTAccelerator(){};

bool TensorRTAccelerator::buildFromONNXModel()
{
    /// REMARK: to load a new onnx model, uncommon the following lines until the next <REMARK:> script
    // -----------------------------------------------------------------------------------------------------------------------
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

        if (!mEngine)
    {
        return false;
    }

    /// REMARK: the following can be used to find input/output dimension
    // -----------------------------------------------------------------------------------------------------------------------
    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 2);
    // -----------------------------------------------------------------------------------------------------------------------

    std::cout << "Successfully built the engine" << std::endl;

    return true;
}


bool TensorRTAccelerator::buildFromSerializedEngine()
{
    /// REMARK: we can deserialize a serialized engine if we have one:
    // -----------------------------------------------------------------------------------------------------------------------
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(sample::gLogger);
    std::string cached_path = PROGRAM_PATH + "engines/0730_2blocks_best_val_loss.trt";
    std::ifstream fin(cached_path);
    std::string cached_engine = "";
    while (fin.peek() != EOF)
    {
        std::stringstream buffer;
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr),
        samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    std::cout << "Successfully built the engine and made the context" << std::endl;

    return true;
}


//!
//! \brief Uses a ONNX parser to create the Onnx Contact Estimtor Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx Contact Estimtor network
//!
//! \param builder Pointer to the engine builder
//!
bool TensorRTAccelerator::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(
        locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    /// REMARK: if you want to make the inference faster but less accurate:
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}


int TensorRTAccelerator::infer(float *cnn_input_matrix_normalized)
{
    // Create RAII buffer manager object

    if (!mEngine)
    {
        std::cerr << "Failed to load mEngine" << std::endl;
        return false;
    }

    samplesCommon::BufferManager buffers(mEngine);

    // Read the input data into the managed buffers

    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, cnn_input_matrix_normalized))
    {
        std::cerr << "Failed in reading input" << std::endl;
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        std::cerr << "Failed in making execution" << std::endl;
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Get results from the engine and return the output
    return getOutput(buffers);
}

bool TensorRTAccelerator::infer()
{
    // Create RAII buffer manager object
    if (!mEngine) {
        std::cerr << "Failed to load mEngine" << std::endl;
    }
    samplesCommon::BufferManager buffers(mEngine);
    std::cout << "Successfuly built the buffer" << std::endl;
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    std::cout << "Successfully build an execution context" << std::endl;
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        std::cerr << "Failed in reading input" << std::endl;
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        std::cerr << "Failed in making execution" << std::endl;
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}


bool TensorRTAccelerator::serialize()
{
    nvinfer1::IHostMemory *serializedModel = mEngine->serialize();
    std::string serialize_str;
    std::ofstream serialize_output_stream;
    serialize_str.resize(serializedModel->size());
    memcpy((void *)serialize_str.data(), serializedModel->data(), serializedModel->size());
    serialize_output_stream.open(PROGRAM_PATH + "engines/0730_2blocks_best_val_loss.trt");

    serialize_output_stream << serialize_str;
    serialize_output_stream.close();
    serializedModel->destroy();

    std::cout << "Successfully serialized the engine" << std::endl;
    return true;
}


//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool TensorRTAccelerator::processInput(const samplesCommon::BufferManager &buffers, const float *cnn_input_matrix_normalized)
{
    const int inputH = 75;
    const int inputW = 54;

    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    int number_of_items = 75 * 54;
    // hostDataBuffer.resize(number_of_items);
    for (int i = 0; i < inputH * inputW; i++)
    {
        // std::cout <<  cnn_input_matrix_normalized[i] << std::endl;
        hostDataBuffer[i] = cnn_input_matrix_normalized[i];
    }

    return true;
}

int TensorRTAccelerator::getOutput(const samplesCommon::BufferManager &buffers)
{
    const int outputSize = 16; // 4 legs, 16 status

    float *output = static_cast<float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};

    float current_max = output[0];
    int output_idx = 0;
    for (int i = 1; i < outputSize; i++)
    {
        // sample::gLogInfo << "Leg status " << i << " is " << output[i] << std::endl;
        if (output[i] > current_max)
        {
            current_max = output[i];
            output_idx = i;
        }
    }
    return output_idx;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool TensorRTAccelerator::processInput(const samplesCommon::BufferManager& buffers)
{   
    /// REMARK: if you don't know the input dimension, you can find it by parsing the ONNX model directly;
    /// You cannot find the dimension if you use a serialized engine.
    const int inputH = mInputDims.d[1];
    std::cout << "inputH is: " << inputH << std::endl;

    const int inputW = mInputDims.d[2];
    std::cout << "inputW is: " << inputW << std::endl;

    /// REMARK: after you get the input dimension, you can define them here:
    // const int inputH = 75;
    // const int inputW = 54;
    
    /// REMARK: use a *.bin file to parse the model
    std::vector<uint8_t> fileData(inputH * inputW);
    std::ifstream data_file;
    data_file.open((locateFile("input_matrix_500Hz.bin", mParams.dataDirs)), std::ios::in | std::ios::binary);
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    int number_of_items = 75 * 54;
    // hostDataBuffer.resize(number_of_items);
    data_file.read(reinterpret_cast<char*>(&hostDataBuffer[0]), number_of_items * sizeof(float));
    data_file.close(); 


    return true;
}
//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool TensorRTAccelerator::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    /// REMARK: if you don't know the output dimension, you can find it by parsing the ONNX model directly;
    /// You cannot find the dimension if you use a serialized engine.
    const int outputSize = mOutputDims.d[1];
    std::cout << "outputSize is " << outputSize << std::endl;

    // const int outputSize = 16;
    
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};

    sample::gLogInfo << "Output: " << std::endl;
    float current_max = output[0];
    int output_idx = 0;
    for (int i = 0; i < outputSize; i++) {
        sample::gLogInfo << "Probability of leg status " << i << " before normalization is: " << output[i] << std::endl;
        if (output[i] > current_max) {
            current_max = output[i];
            output_idx = i;
        }
    }
    std::cout << "OUTPUT: " << output_idx << std::endl;
    return true;
}


//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args &args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        std::cout << "Using default directory" << endl;
        params.dataDirs.push_back("weights/");
        params.dataDirs.push_back("data/");
    }
    else //!< Use the data directory provided by the user
    {
        std::cout << "Using directory provided by the user" << endl;
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "0730_2blocks_best_val_loss.onnx";
    params.inputTensorNames.push_back("input");
    params.batchSize = 1; //!< Takes in 1 batch every time
    params.outputTensorNames.push_back("output");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}


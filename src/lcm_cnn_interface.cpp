#include "lcm_cnn_interface.hpp" 


class Handler 
{
public:
    ~Handler() {}
    
    void receive_leg_control_msg(const lcm::ReceiveBuffer* rbuf,
                                    const std::string& chan, 
                                    const leg_control_data_lcmt* msg)
    {
        // std::cout << ++count << std::endl;
        // printf("Received message on channel \"%s\":\n", chan.c_str());
        int size = 12;
        float* leg_control_data = new float[48]();
        arrayCopy(leg_control_data, msg->q, size);
        arrayCopy(leg_control_data + size, msg->qd, size);
        arrayCopy(leg_control_data + size + size, msg->p, size);
        arrayCopy(leg_control_data + size + size + size, msg->v, size);

        cnnInputLegQueue.push(leg_control_data);
    }

    void receive_microstrain_msg(const lcm::ReceiveBuffer* rbuf,
            const std::string& chan, 
            const microstrain_lcmt* msg)
    {
        // printf("Received message on channel \"%s\":\n", chan.c_str());
        float* microstrain_data = new float[6]();
        int size = 3;
        arrayCopy(microstrain_data, msg->acc, size);
        arrayCopy(microstrain_data + size, msg->omega, size);

        cnnInputIMUQueue.push(microstrain_data);
    }

    void receive_contact_ground_truth_msg(const lcm::ReceiveBuffer* rbuf,
            const std::string& chan, 
            const contact_ground_truth_t* msg)
    {
        // printf("Received message on channel \"%s\":\n", chan.c_str());
        std::vector<int8_t> contact_ground_truth_label = msg->contact;
        
        int gt_label = contact_ground_truth_label[0] * 2 * 2 * 2 
                    + contact_ground_truth_label[1] * 2 * 2
                    + contact_ground_truth_label[2] * 2
                    + contact_ground_truth_label[3];
        // std::cout << gt_label << std::endl;

        cnnInputGtLabelQueue.push(gt_label);
    }

private:
    
    // Pass the begin pointer of array1 and array2 to copy array2 to
    // array1 from a certain index.
    void arrayCopy(float array1 [], const float array2 [], int size) {
        for (int i = 0; i < size; ++i) {
            array1[i] = array2[i];
        }
    }

};

const std::string gSampleName = "TensorRT.sample_onnx";

//! \brief  The OnnxToTensorRT class implements the trained ONNX sample
//!
//! \details It creates the network using an ONNX model
//!
class OnnxToTensorRT
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    OnnxToTensorRT(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

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

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, const float* cnnInputMatrix_normalized);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx network by parsing the Onnx model and builds
//!          the engine that will be used to run (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool OnnxToTensorRT::build()
{

    /// REMARK: we can deserialize a serialized engine if we have one:
    // -----------------------------------------------------------------------------------------------------------------------
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    std::string cached_path = "/home/tingjun/Desktop/Cheetah_Proj/deep-contact-estimator/engines/0616_2blocks_best_val_loss.trt";
    std::ifstream fin(cached_path);
    std::string cached_engine = "";
    while (fin.peek() != EOF) {
        std::stringstream buffer;
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine> (
                                    runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr),
                                    samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    // std::cout << "Successfully built the engine" << std::endl;

    return true;
}


//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool OnnxToTensorRT::infer(float* cnnInputMatrix_normalized)
{
    // Create RAII buffer manager object
    if (!mEngine) {
        std::cerr << "Failed to load mEngine" << std::endl;
    }
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
    // std::cout << "Successfuly built the buffer" << std::endl;
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    // std::cout << "Successfully build an execution context" << std::endl;
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, cnnInputMatrix_normalized))
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

bool OnnxToTensorRT::serialize() {
    nvinfer1::IHostMemory *serializedModel = mEngine->serialize();
    std::string serialize_str;
    std::ofstream serialize_output_stream;
    serialize_str.resize(serializedModel->size());
    memcpy((void*)serialize_str.data(), serializedModel->data(), serializedModel->size());
    serialize_output_stream.open("/home/tingjun/Desktop/mini_cheetah/deep-contact-estimator/engines/0616_2blocks_best_val_loss.trt");
    
    serialize_output_stream << serialize_str;
    serialize_output_stream.close();
    serializedModel->destroy();
    
    std::cout << "Successfully serialized the engine" << std::endl;
    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool OnnxToTensorRT::processInput(const samplesCommon::BufferManager& buffers, const float* cnnInputMatrix_normalized)
{   
    const int inputH = 150;
    const int inputW = 54;
    
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    int number_of_items = 150 * 54;
    // hostDataBuffer.resize(number_of_items);
    for (int i = 0; i < inputH * inputW; i++) {
        // std::cout <<  cnnInputMatrix_normalized[i] << std::endl;
        hostDataBuffer[i] = cnnInputMatrix_normalized[i];
    }


    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool OnnxToTensorRT::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = 16;
    
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};

    float current_max = output[0];
    int output_idx = 0;
    for (int i = 1; i < outputSize; i++) {
        // gLogInfo << "Leg status " << i << " is " << output[i] << std::endl;
        if (output[i] > current_max) {
            current_max = output[i];
            output_idx = i;
        }
    }
    std::cout << "CNN output is: " << output_idx << std::endl;
    return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        // std::cout << "Using default directory" << endl;
        params.dataDirs.push_back("weights/");
        params.dataDirs.push_back("data/");
        // params.dataDirs.push_back("data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        // std::cout << "Using directory provided by the user" << endl;
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "0616_2blocks_best_val_loss.onnx";
    params.inputTensorNames.push_back("input");
    params.batchSize = 1;
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


MatrixBuilder::MatrixBuilder()
{
    input_h = 150;
    input_w = 54;
    dataRequire = 150;
    cnnInputMatrix = std::vector<std::vector<float>> (input_h, std::vector<float>(input_w));
    cnnInputMatrix_normalized = new float[input_h * input_w];
    mean_vector = std::vector<float> (input_w, 0);
    std_vector = std::vector<float> (input_w, 0);
}

MatrixBuilder::~MatrixBuilder() {
    delete[] cnnInputMatrix_normalized;
}

void MatrixBuilder::BuildMatrix (){
        // Get leg input from queue
        while (true){
            if (!cnnInputLegQueue.empty() && !cnnInputIMUQueue.empty() && !cnnInputGtLabelQueue.empty()){
                mtx.lock();
                // Get GTlabel from queue
                int gtLabel = cnnInputGtLabelQueue.front();
                cnnInputGtLabelQueue.pop();
                mtx.unlock();
                // Start to build a new line and generate a new input
                std::vector<float> newLine(54);
                int idx = 0; // keep track of the current newLine idx;
                int legTypeDataNum = 12;
                int IMUTypeDataNum = 3;
                
                // get input data:
                mtx.lock();
                for (int i = 0; i < legTypeDataNum; ++i){
                    newLine[idx++] = cnnInputLegQueue.front()[i];
                }
                for (int i = 0; i < legTypeDataNum; ++i)
                    newLine[idx++] = cnnInputLegQueue.front()[i + legTypeDataNum];
                
                for (int i = 0; i < IMUTypeDataNum; ++i)
                    newLine[idx++] = cnnInputIMUQueue.front()[i];

                for (int i = 0; i < IMUTypeDataNum; ++i)
                    newLine[idx++] = cnnInputIMUQueue.front()[i + IMUTypeDataNum];

                for (int i = 0; i < legTypeDataNum; ++i)
                    newLine[idx++] = cnnInputLegQueue.front()[i + legTypeDataNum + legTypeDataNum];

                for (int i = 0; i < legTypeDataNum; ++i)
                    newLine[idx++] = cnnInputLegQueue.front()[i + legTypeDataNum + legTypeDataNum + legTypeDataNum];
                
                // release memory:
                delete[] cnnInputLegQueue.front();
                delete[] cnnInputIMUQueue.front();
                cnnInputLegQueue.pop();
                cnnInputIMUQueue.pop();
                mtx.unlock();

                // Put the newLine to the InputMatrix and destroy the first line:
                cnnInputMatrix.erase(cnnInputMatrix.begin());
                cnnInputMatrix.push_back(newLine);
                dataRequire = std::max(dataRequire - 1, 0);
                if (dataRequire == 0) {
                    /// REMARK: send to CNN network in TRT
                    // We need to normalize the input matrix, to do so,
                    // we need to calculate the mean value and standard
                    // deviation.
                    
                    for (int j = 0; j < input_w; ++j) {
                        for (int i = 0; i < input_h; ++i) {
                            mean_vector[j] += cnnInputMatrix[i][j];
                        }
                        mean_vector[j] = mean_vector[j] / input_h;

                        for (int i = 0; i < input_h; ++i) {
                            std_vector[j] += std::pow((cnnInputMatrix[i][j] - mean_vector[j]), 2.0);
                        }
                        std_vector[j] = std::sqrt(std_vector[j] / (input_h - 1));

                        // Normalize the matrix:
                        for (int i = 0; i < input_h; ++i) {
                            cnnInputMatrix_normalized[i * input_w + j] = (cnnInputMatrix[i][j] - mean_vector[j]) / std_vector[j];
                        }
                    }
                    /// REMARK: write std::vector to a file:
                    // std::ofstream output;
                    // output.open("/home/tingjun/Desktop/TensorRT_PROJECT_USE/data/input_matrix.bin", std::ios::out | std::ios::binary);
                    // size_t size_row = cnnInputMatrix_normalized.size();

                    // for (size_t i = 0; i < size_row; ++i) {
                    //     output.write(reinterpret_cast<char*>(&cnnInputMatrix_normalized[i][0]), input_w * sizeof(float));
                    // }

                    // output.close();
                    std::cout << "The ground truth label is: " << gtLabel << std::endl;

                    /// REMARK: Inference
                    samplesCommon::Args args;
                    bool argsOK = samplesCommon::parseArgs(args, arg_c, arg_v);

                    if (!argsOK)
                    {
                        gLogError << "Invalid arguments" << std::endl;
                        printHelpInfo();
                        return;
                    }
                    if (args.help)
                    {
                        printHelpInfo();
                        return;
                    }

                    OnnxToTensorRT sample(initializeSampleParams(args));
                    // gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;
                    if (!sample.build())
                    {
                        std::cerr << "FAILED: Cannot build the engine" << std::endl;
                        return;
                    }
                    if (!sample.infer(&cnnInputMatrix_normalized[0]))
                    {
                        std::cerr << "FAILED: Cannot use the engine to infer a result" << std::endl;
                        return;
                    }
                }
            } 
        }
    }



int main(int argc, char** argv)
{
    arg_c = argc;
    arg_v = argv;

    /// LCM: subscribe to channels:
    lcm::LCM lcm;
    if(!lcm.good())
        return 1;
    Handler handlerObject;
    lcm.subscribe("leg_control_data", &Handler::receive_leg_control_msg, &handlerObject);
    lcm.subscribe("microstrain", &Handler::receive_microstrain_msg, &handlerObject);
    lcm.subscribe("contact_ground_truth", &Handler::receive_contact_ground_truth_msg, &handlerObject);
    
    std::cout << "Start Running LCM-CNN Interface" << std::endl;
    MatrixBuilder matrix_builder;
    std::thread MatrixThread (&MatrixBuilder::BuildMatrix, &matrix_builder);
    while(0 == lcm.handle());
    MatrixThread.join();

    // return gLogger.reportPass(sampleTest);

    return 0;
}

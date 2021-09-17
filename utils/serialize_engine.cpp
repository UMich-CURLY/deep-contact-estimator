#include "utils/tensorrt_acc.hpp"
// #include "src/config.hpp"

void printHelpInfo();

int main(int argc, char** argv) 
// argc is 1 + the number of arguments
// *argv is the execution line without arguments
{
    const std::string gSampleName = "TensorRT.contact_estimator_model";
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);

    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    TensorRTAccelerator sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for contact estimator ONNX model" << std::endl;

    if (!sample.buildFromONNXModel())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    /// REMARK: if you want to serialize the engine, please uncommon the following line:
    /// Once you have a serialized model, you don't need to do it again. You can directly
    /// use the serialized model in build;
    if (!sample.serialize())
    {
        std::cerr << "Failed to serialize" << std::endl;
    }

    return sample::gLogger.reportPass(sampleTest);
}

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
    std::cout << "--int8          Run in Int8 mode. Used in serializing an engine" << std::endl;
    std::cout << "--fp16          Run in FP16 mode. Used in serializing an engine" << std::endl;
}

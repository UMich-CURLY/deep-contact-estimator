#include "src/lcm_cnn_interface.hpp"
#include "src/contact_estimation.hpp"
#include <stdlib.h>
#include <typeinfo>
// #include "src/config.hpp"

void printHelpInfo();

int main(int argc, char **argv)
{
    /// LCM: subscribe to channels:
    lcm::LCM lcm;
    if (!lcm.good())
        return 1;
    
    char resolved_path[PATH_MAX];
    realpath("../", resolved_path);
    std::cout << resolved_path << std::endl;
    YAML::Node config_ = YAML::LoadFile(std::string(resolved_path) + "/config/interface.yaml");

    LcmMsgQueues_t lcm_msg_in;
    std::mutex mtx;
    std::string mode = config_["mode"].as<std::string>();
    LcmHandler handlerObject(&lcm, &lcm_msg_in, &mtx, mode);

    std::cout << "Start Running LCM-CNN Interface" << std::endl;

    // Takes input arguments
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return -1;
    }
    if (args.help)
    {
        printHelpInfo();
        return -1;
    }

    /// INTERFACE: use multiple threads to avoid missing messages:
    std::queue<float *> cnn_input_queue;
    std::queue<float *> new_data_queue;

    std::ofstream myfile;
    std::ofstream myfile_leg_p;
    int debug_flag = config_["debug_flag"].as<int>();
    std::cout << "debug_flag: " << debug_flag << std::endl;
    std::string PROGRAM_PATH = config_["program_path"].as<std::string>();
    int input_w = config_["input_w"].as<int>();
    int input_h = config_["input_h"].as<int>();
    int num_legs = config_["num_legs"].as<int>();


    if (debug_flag == 1)
    {
       myfile.open(PROGRAM_PATH + "contact_est_lcm.csv");
       myfile_leg_p.open(PROGRAM_PATH + "p_lcm.csv");
    }

    LcmCnnInterface matrix_builder(args, &lcm_msg_in, &mtx, debug_flag, myfile_leg_p, &config_, input_h, input_w);
    ContactEstimation engine_builder(args, &lcm, &mtx, debug_flag, myfile, myfile_leg_p, &lcm_msg_in, &config_, input_h, input_w, num_legs);
    std::thread BuildMatrixThread(&LcmCnnInterface::buildMatrix, &matrix_builder, std::ref(cnn_input_queue), std::ref(new_data_queue));
    std::thread CNNInferenceThread(&ContactEstimation::makeInference, &engine_builder, std::ref(cnn_input_queue), std::ref(new_data_queue));

    std::cout << "started thread" << std::endl;

    while (0 == lcm.handle());
    
    BuildMatrixThread.join();
    CNNInferenceThread.join();

    if (debug_flag == 1) {
        myfile.close();
        myfile_leg_p.close();
    }

    return 0;
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
    std::cout << "--int8          Run in Int8 mode. Used in serializing an engine" << std::endl;
    std::cout << "--fp16          Run in FP16 mode. Used in serializing an engine" << std::endl;
}
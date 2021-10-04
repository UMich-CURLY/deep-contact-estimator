#include "src/contact_estimation.hpp"
#include "utils/tensorrt_acc.hpp"
#include "src/lcm_cnn_interface.hpp"
#include <stdlib.h>

ContactEstimation::ContactEstimation(const samplesCommon::Args &args, lcm::LCM* lcm, std::mutex* mtx, 
                                    int debug_flag, std::ofstream& myfile, std::ofstream& myfile_leg_p, 
                                    LcmMsgQueues_t* lcm_msg_in, YAML::Node* config, const int input_h, const int input_w,
                                    const int num_legs)
    : config_(config),
      input_h_(input_h),
      input_w_(input_w),
      num_legs_(num_legs),
      sample(initializeSampleParams(args)),
      lcm_(lcm),
      mtx_(mtx),
      debug_flag_(debug_flag),
      myfile_(myfile),
      myfile_leg_p_(myfile_leg_p),
      lcm_msg_in_(lcm_msg_in)      
{
    if (!sample.buildFromSerializedEngine())
    {
        std::cerr << "FAILED: Cannot build the engine" << std::endl;
        return;
    }
    if (!lcm_->good())
        return;
}

ContactEstimation::~ContactEstimation() {}

void ContactEstimation::makeInference(std::queue<float *> &cnn_input_queue, std::queue<float *> &new_data_queue)
{
    while (true)
    {
        if (!cnn_input_queue.empty())
        {
            /// REMARK: Inference (Here we added a timer to calculate the inference frequency)
            /*
    	    cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        	*/
            mtx_->lock();

            float *cnn_input_matrix_normalized = cnn_input_queue.front();
            cnn_input_queue.pop();
            std::shared_ptr<synced_proprioceptive_lcmt> synced_msgs = lcm_msg_in_->synced_msgs_queue.front();
            lcm_msg_in_->synced_msgs_queue.pop();
            
            mtx_->unlock();

            int output_idx = sample.infer(cnn_input_matrix_normalized);
            if (output_idx == -1)
            {
                std::cerr << "FAILED: Cannot use the engine to infer a result" << std::endl;
                return;
            }

            delete[] cnn_input_matrix_normalized;
            publishOutput(output_idx, synced_msgs);

            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            // float milliseconds = 0;
            // cudaEventElapsedTime(&milliseconds, start, stop);
            // std::cout << "It's frequency is " << 1000 / milliseconds << " Hz" << std::endl;
        }
    }
}

void ContactEstimation::publishOutput(int output_idx, std::shared_ptr<synced_proprioceptive_lcmt> synced_msgs)
{
    std::string binary = std::bitset<4>(output_idx).to_string(); // to binary
    synced_msgs.get()->num_legs = num_legs_;
    synced_msgs.get()->contact = {0, 0, 0, 0};
    for (int i = 0; i < synced_msgs.get()->num_legs; i++)
    {
        synced_msgs.get()->contact[i] = binary[i] == '1';
       	if (debug_flag_ == 1)
        {
            myfile_ << binary[i] << ',';
        }
    }
    
    
    if (debug_flag_ == 1)
    {
        myfile_ << '\n';
        myfile_ << std::flush;
    }
    
    lcm_->publish("synced_proprioceptive_data", synced_msgs.get());
}

#include "src/contact_estimation.hpp"
#include "utils/tensorrt_acc.hpp"
#include "src/lcm_cnn_interface.hpp"


ContactEstimation::ContactEstimation(const samplesCommon::Args &args, lcm::LCM* lcm_, std::mutex* mtx_, 
                                    int debug_flag_, std::ofstream& myfile_, std::ofstream& myfile_leg_p_, 
                                    LcmMsgQueues_t* lcm_msg_in)
    : input_h(75),
      input_w(54),
      sample(initializeSampleParams(args)),
      lcm(lcm_),
      mtx(mtx_),
      debug_flag(debug_flag_),
      myfile(myfile_),
      myfile_leg_p(myfile_leg_p_),
      lcm_msg_in_(lcm_msg_in)
{
    // cnn_input_matrix_normalized = new float[input_h * input_w];
    if (!sample.buildFromSerializedEngine())
    {
        std::cerr << "FAILED: Cannot build the engine" << std::endl;
        return;
    }
    if (!lcm->good())
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
            mtx->lock();

            float *cnn_input_matrix_normalized = cnn_input_queue.front();
            cnn_input_queue.pop();
            std::shared_ptr<synced_proprioceptive_lcmt> synced_msgs = lcm_msg_in_->synced_msgs_queue.front();
            lcm_msg_in_->synced_msgs_queue.pop();
            mtx->unlock();

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
    synced_msgs.get()->num_legs = 4;
    synced_msgs.get()->contact = {0, 0, 0, 0};
    for (int i = 0; i < synced_msgs.get()->num_legs; i++)
    {
        synced_msgs.get()->contact[i] = binary[i] == '1';
       	if (debug_flag == 1)
        {
            myfile << synced_msgs.get()->contact[i] << ',';
        }
    }
    
    
    if (debug_flag == 1)
    {
        myfile << '\n';
        myfile << std::flush;
    }
    
    lcm->publish("synced_proprioceptive_data", synced_msgs.get());
}

#include "src/contact_estimation.hpp"
#include "utils/tensorrt_acc.hpp"
#include "src/lcm_cnn_interface.hpp"


ContactEstimation::ContactEstimation(const samplesCommon::Args &args, lcm::LCM* lcm_, std::mutex* mtx_, 
                                    int debug_flag_, std::ofstream& myfile_, std::ofstream& myfile_leg_p_, 
                                    LcmMsgQueues_t* lcm_msg_in_)
    : input_h(600),
      input_w(46),
      sample(initializeSampleParams(args)),
      lcm(lcm_),
      mtx(mtx_),
      debug_flag(debug_flag_),
      myfile(myfile_),
      myfile_leg_p(myfile_leg_p_),
      lcm_msg_in(lcm_msg_in_)
{
    // cnn_input_matrix_normalized = new float[input_h * input_w];
    if (!sample.buildFromSerializedEngine())
    {
        std::cerr << "FAILED: Cannot build the engine" << std::endl;
        return;
    }
    if (!lcm->good())
        return;
    // cnn_output.num_legs = 4;
    // cnn_output.contact = {0, 0, 0, 0};
    cnn_output.num_legs = 2;
    cnn_output.contact = {0, 0};
}

ContactEstimation::~ContactEstimation() {}

void ContactEstimation::makeInference(std::queue<float *> &cnn_input_queue, std::queue<float *> &new_data_queue)
{
    while (true)
    {
        if (!cnn_input_queue.empty() && !new_data_queue.empty())
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
            float *new_data = new_data_queue.front();
            new_data_queue.pop();
            mtx->unlock();

            int idx = 0;
            const int legTypeDataNum = 14;
            const int IMUTypeDataNum = 3;
            for (int i = 0; i < legTypeDataNum; ++i)
            {
                cnn_output.q[i] = new_data[idx];
                ++idx;
            }
            // leg_control_data.qd:
            for (int i = 0; i < legTypeDataNum; ++i)
            {
                cnn_output.qd[i] = new_data[idx];
                ++idx;
            }
            // microstrain(IMU).acc:
            for (int i = 0; i < IMUTypeDataNum; ++i)
            {
                cnn_output.acc[i] = new_data[idx];
                ++idx;
            }
            // microstrain(IMU).omega:
            for (int i = 0; i < IMUTypeDataNum; ++i)
            {
                cnn_output.omega[i] = new_data[idx];
                ++idx;
            }
            // leg_control_data.p:

            for (int i = 0; i < 6; ++i)
            {
                cnn_output.p[i] = new_data[idx];
            
	       	if (debug_flag == 1)
                {
                    myfile_leg_p << new_data[idx] << ',';
                }
                ++idx;		
            }

    	    if (debug_flag == 1)
            {
                myfile_leg_p << '\n';
                myfile_leg_p.flush();
            }

            // leg_control_data.v:
            for (int i = 0; i < 6; ++i)
            {
                cnn_output.v[idx] = new_data[idx];
                ++idx;
            }

            int output_idx = sample.infer(cnn_input_matrix_normalized);
            if (output_idx == -1)
            {
                std::cerr << "FAILED: Cannot use the engine to infer a result" << std::endl;
                return;
            }

            mtx->lock();
            cnn_output.timestamp = lcm_msg_in->timestamp_queue.front();
            lcm_msg_in->timestamp_queue.pop();
            mtx->unlock();

            delete[] cnn_input_matrix_normalized;
            delete[] new_data;
            publishOutput(output_idx);

            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            float milliseconds = 0;
            // cudaEventElapsedTime(&milliseconds, start, stop);
            // std::cout << "It's frequency is " << 1000 / milliseconds << " Hz" << std::endl;
        }
    }
}

void ContactEstimation::publishOutput(int output_idx)
{
    std::string binary = std::bitset<2>(output_idx).to_string(); // to binary
    // std::cout << "Contact_state: " << output_idx << std::endl;
    for (int i = 0; i < cnn_output.num_legs; i++)
    {
        cnn_output.contact[i] = binary[i];
       	if (debug_flag == 1)
        {
            myfile << cnn_output.contact[i] << ',';
        }
    }
    
    
    if (debug_flag == 1)
    {
        myfile << '\n';
        myfile << std::flush;
    }
    

    lcm->publish("synced_proprioceptive_data", &cnn_output);
}

#include "lcm_cnn_interface.hpp" 



class Handler 
{
    public:
        ~Handler() {}
        
        void receive_leg_control_msg(const lcm::ReceiveBuffer* rbuf,
                                     const std::string& chan, 
                                     const leg_control_data_lcmt* msg)
        {
            // printf("Received message on channel \"%s\":\n", chan.c_str());
            std::vector<std::vector<float>> leg_control_data(4);
            // msg->q.begin() will show an error so I used std::begin() instead.
            leg_control_data[0] = std::vector<float> (std::begin(msg->q), std::end(msg->q));
            leg_control_data[1] = std::vector<float> (std::begin(msg->qd), std::end(msg->qd));
            leg_control_data[2] = std::vector<float> (std::begin(msg->p), std::end(msg->p));
            leg_control_data[3] = std::vector<float> (std::begin(msg->v), std::end(msg->v));

            cnnInputLegQueue.push(leg_control_data);
        }

        void receive_microstrain_msg(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const microstrain_lcmt* msg)
        {
            // printf("Received message on channel \"%s\":\n", chan.c_str());
            std::vector<std::vector<float>> microstrain_data(4);
            microstrain_data[0] = std::vector<float> (std::begin(msg->omega), std::end(msg->omega));
            microstrain_data[1] = std::vector<float> (std::begin(msg->acc), std::end(msg->acc));

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
    
};

class MatrixBuilder
{
public:
    static void BuildMatrix (){
        // Get leg input from queue
        while (true){
            if (!cnnInputLegQueue.empty() && !cnnInputIMUQueue.empty() && !cnnInputGtLabelQueue.empty()){
                // ++count;
                // std::cout << count << std::endl;
                mtx.lock();
                std::vector<float> legInput_q = cnnInputLegQueue.front()[0];
                std::vector<float> legInput_qd = cnnInputLegQueue.front()[1];
                std::vector<float> legInput_p = cnnInputLegQueue.front()[2];
                std::vector<float> legInput_v = cnnInputLegQueue.front()[3];
                cnnInputLegQueue.pop();

                // Get microstrain input from queue
                std::vector<float> IMUInput_omega = cnnInputIMUQueue.front()[0];
                std::vector<float> IMUInput_acc = cnnInputIMUQueue.front()[1];
                cnnInputIMUQueue.pop();

                // Get GTlabel from queue
                int gtLabel = cnnInputGtLabelQueue.front();
                cnnInputGtLabelQueue.pop();
                mtx.unlock();
                // Start to build a new line and generate a new input
                std::vector<float> newLine(54);
                int idx = 0; // keep track of the current newLine idx;
                int legTypeDataNum = 12;
                int IMUTypeDataNum = 3;
                
                for (int i = 0; i < legTypeDataNum; ++i)
                    newLine[idx++] = legInput_q[i];

                for (int i = 0; i < legTypeDataNum; ++i)
                    newLine[idx++] = legInput_qd[i];
                
                for (int i = 0; i < IMUTypeDataNum; ++i)
                    newLine[idx++] = IMUInput_acc[i];

                for (int i = 0; i < IMUTypeDataNum; ++i)
                    newLine[idx++] = IMUInput_omega[i];

                for (int i = 0; i < legTypeDataNum; ++i)
                    newLine[idx++] = legInput_p[i];

                for (int i = 0; i < legTypeDataNum; ++i)
                    newLine[idx++] = legInput_v[i];
                

                // Put the newLine to the InputMatrix and destroy the first line:
                cnnInputMatrix.erase(cnnInputMatrix.begin());
                cnnInputMatrix.push_back(newLine);
                dataRequire = std::max(dataRequire - 1, 0);

                
                // if (dataRequire == 0) {
                    /// TODO: send to CNN network in TRT
                // }
            } 
        }
    }
};


int main(int argc, char** argv)
{
    lcm::LCM lcm;
    if(!lcm.good())
        return 1;
    Handler handlerObject;
    lcm.subscribe("leg_control_data", &Handler::receive_leg_control_msg, &handlerObject);
    lcm.subscribe("microstrain", &Handler::receive_microstrain_msg, &handlerObject);
    lcm.subscribe("contact_ground_truth", &Handler::receive_contact_ground_truth_msg, &handlerObject);

    std::thread MatrixThread (MatrixBuilder::BuildMatrix);
    while(0 == lcm.handle());

    MatrixThread.join();

    return 0;
}

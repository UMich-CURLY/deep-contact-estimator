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
        float leg_control_data [48];
        int size = 12;
        arrayCopy(std::begin(leg_control_data), msg->q, size);
        arrayCopy(std::begin(leg_control_data) + size, msg->q, size);
        arrayCopy(std::begin(leg_control_data) + size + size, msg->q, size);
        arrayCopy(std::begin(leg_control_data) + size + size + size, msg->q, size);


        cnnInputLegQueue.push(leg_control_data);
    }

    void receive_microstrain_msg(const lcm::ReceiveBuffer* rbuf,
            const std::string& chan, 
            const microstrain_lcmt* msg)
    {
        // printf("Received message on channel \"%s\":\n", chan.c_str());
        float microstrain_data [6];
        int size = 3;
        arrayCopy(std::begin(microstrain_data), msg->acc, size);
        arrayCopy(std::begin(microstrain_data) + size, msg->omega, size);

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

    void arrayCopy(float array1 [], const float array2 [], int size) {
        for (int i = 0; i < size; ++i) {
            array1[i] = array2[i];
        }
    }

};

class MatrixBuilder
{
public:
    static void BuildMatrix (){
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
                for (int i = 0; i < legTypeDataNum; ++i)
                    newLine[idx++] = cnnInputLegQueue.front()[i];

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
                
                cnnInputLegQueue.pop();
                cnnInputIMUQueue.pop();
                mtx.unlock();

                // std::cout << "New Line: [";
                // for (int i = 0; i < 8; ++i) {
                //     std::cout << newLine[i * 5] << ", ";
                // }
                // std::cout << "]" << std::endl;
                
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

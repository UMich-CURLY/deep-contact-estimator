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

class MatrixBuilder
{
public:
    MatrixBuilder() {
        std::vector<float> mean_vector(input_w, 0);
        std::vector<float> std_vector(input_w, 0);
        int dataRequire = 150;
        std::vector<std::vector<float>> cnnInputMatrix(input_h, std::vector<float>(input_w));
        std::vector<std::vector<float>> cnnInputMatrix_normalized(input_h, std::vector<float>(input_w));
    }

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
                std::cout << dataRequire << std::endl;
                if (dataRequire == 0) {
                    /// TODO: send to CNN network in TRT
                    // We need to normalize the input matrix, to do so,
                    // we need to calculate the mean value and standard
                    // deviation.
                    
                    for (int j = 0; j < input_w; ++j) {
                        for (int i = 0; i < input_h; ++i) {
                            mean_vector[j] += cnnInputMatrix[i][j];
                        }
                        // std::cout << "Mean before divide: " << mean_vector[j] << std::endl;
                        mean_vector[j] = mean_vector[j] / input_h;
                        // std::cout << "Mean after divide: " << mean_vector[j] << std::endl;

                        for (int i = 0; i < input_h; ++i) {
                            std_vector[j] += std::pow((cnnInputMatrix[i][j] - mean_vector[j]), 2.0);
                        }
                        // std::cout << "STD before divide: " << std_vector[j] << std::endl;
                        std_vector[j] = std_vector[j] / (input_h - 1);
                        // std::cout << "STD after divide: " << std_vector[j] << std::endl;

                        // Normalize the matrix:
                        for (int i = 0; i < input_h; ++i) {
                            cnnInputMatrix_normalized[i][j] = (cnnInputMatrix[i][j] - mean_vector[j]) / std_vector[j];
                        }
                    }

                    std::ofstream output;
                    output.open("/home/tingjun/Desktop/TensorRT_PROJECT_USE/data/input_matrix.bin", std::ios::out | std::ios::binary);
                    size_t size_row = cnnInputMatrix_normalized.size();
                    std::cout << cnnInputMatrix_normalized[0][0] << std::endl;
                    std::cout << cnnInputMatrix_normalized[5][5] << std::endl;
                    std::cout << cnnInputMatrix_normalized[10][10] << std::endl;
                    std::cout << cnnInputMatrix_normalized[15][15] << std::endl;


                    for (size_t i = 0; i < size_row; ++i) {
                        output.write(reinterpret_cast<char*>(&cnnInputMatrix_normalized[i][0]), input_w * sizeof(float));
                    }

                    output.close();
                    std::cout << "The ground truth label is: " << gtLabel << std::endl;
                    break;
                }
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
    
    std::cout << "Start Running LCM-CNN Interface" << std::endl;
    std::thread MatrixThread (MatrixBuilder::BuildMatrix);
    while(0 == lcm.handle());

    MatrixThread.join();

    return 0;
}

#include "lcm_cnn_interface.hpp" 



class Handler 
{
    public:
        ~Handler() {}
        void receive_leg_control_msg(const lcm::ReceiveBuffer* rbuf,
                                     const std::string& chan, 
                                     const leg_control_data_lcmt* msg)
        {
            int i;
            printf("Received message on channel \"%s\":\n", chan.c_str());
            std::vector<std::vector<float>> leg_control_data(4);
            leg_control_data[0] = std::vector<float> (std::begin(msg->q), std::end(msg->q));
            leg_control_data[1] = std::vector<float> (std::begin(msg->qd), std::end(msg->qd));
            leg_control_data[2] = std::vector<float> (std::begin(msg->p), std::end(msg->p));
            leg_control_data[3] = std::vector<float> (std::begin(msg->v), std::end(msg->v));
        }

        void receive_microstrain_msg(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const microstrain_lcmt* msg)
        {
            printf("Received message on channel \"%s\":\n", chan.c_str());
            std::vector<std::vector<float>> microstrain_data(4);
            microstrain_data[0] = std::vector<float> (std::begin(msg->omega), std::end(msg->omega));
            microstrain_data[1] = std::vector<float> (std::begin(msg->acc), std::end(msg->acc));
        }

        void receive_contact_ground_truth_msg(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const contact_ground_truth_t* msg)
        {
            printf("Received message on channel \"%s\":\n", chan.c_str());
            std::vector<int8_t> contact_ground_truth_label = msg->contact;
            int gt_label = contact_ground_truth_label[0] * 2 * 2 * 2 
                        + contact_ground_truth_label[1] * 2 * 2
                        + contact_ground_truth_label[2] * 2
                        + contact_ground_truth_label[3];
            std::cout << gt_label << std::endl;
        }
    
};

class MatrixBuilder
{
public:
    /*
        Constructor for MatrixBuilder:
    */
    MatrixBuilder (std::vector<std::vector<float>> &leg_control_data,
                   std::vector<std::vector<float>> &microstrain_data,
                   int gt_label){

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

    while(0 == lcm.handle());
    return 0;
}

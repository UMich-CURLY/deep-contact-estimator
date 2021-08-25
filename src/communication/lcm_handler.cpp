#include "communication/lcm_handler.hpp"

LcmHandler::LcmHandler(lcm::LCM* lcm, LcmMsgQueues_t* lcm_msg_in, std::mutex* cdata_mtx):
lcm_(lcm), lcm_msg_in_(lcm_msg_in), cdata_mtx_(cdata_mtx)
{
    
    lcm->subscribe("leg_control_data", &LcmHandler::receiveLegControlMsg, this);
    lcm->subscribe("microstrain", &LcmHandler::receiveMicrostrainMsg, this);
    // lcm.subscribe("contact_ground_truth", &&LcmHandler::receiveContactGroundTruthMsg, this);
    start_time = 0;
    std::cout << "Subscribed to channels" << std::endl;
    
}

LcmHandler::~LcmHandler() {}

void LcmHandler::receiveLegControlMsg(const lcm::ReceiveBuffer *rbuf,
                                   const std::string &chan,
                                   const leg_control_data_lcmt *msg)
{
    if (start_time == 0) {
        start_time = rbuf->recv_utime;
    }

    int dim = 12;
    std::shared_ptr<LcmLegStruct> leg_control_data = std::make_shared<LcmLegStruct>();
    arrayCopy(leg_control_data.get()->q, msg->q, dim);
    arrayCopy(leg_control_data.get()->qd, msg->qd, dim);
    arrayCopy(leg_control_data.get()->p, msg->p, dim);
    arrayCopy(leg_control_data.get()->v, msg->v, dim);

    /// LOW: 500Hz version:
    lcm_msg_in_->cnn_input_leg_queue.push(leg_control_data);

    /// HIGH: 1000Hz version:
    // If the latest_idx reaches the limit of the buffer vector:
    // if (latest_idx == cnn_input_leg_vector.size() - 1) {
    //     latest_idx = -1;
    // }
    // if (cnn_input_leg_vector[latest_idx + 1] != nullptr) {
    //     delete[] cnn_input_leg_vector[latest_idx + 1];
    // }
    // lcm_msg_in_->cnn_input_leg_vector[++latest_idx] = leg_control_data;
}

void LcmHandler::receiveMicrostrainMsg(const lcm::ReceiveBuffer *rbuf,
                                    const std::string &chan,
                                    const microstrain_lcmt *msg)
{
    /// LOW: 500Hz version:
    if (start_time == 0) {
        start_time = rbuf->recv_utime;
    }
    if (lcm_msg_in_->cnn_input_leg_queue.size() >= lcm_msg_in_->cnn_input_imu_queue.size())
    {
        std::shared_ptr<LcmIMUStruct> microstrain_data = std::make_shared<LcmIMUStruct>();
        int dim = 3;
        arrayCopy(microstrain_data.get()->acc, msg->acc, dim);
        arrayCopy(microstrain_data.get()->omega, msg->omega, dim);
        double timestamp = (1.0 * (rbuf->recv_utime - start_time)) / pow(10, 6);
        lcm_msg_in_->timestamp_queue.push(timestamp);
        lcm_msg_in_->cnn_input_imu_queue.push(microstrain_data);
    }

    /// HIGH: 1000Hz version:
    // if (latest_idx != -1 && cnn_input_leg_vector[latest_idx] != nullptr) {
    //     std::shared_ptr<LcmIMUStruct> microstrain_data = std::make_shared<LcmIMUStruct>();
    //     arrayCopy(microstrain_data.get()->acc, msg->acc, size);
    //     arrayCopy(microstrain_data.get()->omega, msg->omega, size);

    //     cnn_input_imu_queue.push(microstrain_data);
    //     std::shared_ptr<LcmLegStruct> leg_control_data = std::make_shared<LcmLegStruct>();
    //     arrayCopy(leg_control_data.get()->q, msg->q);
    //     arrayCopy(leg_control_data.get()->qd, msg->qd);
    //     arrayCopy(leg_control_data.get()->p, msg->p);
    //     arrayCopy(leg_control_data.get()->v, msg->v);
    //     lcm_msg_in_->cnn_input_leg_queue.push(leg_control_data);
    // }
}

void LcmHandler::receiveContactGroundTruthMsg(const lcm::ReceiveBuffer *rbuf,
                                           const std::string &chan,
                                           const contact_ground_truth_t *msg)
{
    std::vector<int8_t> contact_ground_truth_label = msg->contact;

    int gt_label = contact_ground_truth_label[0] * 2 * 2 * 2 + contact_ground_truth_label[1] * 2 * 2 + contact_ground_truth_label[2] * 2 + contact_ground_truth_label[3];

    lcm_msg_in_->cnn_input_gtlabel_queue.push(gt_label);
}

void LcmHandler::arrayCopy(float array1[], const float array2[], const int dim)
{
    for (int i = 0; i < dim; ++i)
    {
        array1[i] = array2[i];
    }
}

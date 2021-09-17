#include "communication/lcm_handler.hpp"
#include <stdlib.h>

LcmHandler::LcmHandler(lcm::LCM* lcm, LcmMsgQueues_t* lcm_msg_in, std::mutex* cdata_mtx, const std::string &mode):
lcm_(lcm), lcm_msg_in_(lcm_msg_in), mtx_(cdata_mtx)
{
    char resolved_path[PATH_MAX];
    realpath("../", resolved_path);
    config_ = YAML::LoadFile(std::string(resolved_path) + "/config/interface.yaml");
    
    if (mode == "normal") {
        lcm->subscribe(config_["lcm_leg_channel"].as<std::string>(), &LcmHandler::receiveLegControlMsg, this);
        lcm->subscribe(config_["lcm_imu_channel"].as<std::string>(), &LcmHandler::receiveMicrostrainMsg, this); 
    }

    else if (mode == "fast") {
        lcm->subscribe(config_["lcm_leg_channel"].as<std::string>(), &LcmHandler::receiveLegControlMsg_Fast, this);
        lcm->subscribe(config_["lcm_imu_channel"].as<std::string>(), &LcmHandler::receiveMicrostrainMsg_Fast, this);
    }

    // lcm.subscribe(config_["contact_ground_truth"].as<std::string>(), &&LcmHandler::receiveContactGroundTruthMsg, this);
    start_time_ = 0;
    q_dim = config_["leg_q_dimension"].as<int>();
    qd_dim = config_["leg_qd_dimension"].as<int>();
    p_dim = config_["leg_p_dimension"].as<int>();
    v_dim = config_["leg_v_dimension"].as<int>();
    tau_est_dim = config_["leg_tau_dimension"].as<int>();

    acc_dim = config_["imu_acc_dimension"].as<int>();
    omega_dim = config_["imu_omega_dimension"].as<int>();
    quat_dim = config_["imu_quat_dimension"].as<int>();
    rpy_dim = config_["imu_rpy_dimension"].as<int>();

    std::cout << "Subscribed to channels" << std::endl;
    
}

LcmHandler::~LcmHandler() {}

void LcmHandler::receiveLegControlMsg(const lcm::ReceiveBuffer *rbuf,
                                   const std::string &chan,
                                   const leg_control_data_lcmt *msg)
{
    if (start_time_ == 0) {
        start_time_ = rbuf->recv_utime;
    }

    std::shared_ptr<LcmLegStruct> leg_control_data = std::make_shared<LcmLegStruct>();
    arrayCopy(leg_control_data.get()->q, msg->q, q_dim);
    arrayCopy(leg_control_data.get()->qd, msg->qd, qd_dim);
    arrayCopy(leg_control_data.get()->p, msg->p, p_dim);
    arrayCopy(leg_control_data.get()->v, msg->v, v_dim);
    arrayCopy(leg_control_data.get()->tau_est, msg->tau_est, tau_est_dim);

    /// LOW: 500Hz version:
    mtx_->lock();
    lcm_msg_in_->cnn_input_leg_queue.push(leg_control_data);
    mtx_->unlock();
}

void LcmHandler::receiveLegControlMsg_Fast(const lcm::ReceiveBuffer *rbuf,
                                   const std::string &chan,
                                   const leg_control_data_lcmt *msg)
{
    if (start_time_ == 0) {
        start_time_ = rbuf->recv_utime;
    }

    std::shared_ptr<LcmLegStruct> leg_control_data = std::make_shared<LcmLegStruct>();
    arrayCopy(leg_control_data.get()->q, msg->q, q_dim);
    arrayCopy(leg_control_data.get()->qd, msg->qd, qd_dim);
    arrayCopy(leg_control_data.get()->p, msg->p, p_dim);
    arrayCopy(leg_control_data.get()->v, msg->v, v_dim);
    arrayCopy(leg_control_data.get()->tau_est, msg->tau_est, tau_est_dim);

    /// HIGH: 1000Hz version:
    mtx_->lock();
    if (!lcm_msg_in_->cnn_input_leg_stack.empty()) 
        lcm_msg_in_->cnn_input_leg_stack.pop();
    
    lcm_msg_in_->cnn_input_leg_stack.push(leg_control_data);
    mtx_->unlock();
}


void LcmHandler::receiveMicrostrainMsg(const lcm::ReceiveBuffer *rbuf,
                                    const std::string &chan,
                                    const microstrain_lcmt *msg)
{
    /// LOW: 500Hz version:
    if (start_time_ == 0) {
        start_time_ = rbuf->recv_utime;
    }
    if (lcm_msg_in_->cnn_input_leg_queue.size() > lcm_msg_in_->cnn_input_imu_queue.size())
    {
        std::shared_ptr<LcmIMUStruct> microstrain_data = std::make_shared<LcmIMUStruct>();
        arrayCopy(microstrain_data.get()->acc, msg->acc, acc_dim);
        arrayCopy(microstrain_data.get()->omega, msg->omega, omega_dim);
        arrayCopy(microstrain_data.get()->quat, msg->quat, quat_dim);
        arrayCopy(microstrain_data.get()->rpy, msg->rpy, rpy_dim);
        microstrain_data.get()->good_packets = msg->good_packets;
        microstrain_data.get()->bad_packets = msg->bad_packets;

        double timestamp = (1.0 * (rbuf->recv_utime - start_time_)) / pow(10, 6);
        lcm_msg_in_->timestamp_queue.push(timestamp);
        lcm_msg_in_->cnn_input_imu_queue.push(microstrain_data);
    }
}


void LcmHandler::receiveMicrostrainMsg_Fast(const lcm::ReceiveBuffer *rbuf,
                                    const std::string &chan,
                                    const microstrain_lcmt *msg)
{
    if (start_time_ == 0) {
        start_time_ = rbuf->recv_utime;
    }

    /// HIGH: 1000Hz version:
    if (!lcm_msg_in_->cnn_input_leg_stack.empty()) {
        std::shared_ptr<LcmIMUStruct> microstrain_data = std::make_shared<LcmIMUStruct>();
        arrayCopy(microstrain_data.get()->acc, msg->acc, acc_dim);
        arrayCopy(microstrain_data.get()->omega, msg->omega, omega_dim);
        arrayCopy(microstrain_data.get()->quat, msg->quat, quat_dim);
        arrayCopy(microstrain_data.get()->rpy, msg->rpy, rpy_dim);
        microstrain_data.get()->good_packets = msg->good_packets;
        microstrain_data.get()->bad_packets = msg->bad_packets;

        double timestamp = (1.0 * (rbuf->recv_utime - start_time_)) / pow(10, 6);

        mtx_->lock();
        lcm_msg_in_->cnn_input_leg_queue.push(lcm_msg_in_->cnn_input_leg_stack.top());
        lcm_msg_in_->timestamp_queue.push(timestamp);
        lcm_msg_in_->cnn_input_imu_queue.push(microstrain_data);
        mtx_->unlock();
    }
}


void LcmHandler::receiveContactGroundTruthMsg(const lcm::ReceiveBuffer *rbuf,
                                           const std::string &chan,
                                           const contact_ground_truth_t *msg)
{
    std::vector<int8_t> contact_ground_truth_label = msg->contact;

    int gt_label = contact_ground_truth_label[0] * 2 * 2 * 2 + contact_ground_truth_label[1] * 2 * 2 + contact_ground_truth_label[2] * 2 + contact_ground_truth_label[3];
    for (int i = 0; i < contact_ground_truth_label.size(); i++) {
        int j = 0;
        int contact_tmp = contact_ground_truth_label[i];
        while (j < 4 - i) {
            contact_tmp *= 2;
            ++j;
        }
        gt_label += contact_tmp;
    }
    lcm_msg_in_->cnn_input_gtlabel_queue.push(gt_label);
}

void LcmHandler::arrayCopy(float array1[], const float array2[], const int dim)
{
    for (int i = 0; i < dim; ++i)
    {
        array1[i] = array2[i];
    }
}

#ifndef lcm_handler_h
#define lcm_handler_h

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h> 

#include <lcm/lcm-cpp.hpp>
#include "../lcm_types/cpp/leg_control_data_lcmt.hpp"
#include "../lcm_types/cpp/microstrain_lcmt.hpp"
#include "../lcm_types/cpp/contact_t.hpp"
#include "../lcm_types/cpp/contact_ground_truth_t.hpp"
#include "../lcm_types/cpp/synced_proprioceptive_lcmt.hpp"

#include "lcm_msg_queue.hpp"
#include "yaml-cpp/yaml.h"

//!
//! \brief The Handler class takes in LCM messages from subscribed channels and process them
//! 
class LcmHandler
{
public:
    ~LcmHandler();

    LcmHandler(lcm::LCM* lcm, LcmMsgQueues_t* lcm_msg_in, std::mutex* cdata_mtx);

    //!
    //! \brief Receives messages from the "leg_control_data" channel and  
    //! stores them in an array, then put the array pointer into a queue
    //!
    void receiveLegControlMsg(const lcm::ReceiveBuffer* rbuf,
                                 const std::string& chan, 
                                 const leg_control_data_lcmt* msg);

    //!
    //! \brief Receives messages from the "microstrain" channel and  
    //! stores them in an array, then put the array pointer into a queue
    //!
    void receiveMicrostrainMsg(const lcm::ReceiveBuffer* rbuf,
                                 const std::string& chan, 
                                 const microstrain_lcmt* msg);

    //!
    //! \brief Receives messages from the "contact_ground_truth" channel and  
    //! stores it in an array, then put the array pointer into a queue
    //!
    void receiveContactGroundTruthMsg(const lcm::ReceiveBuffer* rbuf,
                                          const std::string& chan, 
                                          const contact_ground_truth_t* msg);

private:
    //! \brief Copy values from array2 to array1
    //! 
    //! \details Pass the begin pointer of array1 and array2 to copy values in array2 
    //! to array1 starting from the pointer
    //! 
    void arrayCopy(float array1 [], const float array2 [], const int dim);
    
    LcmMsgQueues_t* lcm_msg_in_; //<! Stores necessary lcm messages
    std::mutex* cdata_mtx_; //<! use in mutex lock
    lcm::LCM* lcm_; //<! lcm object
    int64_t start_time_; //<! the starting time of the interface
    YAML::Node config_; //<! load interface config file
    int q_dim;
    int qd_dim;
    int p_dim;
    int v_dim;
    int tau_est_dim;

    int acc_dim;
    int omega_dim;
    int quat_dim;
    int rpy_dim;
};

#endif

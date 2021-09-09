#pragma once

#include <stdio.h>
#include <iostream>

struct LcmLegStruct {
    float q[12];
    float qd[12];
    float p[12];
    float v[12];
    float tau_est[12];
};

struct LcmIMUStruct {
    float quat[4];
    float rpy[3];
    float omega[3];
    float acc[3];
    int64_t good_packets;
    int64_t bad_packets;
};

struct LcmContactStruct {
    int8_t  num_legs;
    double  timestamp;
    int8_t  contact[4];
};

struct LcmSyncedOutputStruct {
    int8_t  num_legs;
    double  timestamp;
    bool  contact[4];
    
    float q[12];
    float qd[12];
    float p[12];
    float v[12];
    float tau_est[12];

    float quat[4];
    float rpy[3];
    float omega[3];
    float acc[3];
    int64_t good_packets;
    int64_t bad_packets;
};

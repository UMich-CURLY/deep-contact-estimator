# This is a send message demo and combines messages from
# leg_control_data_lcmt and microstrain_lcmt into
# cnn_input_lcmt
import sys
sys.path.append("../lcm_types")
# from python import contact_t
from python import leg_control_data_lcmt
from python import microstrain_lcmt
from python import cnn_input_lcmt
import lcm
import time
import numpy as np

lc = lcm.LCM()

leg_control_data_msg = leg_control_data_lcmt()
leg_control_data_msg.q = [k for k in range(12)]
leg_control_data_msg.qd = [0.0] * 12
leg_control_data_msg.p = [1.0] * 12
leg_control_data_msg.v = [2.0] * 12
leg_control_data_msg.tau_est = [3.0] * 12

microstrain_msg = microstrain_lcmt()
microstrain_msg.omega = [3.0] * 3
microstrain_msg.acc = [4.0] * 3

cnn_input_msg = cnn_input_lcmt()
cnn_input_msg.q = leg_control_data_msg.q
cnn_input_msg.qd = leg_control_data_msg.qd
cnn_input_msg.p = leg_control_data_msg.p
cnn_input_msg.v = leg_control_data_msg.v
cnn_input_msg.tau_est = leg_control_data_msg.tau_est
cnn_input_msg.omega = microstrain_msg.omega
cnn_input_msg.acc = microstrain_msg.acc

lc.publish("leg_control_data", leg_control_data_msg.encode())
lc.publish("microstrain", microstrain_msg.encode())

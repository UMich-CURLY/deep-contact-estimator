# This file takes in and decode lcm messages, then reconstruct the data to a CNN input matrix
import sys

sys.path.append("../lcm_types")
from python import contact_t
from python import leg_control_data_lcmt
from python import microstrain_lcmt
from CNN_INPUT import CNNInput
import lcm
import numpy as np


def my_handler(channel, data):
    if channel == "leg_control_data":
        leg_control_data_msg = leg_control_data_lcmt.decode(data)
        # print("q is: %s" % str(leg_control_data_msg.q))
        # print("qd is: %s" % str(leg_control_data_msg.qd))
        cnn_input.q = np.array(leg_control_data_msg.q)
        cnn_input.qd = np.array(leg_control_data_msg.qd)
        cnn_input.p = np.array(leg_control_data_msg.p)
        cnn_input.v = np.array(leg_control_data_msg.v)
        cnn_input.leg_control_data_ready = True

    if channel == "microstrain":
        microstrain_msg = microstrain_lcmt.decode(data)
        # print("omega is: %s" % str(microstrain_msg.omega))
        # print("acc is: %s" % str(microstrain_msg.acc))
        cnn_input.omega = np.array(microstrain_msg.omega)
        cnn_input.acc = np.array(microstrain_msg.acc)
        cnn_input.microstrain_ready = True

    if channel == "contact_data":
        contact_msg = contact_t.decode(data)
        print("omega is: %s" % str(contact_msg.num_legs))
        print("acc is: %s" % str(contact_msg.contac))


# Define LCM subscription channels:
lc = lcm.LCM()
subscription1 = lc.subscribe("microstrain", my_handler)
subscription2 = lc.subscribe("leg_control_data", my_handler)
# subscription3 = lc.subscribe("contact_data", my_handler)
# Define cnn input class:
input_rows = 150
input_cols = 54
cnn_input = CNNInput(input_rows, input_cols)

try:
    while True:
        if cnn_input.leg_control_data_ready and cnn_input.microstrain_ready:
            cnn_input.build_input_matrix()
            cnn_input.leg_control_data_ready = False
            cnn_input.microstrain_ready = False
            if cnn_input.data_require == 0:
                print("CNN_INPUT Matrix is: ", cnn_input.cnn_input_matrix)

        lc.handle()


except KeyboardInterrupt:
    pass

lc.unsubscribe(subscription1)
lc.unsubscribe(subscription2)
# lc.unsubscribe(subscription3)

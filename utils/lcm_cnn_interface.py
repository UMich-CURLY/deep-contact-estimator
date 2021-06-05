# This file takes in and decode lcm messages, then reconstruct the data to a CNN input matrix
import sys
sys.path.append("..")
from lcm_types.python import contact_ground_truth_t
from lcm_types.python import contact_t
from lcm_types.python import leg_control_data_lcmt
from lcm_types.python import microstrain_lcmt
from src import pass_to_cnn
from CNN_INPUT import CNNInput
import lcm
import numpy as np
# Testing section:
import yaml
import argparse
import os


def binary2decimal(a, axis=-1):
    return np.right_shift(np.packbits(a, axis=axis), 8 - a.shape[axis]).squeeze()


def decimal2binary(num, length):
    num = int(num)
    res = []
    while num > 0:
        res = [(num % 2)] + res
        num = num // 2

    while len(res) < length:
        res = [0] + res

    return res


def my_handler(channel, data):
    if channel == "leg_control_data":
        leg_control_data_msg = leg_control_data_lcmt.decode(data)
        cnn_input.q = np.array(leg_control_data_msg.q)
        cnn_input.qd = np.array(leg_control_data_msg.qd)
        cnn_input.p = np.array(leg_control_data_msg.p)
        cnn_input.v = np.array(leg_control_data_msg.v)
        cnn_input.leg_control_data_ready = True

    # If the frequency of microstrain is doubled, we also need to double the frequency of leg_control_data
    # Or slow down the frequency of microstrain
    if channel == "microstrain" and cnn_input.leg_control_data_ready:
        microstrain_msg = microstrain_lcmt.decode(data)
        cnn_input.omega = np.array(microstrain_msg.omega)
        cnn_input.acc = np.array(microstrain_msg.acc)
        cnn_input.microstrain_ready = True

    if channel == "contact":
        contact_msg = contact_t.decode(data)
        print("num_leg is: %s" % str(contact_msg.num_legs))
        print("contact state is: %s" % str(contact_msg.contact))

    if channel == "contact_ground_truth":
        contact_ground_truth_msg = contact_ground_truth_t.decode(data)
        cnn_input.new_label = binary2decimal(np.array(contact_ground_truth_msg.contact))
        cnn_input.label_ready = True


# Define LCM subscription channels:
lc = lcm.LCM()
subscription1 = lc.subscribe("microstrain", my_handler)
subscription2 = lc.subscribe("leg_control_data", my_handler)
subscription3 = lc.subscribe("contact_ground_truth", my_handler)
# subscription4 = lc.subscribe("contact", my_handler)

# Define cnn input class:
input_rows = 150
input_cols = 54
cnn_input = CNNInput(input_rows, input_cols)

# For testing:
parser = argparse.ArgumentParser(description='Convert mat to numpy.')
parser.add_argument('--config_name', type=str,
                    default=os.path.dirname(os.path.abspath(__file__)) + '/../config/mat2numpy_config.yaml')
args = parser.parse_args()
config = yaml.load(open(args.config_name))

try:
    while True:
        # handle the next message
        lc.handle()

        # Check whether the input is enough:
        input_ready = cnn_input.leg_control_data_ready and cnn_input.microstrain_ready and cnn_input.label_ready
        if input_ready:
            # Put the current input to the input matrix
            cnn_input.build_input_matrix()
            cnn_input.leg_control_data_ready = False
            cnn_input.microstrain_ready = False
            cnn_input.label_ready = False
            if cnn_input.data_require == 0:
                # Save to path:
                # np.save(config['save_path_test_lcm'] + "test_lcm.npy", cnn_input.cnn_input_matrix)
                # np.save(config['save_path_test_lcm'] + "test_label_lcm.npy", cnn_input.label)

                # Publish channel:
                contact_msg = contact_t()
                # This is just for presenting the result. In actual deployment, we don't have the gt_label
                prediction = pass_to_cnn.receive_msg(cnn_input.cnn_input_matrix, cnn_input.label)
                contact_msg.num_legs = 4
                contact_msg.contact = decimal2binary(prediction, contact_msg.num_legs)
                print(contact_msg.contact)
                lc.publish("contact", contact_msg.encode())


except KeyboardInterrupt:
    pass

lc.unsubscribe(subscription1)
lc.unsubscribe(subscription2)
lc.unsubscribe(subscription3)
# lc.unsubscribe(subscription4)

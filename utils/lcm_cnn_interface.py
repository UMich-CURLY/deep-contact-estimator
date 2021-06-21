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
import threading
import queue
import time


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
        cnn_input_leg_list = [leg_control_data_msg.q, leg_control_data_msg.qd,
                              leg_control_data_msg.p, leg_control_data_msg.v]
        cnn_input_leg_q.put(cnn_input_leg_list)
        cnn_input.leg_control_data_ready = True

    # If the frequency of microstrain is doubled, we also need to double the frequency of leg_control_data
    # Or slow down the frequency of microstrain
    if channel == "microstrain" and cnn_input.leg_control_data_ready:
        microstrain_msg = microstrain_lcmt.decode(data)
        cnn_input_mcst_list = [microstrain_msg.omega, microstrain_msg.acc]
        cnn_input_mcst_q.put(cnn_input_mcst_list)
        cnn_input.microstrain_ready = True

    if channel == "contact_ground_truth":
        contact_ground_truth_msg = contact_ground_truth_t.decode(data)
        gt_label_q.put(binary2decimal(np.array(contact_ground_truth_msg.contact)))


def listen_and_publish():
    while True:
        lc.handle()
        input_ready = cnn_input.leg_control_data_ready and cnn_input.microstrain_ready
        if input_ready:
            cnn_input_q.put(cnn_input)
            cnn_input.leg_control_data_ready = False
            cnn_input.microstrain_ready = False


def get_cnn_output():
    while True:
        # Put the current input to the input matrix
        if not cnn_input_leg_q.empty() and not cnn_input_mcst_q.empty():
            # Assign values:
            leg_input = cnn_input_leg_q.get()
            mcst_input = cnn_input_mcst_q.get()
            # Delete the following line in actual deployment:
            gt_label = gt_label_q.get()
            cnn_input.q = np.array(leg_input[0])
            cnn_input.qd = np.array(leg_input[1])
            cnn_input.p = np.array(leg_input[2])
            cnn_input.v = np.array(leg_input[3])
            cnn_input.omega = np.array(mcst_input[0])
            cnn_input.acc = np.array(mcst_input[1])

            # Delete the following line in actual deployment:
            cnn_input.new_label = np.array(gt_label)

            # Build the input matrix and identify the if there are enough valid rows in the matrix:
            cnn_input.build_input_matrix()
            if cnn_input.data_require > 0:
                continue
            # Calculate and publish messages:`
            contact_msg = contact_t()
            prediction = pass_to_cnn.receive_input(cnn_input.cnn_input_matrix, input_rows, input_cols, cnn_input.label)
            cnn_input.count += 1
            contact_msg.timestamp = cnn_input.count
            contact_msg.num_legs = 4
            contact_msg.contact = decimal2binary(prediction, contact_msg.num_legs)
            print(contact_msg.contact)
            lc.publish("contact", contact_msg.encode())

# Define LCM subscription channels:
lc = lcm.LCM()
subscription1 = lc.subscribe("microstrain", my_handler)
subscription2 = lc.subscribe("leg_control_data", my_handler)
subscription3 = lc.subscribe("contact_ground_truth", my_handler)

# Define cnn input class:
input_rows = 150
input_cols = 54
cnn_input = CNNInput(input_rows, input_cols)
count = 0
cnn_input_q = queue.Queue()
cnn_input_leg_q = queue.Queue()
cnn_input_mcst_q = queue.Queue()
cnn_output_q = queue.Queue()
gt_label_q = queue.Queue()


# Multithreading:
cnn_output_thread = threading.Thread(target=get_cnn_output)

try:
    cnn_output_thread.start()
    listen_and_publish()

except KeyboardInterrupt:
    cnn_output_thread.join()
    pass

lc.unsubscribe(subscription1)
lc.unsubscribe(subscription2)

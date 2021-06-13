import numpy as np


class CNNInput:
    def __init__(self, num_rows, num_features):
        self.q = np.zeros(12)
        self.qd = np.zeros(12)
        self.p = np.zeros(12)
        self.v = np.zeros(12)
        self.omega = np.zeros(3)
        self.acc = np.zeros(3)
        self.cnn_input_matrix = np.zeros((num_rows, num_features))
        self.label = np.zeros((num_rows))
        self.new_label = np.zeros(1)
        print("shape of the CNN input matrix is: ", np.shape(self.cnn_input_matrix))
        self.leg_control_data_ready = False
        self.microstrain_ready = False
        self.data_require = num_rows

    def build_input_matrix(self):
        # Concatenate the new data into a new row:
        new_row = np.hstack((self.q, self.qd, self.acc, self.omega, self.p, self.v)).reshape(1, -1)
        # Discard the first row in cnn_input:
        self.cnn_input_matrix = np.vstack((self.cnn_input_matrix[1:, :], new_row))
        self.label = np.hstack((self.label[1:], self.new_label))
        self.data_require = max(self.data_require - 1, 0)

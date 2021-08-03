import numpy as np
import scipy.io

label_mat = scipy.io.loadmat('label_gt.mat')
input_matrix_mat = scipy.io.loadmat('input_matrix_500Hz.mat')

label_gt = np.array(label_mat['label_gt'])
input_matrix = np.array(input_matrix_mat['input_matrix'])


np.save("./" + "test_lcm.npy", input_matrix)
np.save("./" + "test_label_lcm.npy", label_gt)



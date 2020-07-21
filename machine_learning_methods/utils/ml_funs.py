import numpy as np
from machine_learning_methods.utils.estimate_contact import estimateContacts
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

def get_ground_truth(data):
    N = data.shape[0]
    ground_truth = np.zeros((N, 2))
    for i in tqdm(range(N)):
        encoder = data[i, 0:14]
        cl, cr = estimateContacts(encoder, 100)
        ground_truth[i, 0] = cl
        ground_truth[i, 1] = cr

    return ground_truth

def get_acc(ground_truth, rst):
    ground_truth = ground_truth.reshape(-1, )
    rst = rst.reshape(-1, )
    N = ground_truth.shape[0]
    ct_correct = np.sum(ground_truth == rst)
    print(ct_correct, N)

    return ct_correct/N

def combine_Y(labels):
    N = labels.shape[0]
    Y = np.zeros((N, ))

    for i in tqdm(range(N)):
        if labels[i, 0] == 0 and labels[i, 1] == 1:
            Y[i] = 0

        if labels[i, 0] == 1 and labels[i, 1] == 0:
            Y[i] = 1

        if labels[i, 0] == 1 and labels[i, 1] == 1:
            Y[i] = 2

    return Y.astype(int)


def km_get_acc(y_true, y_predicted, cluster_number):

    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[int(y_predicted[i]), int(y_true[i])] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return accuracy
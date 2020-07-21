import numpy as np
from tqdm import tqdm
import torch



def combine_Y(labels):
    N = labels.shape[0]
    Y = np.zeros((N, ))
    for i in tqdm(range(N)):
        # if labels[i, 0] == 0 and labels[i, 1] == 0:
        #     Y[i] = 0
        #     c0 = c0 + 1
        if labels[i, 0] == 0 and labels[i, 1] == 1:
            Y[i] = 0

        if labels[i, 0] == 1 and labels[i, 1] == 0:
            Y[i] = 1

        if labels[i, 0] == 1 and labels[i, 1] == 1:
            Y[i] = 2

    return Y.astype(int)

def get_acc(dataset, nt):
    crt = 0
    total = 0
    for data in tqdm(dataset):
        x, y = data['data'], data['label']
        x = x.cuda()
        y = y.cuda()
        y_h = nt(x)
        _, y_s = torch.max(y_h, 1)
        total += y.size(0)
        crt += (y_s == y).sum().item()
    print(crt, total)

    return crt/total
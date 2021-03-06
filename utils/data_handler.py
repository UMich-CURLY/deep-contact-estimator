import os
import argparse
import glob

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


def load_data_from_mat(data_pth, train_ratio=0.7, val_ratio=0.15):
    """
    Load data from .mat file.

    Inputs:
    - data_pth: path to the data folder
    - train_ratio: ratio of training data
    - val_ratio: ratio of validation data

    Data should be stored in .mat file, and contain:
    - q: joint encoder value (num_data,12)
    - qd: joint angular velocity (num_data,12)
    - p: foot position from FK (num_data,12)
    - v: foot velocity from FK (num_data,12)
    - tau_est: estimated control torque (num_data,12)
    - imu_acc: linear acceleration from imu (num_data,3)
    - imu_omega: angular velocity from imu (num_data,3)
    - contacts: contact data (num_data,4)
                contacts are stored as binary values, in the order
                of right_front, left_front, right_hind, left_hind.

                FRONT
                1 0  RIGHT
                3 2
                BACK

                Contact value will be treated as binary values in the
                order of contacts[0], contacts[1], contacts[2], contacts[3]
                and be converted to decimal value in this function.

                Ex. [1,0,0,1] -> 9
                    [0,1,1,0] -> 6
                     
    - F (optional): ground reaction force

    Output:
    - data['train']: train_data
    - data['val']: val_data
    - data['test']: test_data
    - data['train_label']: train_label
    - data['val_label']: val_label
    - data['test_label']: test_label
    """

    num_features = 66    
    train_data = np.zeros((0,num_features))
    val_data = np.zeros((0,num_features))
    test_data = np.zeros((0,num_features))
    train_label = np.zeros((0,1))
    val_label = np.zeros((0,1))
    test_label = np.zeros((0,1))
    data = {}

    # for all dataset in the folder
    for data_name in glob.glob(data_pth+'*'): 
        
        print("loading... ", data_name)

        # load data
        raw_data = sio.loadmat(data_name)

        contacts = raw_data['contacts']
        q = raw_data['q']
        p = raw_data['p']
        qd = raw_data['qd']
        v = raw_data['v']
        tau_est = raw_data['tau_est']
        acc = raw_data['imu_acc']
        omega = raw_data['imu_omega']
        
        # concatenate current data. First we try without GRF
        cur_data = np.concatenate((q,qd,acc,omega,p,v,tau_est),axis=1)
        
        # separate data into train/val/test
        num_data = np.shape(q)[0]
        num_train = int(train_ratio*num_data)
        num_val = int(val_ratio*num_data)
        num_test = num_data-num_train-num_val

        # normalize data
        cur_val = cur_data[:num_val,:]
        cur_val = (cur_val-np.mean(cur_val,axis=0))/np.std(cur_val,axis=0)        
        cur_test = cur_data[num_val:num_val+num_test,:]
        cur_test = (cur_test-np.mean(cur_test,axis=0))/np.std(cur_test,axis=0)
        cur_train = cur_data[num_val+num_test:,:]
        cur_train = (cur_train-np.mean(cur_train,axis=0))/np.std(cur_train,axis=0)
        
        

        # stack with all other sequences
        train_data = np.vstack((train_data,cur_train))
        val_data = np.vstack((val_data,cur_val))
        test_data = np.vstack((test_data,cur_test))

        
        # convert labels from binary to decimal
        cur_label = binary_to_decimal(contacts).reshape((-1,1))   

        # stack labels 
        
        val_label = np.vstack((val_label,cur_label[:num_val,:]))
        test_label = np.vstack((test_label,cur_label[num_val:num_val+num_test,:]))
        train_label = np.vstack((train_label,cur_label[num_val+num_test:,:]))

        # break

    data['train'] = train_data
    data['val'] = val_data
    data['test'] = test_data
    data['train_label'] = train_label.reshape(-1,)
    data['val_label'] = val_label.reshape(-1,)
    data['test_label'] = test_label.reshape(-1,)

    return data

def binary_to_decimal(a, axis=-1):
    return np.right_shift(np.packbits(a, axis=axis), 8 - a.shape[axis]).squeeze()


class contact_dataset(Dataset):

    def __init__(self, data, label, device='cuda'):
        self.data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
        self.label = torch.from_numpy(label).type('torch.LongTensor').to(device)

    def __len__(self):
        return list(self.data.size())[0]

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        this_data = self.data[idx]
        this_label = self.label[idx]
        sample = {'data': this_data, 'label': this_label}

        return sample


# def main():
#     parser = argparse.ArgumentParser(description='Train network')
#     parser.add_argument('--data_folder', type=str, help='path to contact dataset', default="/home/justin/data/2021-02-21_contact_data_in_lab/cnn_data/")
#     args = parser.parse_args()

#     data = load_data_from_mat(args.data_folder,0.7,0.15)


# if __name__ == '__main__':
#     main()
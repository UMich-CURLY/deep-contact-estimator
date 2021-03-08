import os
import argparse
import glob

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math

import torch
from torch.utils.data import Dataset, DataLoader


def load_data_from_mat(data_pth, window_size, window_size_ratio, train_ratio=0.7, val_ratio=0.15):
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
    train_data = np.zeros((0,window_size,num_features))
    val_data = np.zeros((0,window_size,num_features))
    test_data = np.zeros((0,window_size,num_features))
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
        
        # convert labels from binary to decimal
        cur_label = binary_to_decimal(contacts).reshape((-1,1)) 

        # stack data with window size
        stacked_cur_data, stacked_cur_label = slice_time_window_and_stack(cur_data,cur_label,window_size,window_size_ratio)

        # separate data into train/val/test
        num_data = stacked_cur_data.shape[0]
        num_train = int(train_ratio*num_data)
        num_val = int(val_ratio*num_data)
        num_test = num_data-num_train-num_val

        # normalize data
        cur_val = stacked_cur_data[:num_val,:,:]
        cur_test = stacked_cur_data[num_val:num_val+num_test,:,:]
        cur_train = stacked_cur_data[num_val+num_test:,:,:]
        
        # stack with all other sequences
        train_data = np.vstack((train_data,cur_train))
        val_data = np.vstack((val_data,cur_val))
        test_data = np.vstack((test_data,cur_test))

        # stack labels with all other sequences
        val_label = np.vstack((val_label,stacked_cur_label[:num_val,:]))
        test_label = np.vstack((test_label,stacked_cur_label[num_val:num_val+num_test,:]))
        train_label = np.vstack((train_label,stacked_cur_label[num_val+num_test:,:]))

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


def slice_time_window_and_stack(data, label, window_size, window_size_ratio):
    """
    Input:
    - data: Time series data we generated. Rows should be time. Cols are features at each time step. (num_data x num_feature)
    - label: Label that aligns with data.
    - window_size: Desired window size along time (row) for network to look at. 

    Output:
    - data_out: Stacked normalized data. (num_data-window_size+1, window_size, num_feature)
    - label_out: Label of the last element in the window. 
    
    Ex. If we have window_size = 10. data_out[0,:,:] = data[0:10,:]. label_out = label[9]
        The new data will be normalized for each window.
    """
    # step = int(window_size_ratio*window_size)
    step = 1
    new_num_data = math.floor((data.shape[0]-window_size+1)/step)+1

    data_out = np.zeros((new_num_data,window_size,data.shape[1]))
    label_out = np.zeros((new_num_data,label.shape[1]))
    
    idx=0
    for i in range(0,data.shape[0]-window_size+1,step):
        data_out[idx,:,:] = (data[i:i+window_size,:]-np.mean(data[i:i+window_size,:],axis=0))/np.std(data[i:i+window_size,:],axis=0)
        label_out[idx,:] = label[i+window_size-1,:] 
        idx += 1

    return data_out, label_out


class contact_dataset(Dataset):

    def __init__(self, data_path, label_path, window_size, device='cuda'):
        """
        At initialization we load .npy files for data and label.
        self.data: a 2D array of all data points. rows are time axis, columns are features. (num_data, num_features)
        self.label: a vector of the corresponding contact states (in decimal). (num_data, 1)
        """
        data = np.load(data_path)
        label = np.load(label_path)
        
        self.num_data = (data.shape[0]-window_size+1)
        self.window_size = window_size
        self.data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
        self.label = torch.from_numpy(label).type('torch.LongTensor').to(device)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        
        """
        In this function we get a batch size of our data and convert them into 3D tensor.

        self.data: a 2D array of all data points. rows are time axis, columns are features. (num_data, num_features)

        We use a sliding window of size = window size to create a 3rd dimension for the network 
        to inference along the time axis. After this, at each time step we will have
        window_size x num_features. Thus we can take window_size of data into consideration each time.

        Ex. If the window size = 10. new_data[0,:,:] = data[0:10,:], new_data[1,:,:] = data[1:11,:].
                                     new_label[0] = label[9],        new_label[1] = label[10].
        
        Output: 
        - data: (batch_size, window_size, num_features)
        - label: (batch_size, 1)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # this_data = torch.zeros((self.window_size, list(self.data.size())[1]))
        # this_label = torch.zeros((list(self.label.size())[1]))
        
        
        this_data = (self.data[idx:idx+self.window_size,:]-torch.mean(self.data[idx:idx+self.window_size,:],dim=0))\
                            /torch.std(self.data[idx:idx+self.window_size,:],dim=0)
        this_label = self.label[idx+self.window_size-1] 
           
            
        sample = {'data': this_data, 'label': this_label}

        return sample


# def main():
#     parser = argparse.ArgumentParser(description='Train network')
#     parser.add_argument('--data_folder', type=str, help='path to contact dataset', default="/home/justin/data/2021-02-21_contact_data_in_lab/cnn_data/")
#     args = parser.parse_args()

#     data = load_data_from_mat(args.data_folder,0.7,0.15)


# if __name__ == '__main__':
#     main()
import os
import argparse
import glob

import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt
import math

import torch
from torch.utils.data import Dataset, DataLoader

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

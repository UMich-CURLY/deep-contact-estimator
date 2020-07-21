import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from deep_learning_methods.utils.dp_funs import combine_Y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=512,
                      out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024,
                      out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024,
                      out_features=3),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 20)
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)


        out6 = out3.view(out3.shape[0], -1)
        out7 = self.fc(out6)
        return out7

class robot_dataset(Dataset):

    def __init__(self, data_path, Y_path, mean_path, std_path):
        data_X = np.load(data_path)
        data_mean = np.load(mean_path).reshape(1, -1)
        data_std = np.load(std_path).reshape(1, -1)
        data_X = ((data_X - data_mean) / data_std).astype(np.float32)

        data_Y = np.load(Y_path).astype(int)
        data_Y = combine_Y(data_Y).astype(int)

        self.size = data_X.shape[0]
        self.data_X = data_X
        self.data_Y = data_Y

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        single_data = torch.from_numpy(self.data_X[idx])
        single_lable = self.data_Y[idx]
        sample = {'data': single_data, 'label': single_lable}

        return sample
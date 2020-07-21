import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn.functional as F

class robot_dataset(Dataset):

    def __init__(self, data_path, Y_path, mean_path, std_path):
        data_X = np.load(data_path)
        # print(data_X.shape)
        data_mean = np.load(mean_path).reshape(1, -1)
        data_std = np.load(std_path).reshape(1, -1)

        data_X = ((data_X - data_mean) / data_std).astype(np.float32)

        data_Y = np.load(Y_path).astype(int)

        self.size = data_X.shape[0]
        self.data_X = data_X
        self.data_Y = data_Y

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        single_data = torch.from_numpy(cv.resize(self.data_X[idx], (20, 40)))
        # print(single_data.shape)
        single_lable = self.data_Y[idx]
        sample = {'data': single_data, 'label': single_lable}

        return sample

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        if self.same_shape:
            stride = 1
        else:
            stride = 2

        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)

        if not self.same_shape:
            x = self.conv3(x)

        return F.relu(x+out, True)

class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )

        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )

        self.block4 = nn.Sequential(
            residual_block(128, 512, False),
            residual_block(512, 512),
            nn.AvgPool2d(3)
        )

        self.classifier = nn.Linear(512, num_classes)
        # self.classifier = nn.Sequential(nn.Linear(512, 256),
        #                                 nn.Dropout(0.3),
        #                                 nn.Linear(256, num_classes))

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 20, 40)
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))

        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x





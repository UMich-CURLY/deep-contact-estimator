import sys
sys.path.append('../')

from deep_learning_methods.utils.dp_funs import *
from deep_learning_methods.utils.both_leg_fun import *
import os
import argparse

# input parameters
parser = argparse.ArgumentParser(description='Train network')
parser.add_argument('--path', type=str, help='path to training and testing data', default="/home/justin/data/yichent")
parser.add_argument('--train', type=str, help='file name for the training data', default="train_data_0.npy")
parser.add_argument('--test', type=str, help='file name for the testing data', default="test_data_0.npy")
parser.add_argument('--train_y', type=str, help='file name for the training label', default="train_Y.npy")
parser.add_argument('--test_y', type=str, help='file name for the testing label', default="test_Y.npy")
parser.add_argument('--mean', type=str, help='file name for the data mean', default="data_mean.npy")
parser.add_argument('--std', type=str, help='file name for the data std', default="data_std.npy")
args = parser.parse_args()

# set up path for data
train = os.path.join(args.path, args.train)
test = os.path.join(args.path, args.test)
train_Y = os.path.join(args.path, args.train_y)
test_Y = os.path.join(args.path, args.test_y)

mean = os.path.join(args.path, args.mean)
std = os.path.join(args.path, args.std)

train_data = robot_dataset(data_path=train, Y_path=train_Y, mean_path=mean, std_path=std)
train_dataloader = DataLoader(dataset=train_data, batch_size=30, shuffle=True, num_workers=8, pin_memory=True)
test_data = robot_dataset(data_path=test, Y_path=test_Y, mean_path=mean, std_path=std)
test_dataloader = DataLoader(dataset=test_data, batch_size=100, shuffle=True, num_workers=8, pin_memory=True)

print(Net())
net = Net()
net = net.cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(15):
    running_loss = 0.0
    for i, samples in tqdm(enumerate(train_dataloader, start=0)):
        inputs, labels = samples['data'], samples['label']
        inputs = inputs.cuda()
        labels = labels.cuda()
        print(inputs.shape)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            print("{} epoch, {} objects, loss: {}".format(epoch+1, i+1, running_loss / 1000))
            running_loss = 0.0
    train_acc = get_acc(train_dataloader, net)
    test_acc = get_acc(test_dataloader, net)
    print("{} epoch: train acc:{}%, test:{}%".format(epoch+1, train_acc*100, test_acc*100))

print("training finished")

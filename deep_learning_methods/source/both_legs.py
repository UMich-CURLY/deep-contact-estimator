from deep_learning_methods.utils.dp_funs import *
from deep_learning_methods.utils.both_leg_fun import *

PATH = input("path: ")
# PATH = '/home/yicheng/Documents/imu project/contact_estimate'

train ="{}/deep_learning_methods/data/train_data_0.npy".format(PATH)
test = "{}/deep_learning_methods/data/test_data_0.npy".format(PATH)
train_Y = "{}/deep_learning_methods/data/train_Y.npy".format(PATH)
test_Y = "{}/deep_learning_methods/data/test_Y.npy".format(PATH)

mean = "{}/deep_learning_methods/data/data_mean.npy".format(PATH)
std = "{}/deep_learning_methods/data/data_std.npy".format(PATH)

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

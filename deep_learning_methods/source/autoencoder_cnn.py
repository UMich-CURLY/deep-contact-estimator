from deep_learning_methods.utils.dp_funs import *
from deep_learning_methods.utils.win_data_fun import robot_dataset
from deep_learning_methods.utils.autoencoder_cnn_fun import *
from sklearn.cluster import KMeans

PATH = input("path: ")
# PATH = '/home/yicheng/Documents/imu project/contact_estimate'

mean = "{}/deep_learning_methods/data/data_mean.npy".format(PATH)
std = "{}/deep_learning_methods/data/data_std.npy".format(PATH)

syn = input("data syn or unsyn: ")

if syn == "syn":

    train ="{}/deep_learning_methods/data/train_data_10_WIN_syn.npy".format(PATH)
    test = "{}/deep_learning_methods/data/test_data_10_WIN_syn.npy".format(PATH)
    train_Y = "{}/deep_learning_methods/data/train_Y_10_WIN_syn.npy".format(PATH)
    test_Y = "{}/deep_learning_methods/data/train_data_10_WIN_syn.npy".format(PATH)

else:
    train = "{}/deep_learning_methods/data/train_data_10_WIN_unsyn.npy".format(PATH)
    test = "{}/deep_learning_methods/data/test_data_10_WIN_unsyn.npy".format(PATH)
    train_Y = "{}/deep_learning_methods/data/train_Y_10_WIN_unsyn.npy".format(PATH)
    test_Y = "{}/deep_learning_methods/data/train_data_10_WIN_unsyn.npy".format(PATH)

train_data = robot_dataset(data_path=train, Y_path=train_Y, mean_path=mean, std_path=std)
train_dataloader = DataLoader(dataset=train_data, batch_size=30, shuffle=True, num_workers=8, pin_memory=True)
test_data = robot_dataset(data_path=test, Y_path=test_Y, mean_path=mean, std_path=std)
test_dataloader = DataLoader(dataset=test_data, batch_size=100, shuffle=True, num_workers=8, pin_memory=True)

atcd = autoencoder()
atcd = atcd.cuda()

print("start training")
import torch.optim as optim
criterion = nn.MSELoss()

optimizier = optim.Adam(atcd.parameters(), lr=0.01, weight_decay=1e-4)
for epoch in range(10):
    running_loss = 0.0
    for i, samples in tqdm(enumerate(train_dataloader, start=0)):
        inputs, labels = samples['data'], samples['label']
        inputs = inputs.cuda()

        optimizier.zero_grad()
        outputs = atcd(inputs)

        loss = criterion(outputs.reshape(outputs.shape[0], 10, 20), inputs)
        loss.backward()
        optimizier.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            print("{} epoch, {} objects, loss: {}".format(epoch+1, i+1, running_loss / 1000))
            running_loss = 0.0


torch.save(atcd.state_dict(), 'autoencoder_parm_cnn.pkl')
print("training finished")
model = Encoder()
save_model = torch.load('autoencoder_parm_cnn.pkl')
model_dict =  model.state_dict()
state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
print(state_dict.keys())

model_dict.update(state_dict)
model.load_state_dict(model_dict)

model = model.cuda()

train_X, train_Y = transfer_data(train_dataloader, model)
test_X, test_Y = transfer_data(test_dataloader, model)

km = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=100000,
    n_clusters=3, n_init=20, n_jobs=-1, precompute_distances='auto',
    random_state=None, verbose=0)

km.fit(train_X)

rst_test = km.predict(test_X)
test_acc = km_get_acc(test_Y.astype(int), rst_test, 3)


print("test_acc: ", test_acc)

rst_train = km.predict(train_X)
train_acc = km_get_acc(train_Y.astype(int), rst_train, 3)

print("train_acc: ", train_acc)

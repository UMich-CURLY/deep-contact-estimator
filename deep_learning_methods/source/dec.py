from deep_learning_methods.utils.dec_fun import *

PATH = input("path: ")
# PATH = '/home/yicheng/Documents/imu project/contact_estimate'

mean = "{}/deep_learning_methods/data/data_mean.npy".format(PATH)
std = "{}/deep_learning_methods/data/data_std.npy".format(PATH)

syn = input("data syn or unsyn: ")

if syn == "syn":

    train = "{}/deep_learning_methods/data/train_data_10_WIN_syn.npy".format(PATH)
    test = "{}/deep_learning_methods/data/test_data_10_WIN_syn.npy".format(PATH)
    train_Y = "{}/deep_learning_methods/data/train_Y_10_WIN_syn.npy".format(PATH)
    test_Y = "{}/deep_learning_methods/data/train_data_10_WIN_syn.npy".format(PATH)

else:
    train = "{}/deep_learning_methods/data/train_data_10_WIN_unsyn.npy".format(PATH)
    test = "{}/deep_learning_methods/data/test_data_10_WIN_unsyn.npy".format(PATH)
    train_Y = "{}/deep_learning_methods/data/train_Y_10_WIN_unsyn.npy".format(PATH)
    test_Y = "{}/deep_learning_methods/data/train_data_10_WIN_unsyn.npy".format(PATH)

atcd = Autoencoder()

pretrain(data_path=train, Y_path=train_Y, mean_path=mean, std_path=std, model=atcd, num_epochs=10)
dec = DEC(n_clusters=3, autoencoder=atcd, hidden=10, cluster_centers=None, alpha=1.0)
training(data_path=train, Y_path=train_Y,
         test_X_path=test, test_Y_path=test_Y,
         mean_path=mean, std_path=std, model=dec, num_epochs=20)

import numpy as np
import scipy.io as sio






def main():
    train_data_0 = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/train_data_0.npy")
    train_Y = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/train_Y.npy")
    test_data_0 = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/test_data_0.npy")
    test_Y = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/test_Y.npy")
    train_data_10_WIN_syn = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/train_data_10_WIN_syn.npy")
    train_Y_10_WIN_syn = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/train_Y_10_WIN_syn.npy")
    train_data_10_WIN_unsyn = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/train_data_10_WIN_unsyn.npy")
    train_Y_10_WIN_unsyn = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/train_Y_10_WIN_unsyn.npy")
    test_data_10_WIN_unsyn = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/test_data_10_WIN_unsyn.npy")
    test_Y_10_WIN_unsyn = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/test_Y_10_WIN_unsyn.npy")
    test_data_10_WIN_syn = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/test_data_10_WIN_syn.npy")
    test_Y_10_WIN_syn = np.load("/home/justin/data/cassie_contact_data/deep_learning_methods/data_archive/test_Y_10_WIN_syn.npy")

    out = {}
    out['train_data_0'] = train_data_0
    out['train_Y'] = train_Y

    out['test_data_0'] = test_data_0
    out['test_Y'] = test_Y

    out['train_data_10_WIN_syn'] = train_data_10_WIN_syn
    out['train_Y_10_WIN_syn'] = train_Y_10_WIN_syn

    out['train_data_10_WIN_unsyn'] = train_data_10_WIN_unsyn
    out['train_Y_10_WIN_unsyn'] = train_Y_10_WIN_unsyn

    out['test_data_10_WIN_unsyn'] = test_data_10_WIN_unsyn
    out['test_Y_10_WIN_unsyn'] = test_Y_10_WIN_unsyn

    out['test_data_10_WIN_syn'] = test_data_10_WIN_syn
    out['test_Y_10_WIN_syn'] = test_Y_10_WIN_syn

    sio.savemat("/home/justin/data/cassie_contact_data/deep_learning_methods/data/data.mat",out)

if __name__ == '__main__':
    main()
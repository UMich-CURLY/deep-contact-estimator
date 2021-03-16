import os
import argparse
import glob

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import yaml



def main():

    file_path = "/media/curly_ssd_justin/DockerFolder/code/deep-contact-estimator/results/0315_ws150_lr1e-4_2block_drop_out_history.npy"
    # file_path = "/home/justin/code/deep-contact-estimator/results/0309_ws150_lr1e-5.npy"

    # np_load_old = np.load

    # # modify the default parameters of np.load
    # np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    # # call load_data with allow_pickle implicitly set to true
    # history = np.load(file_path)

    # # restore np.load for future normal usage
    # np.load = np_load_old

    history = np.load(file_path)
    
    print("history:", np.shape(history))
    
    train_loss_history = history.item().get('train_loss_history')
    train_loss_history = np.mean(train_loss_history, axis=1)
    val_loss_history = history.item().get('val_loss_history')
    val_loss_history = np.mean(val_loss_history, axis=1)
    train_acc = history.item().get('train_acc')
    val_acc = history.item().get('val_acc')

    
    
    epochs = range(1,len(train_loss_history)+1)
    plt.plot(epochs, train_loss_history, 'g', label='Training loss')
    plt.plot(epochs, val_loss_history, 'b', label='Validation loss')
    plt.title('Training loss vs. Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



    # epochs = range(1,len(train_loss_history)+1)
    plt.plot(epochs, train_acc, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training accuracy vs. Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
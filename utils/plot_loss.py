import os
import argparse
import glob

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import yaml



def main():

    file_path = "/media/curly_ssd_justin/DockerFolder/code/deep-contact-estimator/results/0308_stacked_test_loss_history.npy"

    history = np.load(file_path)
    train_loss_history = history.item().get('train_loss_history')
    
    print(np.shape(loss_history))
    loss_train = np.mean(loss_history,axis=1)
    epochs = range(1,len(loss_train)+1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
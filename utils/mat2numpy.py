import os
import argparse
import glob

import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt
import math
import yaml

def mat2numpy_one_seq(data_pth, save_pth):
    """
    Load data from .mat file and genearate numpy files without splitting into train/val/test.

    Inputs:
    - data_pth: path to mat data folder
    - save_pth: path to numpy saving directory.

    Data should be stored in .mat file, and contain:
    - q: joint encoder value (num0_data,12)
    - qd: joint angular velocity (num_data,12)
    - p: foot position from FK (num_data,12)
    - v: foot velocity from FK (num_data,12)
    
    - imu_acc: linear acceleration from imu (num_data,3)
    - imu_omega: angular velocity from imu (num_data,3)
    - contacts: contact data (num_data,4)
                contacts are stored as binary values, in the order
                of right_front, left_front, right_hind, left_hind.

                FRONT
                1 0  RIGHT
                3 2
                BACK

                Contact value will be treated as binary values in the
                order of contacts[0], contacts[1], contacts[2], contacts[3]
                and be converted to decimal value in this function.

                Ex. [1,0,0,1] -> 9
                    [0,1,1,0] -> 6

    - tau_est (optional): estimated control torque (num_data,12)
    - F (optional): ground reaction force

    Output:
    """

    for data_name in glob.glob(data_pth+'*'): 
        
        print("loading... ", data_name)

        # load data
        raw_data = sio.loadmat(data_name)

        contacts = raw_data['contacts']
        terrain_label = raw_data['terrain_type']
        q = raw_data['q']
        p = raw_data['p']
        qd = raw_data['qd']
        v = raw_data['v']
        acc = raw_data['imu_acc']
        omega = raw_data['imu_omega']

        # tau_est = raw_data['tau_est']
        # F = raw_data['F']

        # concatenate current data. 
        data = np.concatenate((q,qd,acc,omega,p,v),axis=1)
        
        # convert labels from binary to decimal
        contact_label = binary2decimal(contacts).reshape((-1,1)) 

        print("Saving data to: "+save_pth+os.path.splitext(os.path.basename(data_name))[0]+".npy")

        data_to_save = {'data':data,'contact_label': contact_label,'terrain_label':terrain_label}
        np.save(save_pth+os.path.splitext(os.path.basename(data_name))[0]+".npy",data_to_save)

        # np.save(save_pth+os.path.splitext(os.path.basename(data_name))[0]+"_label.npy",contact_label)

        print("Done!")


def mat2numpy_split(data_pth, save_pth, train_ratio=0.7, val_ratio=0.15):
    """
    Load data from .mat file, concatenate into numpy array, and save as train/val/test.
    Inputs:
    - data_pth: path to mat data folder
    - save_pth: path to numpy saving directory.
    - train_ratio: ratio of training data
    - val_ratio: ratio of validation data
    Data should be stored in .mat file, and contain:
    - q: joint encoder value (num_data,12)
    - qd: joint angular velocity (num_data,12)
    - p: foot position from FK (num_data,12)
    - v: foot velocity from FK (num_data,12)
    - imu_acc: linear acceleration from imu (num_data,3)
    - imu_omega: angular velocity from imu (num_data,3)
    - contacts: contact data (num_data,4)
                contacts are stored as binary values, in the order
                of right_front, left_front, right_hind, left_hind.
                FRONT
                1 0  RIGHT
                3 2
                BACK
                Contact value will be treated as binary values in the
                order of contacts[0], contacts[1], contacts[2], contacts[3]
                and be converted to decimal value in this function.
                Ex. [1,0,0,1] -> 9
                    [0,1,1,0] -> 6
                     
    - tau_est (optional): estimated control torque (num_data,12)
    - F (optional): ground reaction force

    Output:
    - 
    """

    num_features = 54    
    train_data = np.zeros((0,num_features))
    val_data = np.zeros((0,num_features))
    test_data = np.zeros((0,num_features))
    train_contact_label = np.zeros((0,1))
    val_contact_label = np.zeros((0,1))
    test_contact_label = np.zeros((0,1))
    train_terrain_label = np.zeros((0,1))
    val_terrain_label = np.zeros((0,1))
    test_terrain_label = np.zeros((0,1))
    seq_len_train = []
    seq_len_val = []
    seq_len_test = []

    # for all dataset in the folder
    for data_name in glob.glob(data_pth+'*'): 
        
        print("loading... ", data_name)

        # load data
        raw_data = sio.loadmat(data_name)

        contacts = raw_data['contacts']
        overall_terrain_type = raw_data['overall_terrain_type']
        cur_terrain_label = raw_data['terrain_type']
        q = raw_data['q']
        p = raw_data['p']
        qd = raw_data['qd']
        v = raw_data['v']
        acc = raw_data['imu_acc']
        omega = raw_data['imu_omega']

        # tau_est = raw_data['tau_est']
        # F = raw_data['F']
        
        # concatenate current data. First we try without GRF
        cur_data = np.concatenate((q,qd,acc,omega,p,v),axis=1)
        
        # separate data into train/val/test
        num_data = np.shape(q)[0]
        num_train = int(train_ratio*num_data)
        num_val = int(val_ratio*num_data)
        num_test = num_data-num_train-num_val
        cur_val = cur_data[:num_val,:]
        cur_test = cur_data[num_val:num_val+num_test,:]
        cur_train = cur_data[num_val+num_test:,:]

        seq_len_train.append(num_train)
        seq_len_val.append(num_val)
        seq_len_test.append(num_test)

        # stack with all other sequences
        train_data = np.vstack((train_data,cur_train))
        val_data = np.vstack((val_data,cur_val))
        test_data = np.vstack((test_data,cur_test))

        
        # convert labels from binary to decimal
        cur_contact_label = binary2decimal(contacts).reshape((-1,1))   

        # stack labels 
        val_contact_label = np.vstack((val_contact_label,cur_contact_label[:num_val,:]))
        test_contact_label = np.vstack((test_contact_label,cur_contact_label[num_val:num_val+num_test,:]))
        train_contact_label = np.vstack((train_contact_label,cur_contact_label[num_val+num_test:,:]))

        val_terrain_label = np.vstack((val_terrain_label,cur_terrain_label[:num_val,:]))
        test_terrain_label = np.vstack((test_terrain_label,cur_terrain_label[num_val:num_val+num_test,:]))
        train_terrain_label = np.vstack((train_terrain_label,cur_terrain_label[num_val+num_test:,:]))


        # break
    train_contact_label = train_contact_label.reshape(-1,)
    val_contact_label = val_contact_label.reshape(-1,)
    test_contact_label = test_contact_label.reshape(-1,)

    train_terrain_label = train_terrain_label.reshape(-1,)
    val_terrain_label = val_terrain_label.reshape(-1,)
    test_terrain_label = test_terrain_label.reshape(-1,)
    
    train = {'data':train_data,'contact_label':train_contact_label,'terrain_label':train_terrain_label, 'seq_len':np.array(seq_len_train)}
    val = {'data':val_data,'contact_label':val_contact_label,'terrain_label':val_terrain_label, 'seq_len':np.array(seq_len_val)}
    test = {'data':test_data,'contact_label':test_contact_label,'terrain_label':test_terrain_label, 'seq_len':np.array(seq_len_test)}

    print("Saving data...")
    
    np.save(save_pth+"train.npy",train)
    np.save(save_pth+"val.npy",val)
    np.save(save_pth+"test.npy",test)
    # np.save(save_pth+"train_label.npy",train_contact_label)
    # np.save(save_pth+"val_label.npy",val_contact_label)
    # np.save(save_pth+"test_label.npy",test_contact_label)

    print("Generated ", train_data.shape[0], " training data.")
    print("Generated ", val_data.shape[0], " validation data.")
    print("Generated ", test_data.shape[0], " test data.")
    
    # print(train_data.shape[0])
    # print(val_data.shape[0])
    # print(test_data.shape[0])

    print("Done!")
    # return data

def binary2decimal(a, axis=-1):
    return np.right_shift(np.packbits(a, axis=axis), 8 - a.shape[axis]).squeeze()


def main():

    parser = argparse.ArgumentParser(description='Convert mat to numpy.')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/mat2numpy_config.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name))

    if config['mode']=='train':
        mat2numpy_split(config['mat_folder'],config['save_path'],config['train_ratio'],config['val_ratio'])
    elif config['mode']=='inference':
        mat2numpy_one_seq(config['mat_folder'],config['save_path'])

if __name__ == '__main__':
    main()

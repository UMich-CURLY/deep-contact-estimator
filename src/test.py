import os
import argparse
import glob
import sys
sys.path.append('..')
import yaml
from tqdm import tqdm

import torch.optim as optim

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score

from contact_cnn import *
from utils.data_handler import *

def compute_precision(bin_pred_arr, bin_gt_arr, pred_arr, gt_arr):


    precision_of_class = precision_score(gt_arr,pred_arr,average='weighted')
    precision_of_all_legs = precision_score(bin_gt_arr.flatten(),bin_pred_arr.flatten())
    precision_of_legs = []
    for i in range(4):
        precision_of_legs.append(precision_score(bin_gt_arr[:,i],bin_pred_arr[:,i]))

    return precision_of_class, precision_of_legs, precision_of_all_legs

def compute_jaccard(bin_pred_arr, bin_gt_arr, pred_arr, gt_arr):


    jaccard_of_class = jaccard_score(gt_arr,pred_arr,average='weighted')
    jaccard_of_all_legs = jaccard_score(bin_gt_arr.flatten(),bin_pred_arr.flatten())
    jaccard_of_legs = []
    for i in range(4):
        jaccard_of_legs.append(jaccard_score(bin_gt_arr[:,i],bin_pred_arr[:,i]))

    return jaccard_of_class, jaccard_of_legs, jaccard_of_all_legs

def compute_accuracy(dataloader, model):
    # compute accuracy in batch

    num_correct = 0
    num_data = 0
    correct_per_leg = np.zeros(4)
    bin_pred_arr = np.zeros((0,4))
    bin_gt_arr = np.zeros((0,4))
    pred_arr = np.zeros((0))
    gt_arr = np.zeros((0))
    with torch.no_grad():
        for sample in tqdm(dataloader): # tqdm shows a smart progress meter
            input_data = sample['data']
            gt_label = sample['label']

            output = model(input_data)

            _, prediction = torch.max(output,1)
            # print("gt_label: ", gt_label)
            # print("prediction: " , prediction)

            

            bin_pred = decimal2binary(prediction)
            bin_gt = decimal2binary(gt_label)

            bin_pred_arr = np.vstack((bin_pred_arr,bin_pred.cpu().numpy()))
            bin_gt_arr = np.vstack((bin_gt_arr,bin_gt.cpu().numpy()))

            pred_arr = np.hstack((pred_arr,prediction.cpu().numpy()))
            gt_arr = np.hstack((gt_arr,gt_label.cpu().numpy()))

            correct_per_leg += (bin_pred==bin_gt).sum(axis=0).cpu().numpy()
            num_data += input_data.size(0)
            num_correct += (prediction==gt_label).sum().item()


    return num_correct/num_data, correct_per_leg/num_data, bin_pred_arr, bin_gt_arr, pred_arr, gt_arr

def decimal2binary(x):
    mask = 2**torch.arange(4-1,-1,-1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    parser = argparse.ArgumentParser(description='Test the contcat network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/test_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name))

    test_data = contact_dataset(data_path=config['data_folder']+"test_lcm.npy",\
                                label_path=config['data_folder']+"test_label_lcm.npy",\
                                window_size=config['window_size'], device=device)
    test_dataloader = DataLoader(dataset=test_data, batch_size=config['batch_size'])
    # test_dataloader = DataLoader(dataset=test_data, batch_size=1)



    # init network
    model = contact_cnn()

    checkpoint = torch.load(config['model_load_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    test_acc, acc_per_leg, bin_pred_arr, bin_gt_arr, pred_arr, gt_arr = compute_accuracy(test_dataloader, model)
    precision_of_class, precision_of_legs, precision_of_all_legs = compute_precision(bin_pred_arr,bin_gt_arr,pred_arr, gt_arr)
    jaccard_of_class, jaccard_of_legs, jaccard_of_all_legs = compute_jaccard(bin_pred_arr,bin_gt_arr,pred_arr, gt_arr)

    print("Test accuracy in terms of class is: %.4f" % test_acc)
    print("Accuracy of leg 0 is: %.4f" % acc_per_leg[0])
    print("Accuracy of leg 1 is: %.4f" % acc_per_leg[1])
    print("Accuracy of leg 2 is: %.4f" % acc_per_leg[2])
    print("Accuracy of leg 3 is: %.4f" % acc_per_leg[3])
    print("Accuracy is: %.4f" % (np.sum(acc_per_leg)/4.0))
    print("---------------")
    print("Precision of class is: %.4f" % precision_of_class)
    print("Precision of leg 0 is: %.4f" % precision_of_legs[0])
    print("Precision of leg 1 is: %.4f" % precision_of_legs[1])
    print("Precision of leg 2 is: %.4f" % precision_of_legs[2])
    print("Precision of leg 3 is: %.4f" % precision_of_legs[3])
    print("Precision of all legs is: %.4f" % precision_of_all_legs)
    print("---------------")
    print("jaccard of class is: %.4f" % jaccard_of_class)
    print("jaccard of leg 0 is: %.4f" % jaccard_of_legs[0])
    print("jaccard of leg 1 is: %.4f" % jaccard_of_legs[1])
    print("jaccard of leg 2 is: %.4f" % jaccard_of_legs[2])
    print("jaccard of leg 3 is: %.4f" % jaccard_of_legs[3])
    print("jaccard of all legs is: %.4f" % jaccard_of_all_legs)

    # print(test_acc)
    # print(acc_per_leg[0])
    # print(acc_per_leg[1])
    # print(acc_per_leg[2])
    # print(acc_per_leg[3])
    # print((np.sum(acc_per_leg)/4.0))
    # print("---------------")
    # print(precision_of_class)
    # print(precision_of_legs[0])
    # print(precision_of_legs[1])
    # print(precision_of_legs[2])
    # print(precision_of_legs[3])
    # print(precision_of_all_legs)
    # print(jaccard_of_class)
    # print(jaccard_of_legs[0])
    # print(jaccard_of_legs[1])
    # print(jaccard_of_legs[2])
    # print(jaccard_of_legs[3])
    # print(jaccard_of_all_legs)

if __name__ == '__main__':
    main()
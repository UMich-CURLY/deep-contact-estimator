import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm

import torch.optim as optim

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix

from contact_cnn import *
from utils.data_handler import *

def compute_confusion_mat(bin_contact_pred_arr, bin_contact_gt_arr):
    
    confusion_mat = {}
    
    confusion_mat['leg_rf'] = confusion_matrix(bin_contact_gt_arr[:,0],bin_contact_pred_arr[:,0], labels=[0,1])
    confusion_mat['leg_lf'] = confusion_matrix(bin_contact_gt_arr[:,1],bin_contact_pred_arr[:,1], labels=[0,1])
    confusion_mat['leg_rh'] = confusion_matrix(bin_contact_gt_arr[:,2],bin_contact_pred_arr[:,2], labels=[0,1])
    confusion_mat['leg_lh'] = confusion_matrix(bin_contact_gt_arr[:,3],bin_contact_pred_arr[:,3], labels=[0,1])
    confusion_mat['total'] = confusion_mat['leg_rf'] + confusion_mat['leg_lf'] + confusion_mat['leg_rh'] + confusion_mat['leg_lh']
    confusion_mat['total_ratio'] = confusion_mat['total'] / np.sum(confusion_mat['total'])
    
    # false negative and false postivie rate
    # false negative = FN/P; false positive = FP/N
    fn_rate = {}
    fp_rate = {}

    fn_rate['leg_rf'] = confusion_mat['leg_rf'][0,1] / (confusion_mat['leg_rf'][0,0]+confusion_mat['leg_rf'][0,1])
    fn_rate['leg_lf'] = confusion_mat['leg_lf'][0,1] / (confusion_mat['leg_lf'][0,0]+confusion_mat['leg_lf'][0,1])
    fn_rate['leg_rh'] = confusion_mat['leg_rh'][0,1] / (confusion_mat['leg_rh'][0,0]+confusion_mat['leg_rh'][0,1])
    fn_rate['leg_lh'] = confusion_mat['leg_lh'][0,1] / (confusion_mat['leg_lh'][0,0]+confusion_mat['leg_lh'][0,1])
    fn_rate['total'] = confusion_mat['total'][0,1] / (confusion_mat['total'][0,0]+confusion_mat['total'][0,1])

    fp_rate['leg_rf'] = confusion_mat['leg_rf'][1,0] / (confusion_mat['leg_rf'][1,0] + confusion_mat['leg_rf'][1,1]) 
    fp_rate['leg_lf'] = confusion_mat['leg_lf'][1,0] / (confusion_mat['leg_lf'][1,0] + confusion_mat['leg_lf'][1,1])
    fp_rate['leg_rh'] = confusion_mat['leg_rh'][1,0] / (confusion_mat['leg_rh'][1,0] + confusion_mat['leg_rh'][1,1])
    fp_rate['leg_lh'] = confusion_mat['leg_lh'][1,0] / (confusion_mat['leg_lh'][1,0] + confusion_mat['leg_lh'][1,1])
    fp_rate['total'] = confusion_mat['total'][1,0] / (confusion_mat['total'][1,0] + confusion_mat['total'][1,1])

    return confusion_mat, fn_rate, fp_rate


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
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_label = sample['label']

            output = model(input_data)

            _, prediction = torch.max(output,1)

            

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

    test_data = contact_dataset(data_path=config['data_folder']+"test.npy",\
                                label_path=config['data_folder']+"test_label.npy",\
                                window_size=config['window_size'],device=device)
    test_dataloader = DataLoader(dataset=test_data, batch_size=config['batch_size'])


    # init network
    model = contact_cnn()

    checkpoint = torch.load(config['model_load_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    test_acc, acc_per_leg, bin_pred_arr, bin_gt_arr, pred_arr, gt_arr = compute_accuracy(test_dataloader, model)
    precision_of_class, precision_of_legs, precision_of_all_legs = compute_precision(bin_pred_arr,bin_gt_arr,pred_arr, gt_arr)
    jaccard_of_class, jaccard_of_legs, jaccard_of_all_legs = compute_jaccard(bin_pred_arr,bin_gt_arr,pred_arr, gt_arr)
    confusion_mat, fn_rate, fp_rate = compute_confusion_mat(bin_pred_arr,bin_gt_arr)


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
    print("---------------")
    print("confusion matrix of leg rf is: ")
    print(confusion_mat['leg_rf'])
    print("confusion matrix of leg lf is: ")
    print(confusion_mat['leg_lf'])
    print("confusion matrix of leg rh is: ")
    print(confusion_mat['leg_rh'])
    print("confusion matrix of leg lh is: ")
    print(confusion_mat['leg_lh'])
    print("confusion matrix sum is: ")
    print(confusion_mat['total'])
    print("confusion matrix ratio: ")
    print(confusion_mat['total_ratio'])
    print("---------------")
    print("false negative rate of leg rf is: %.4f" % fn_rate['leg_rf'])
    print("false negative rate of leg lf is: %.4f" % fn_rate['leg_lf'])
    print("false negative rate of leg rh is: %.4f" % fn_rate['leg_rh'])
    print("false negative rate of leg lh is: %.4f" % fn_rate['leg_lh'])
    print("AVG false negative rate is: %.4f" % fn_rate['total'])
    print("---------------")
    print("false positive rate of leg rf is: %.4f" % fp_rate['leg_rf'])
    print("false positive rate of leg lf is: %.4f" % fp_rate['leg_lf'])
    print("false positive rate of leg rh is: %.4f" % fp_rate['leg_rh'])
    print("false positive rate of leg lh is: %.4f" % fp_rate['leg_lh'])
    print("AVG false positive rate is: %.4f" % fp_rate['total'])
    print("---------------")

    print(test_acc)
    print(acc_per_leg[0])
    print(acc_per_leg[1])
    print(acc_per_leg[2])
    print(acc_per_leg[3])
    print((np.sum(acc_per_leg)/4.0))
    print("---------------")
    print(precision_of_class)
    print(precision_of_legs[0])
    print(precision_of_legs[1])
    print(precision_of_legs[2])
    print(precision_of_legs[3])
    print(precision_of_all_legs)
    print("---------------")
    print(jaccard_of_class)
    print(jaccard_of_legs[0])
    print(jaccard_of_legs[1])
    print(jaccard_of_legs[2])
    print(jaccard_of_legs[3])
    print(jaccard_of_all_legs)
    print("---------------")
    print(fn_rate['leg_rf'])
    print(fn_rate['leg_lf'])
    print(fn_rate['leg_rh'])
    print(fn_rate['leg_lh'])
    print(fn_rate['total'])
    print("---------------")
    print(fp_rate['leg_rf'])
    print(fp_rate['leg_lf'])
    print(fp_rate['leg_rh'])
    print(fp_rate['leg_lh'])
    print(fp_rate['total'])

if __name__ == '__main__':
    main()
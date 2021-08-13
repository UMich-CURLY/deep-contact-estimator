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

from mtl_models import MultiTaskModel
from loss import MultiTaskLoss
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

    num_correct_contact = 0
    num_correct_terrain = 0
    num_data = 0
    correct_per_leg = np.zeros(4)
    bin_contact_pred_arr = np.zeros((0,4))
    bin_contact_gt_arr = np.zeros((0,4))
    contact_pred_arr = np.zeros((0))
    contact_gt_arr = np.zeros((0))
    terrain_pred_arr = np.zeros((0))
    terrain_gt_arr = np.zeros((0))
    prediction = {}
    for sample in tqdm(dataloader):
        input_data = sample['data']
        labels = {
              "contact" : sample['contact_label'] ,
              "terrain" : sample['terrain_label'] 
            }

        output = model(input_data)
        _, prediction['contact'] = torch.max(output['contact'],1)
        _, prediction['terrain'] = torch.max(output['terrain'],1)

        bin_contact_pred = decimal2binary(prediction['contact'])
        bin_contact_gt = decimal2binary(labels['contact'])

        bin_contact_pred_arr = np.vstack((bin_contact_pred_arr,bin_contact_pred.cpu().numpy()))
        bin_contact_gt_arr = np.vstack((bin_contact_gt_arr,bin_contact_gt.cpu().numpy()))

        contact_pred_arr = np.hstack((contact_pred_arr, prediction['contact'].cpu().numpy()))
        contact_gt_arr = np.hstack((contact_gt_arr, labels['contact'].cpu().numpy()))

        terrain_pred_arr = np.hstack((terrain_pred_arr, prediction['terrain'].cpu().numpy()))
        terrain_gt_arr = np.hstack((terrain_gt_arr, labels['terrain'].cpu().numpy()))

        correct_per_leg += (bin_contact_pred==bin_contact_gt).sum(axis=0).cpu().numpy()
        num_data += input_data.size(0)
        num_correct_terrain += (prediction['terrain']==labels['terrain']).sum().item()
        num_correct_contact += (prediction['contact']==labels['contact']).sum().item()

    return num_correct_contact/num_data, correct_per_leg/num_data, num_correct_terrain/num_data, bin_contact_pred_arr, bin_contact_gt_arr, contact_pred_arr, contact_gt_arr, terrain_pred_arr, terrain_gt_arr

def compute_confusion_mat(tasks_dict, bin_contact_pred_arr, bin_contact_gt_arr, contact_pred_arr, contact_gt_arr, terrain_pred_arr, terrain_gt_arr):
    
    confusion_mat = {}
    
    confusion_mat['leg_rf'] = confusion_matrix(bin_contact_gt_arr[:,0],bin_contact_pred_arr[:,0], labels=[0,1])
    confusion_mat['leg_lf'] = confusion_matrix(bin_contact_gt_arr[:,1],bin_contact_pred_arr[:,1], labels=[0,1])
    confusion_mat['leg_rh'] = confusion_matrix(bin_contact_gt_arr[:,2],bin_contact_pred_arr[:,2], labels=[0,1])
    confusion_mat['leg_lh'] = confusion_matrix(bin_contact_gt_arr[:,3],bin_contact_pred_arr[:,3], labels=[0,1])

    confusion_mat['terrain'] = confusion_matrix(terrain_gt_arr,terrain_pred_arr, labels=[0,1,2,3,4,5,6,7,8])

    
    return confusion_mat

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
                                window_size=config['window_size'],device=device)
    test_dataloader = DataLoader(dataset=test_data, batch_size=config['batch_size'])


    # init network
    tasks_dict = {
      "contact" : 16,
      "terrain" : 9
    }
    model = MultiTaskModel(tasks_dict,config)
    checkpoint = torch.load(config['model_load_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    test_acc = {}
    test_acc['contact'], acc_per_leg, test_acc['terrain'], bin_contact_pred_arr, bin_contact_gt_arr, \
    contact_pred_arr, contact_gt_arr, terrain_pred_arr, terrain_gt_arr = compute_accuracy(test_dataloader, model)
    precision_of_class, precision_of_legs, precision_of_all_legs = compute_precision(bin_contact_pred_arr, bin_contact_gt_arr, contact_pred_arr, contact_gt_arr)
    jaccard_of_class, jaccard_of_legs, jaccard_of_all_legs = compute_jaccard(bin_contact_pred_arr, bin_contact_gt_arr, contact_pred_arr, contact_gt_arr)
    conf_mat = compute_confusion_mat(tasks_dict,bin_contact_pred_arr,bin_contact_gt_arr,contact_pred_arr,contact_gt_arr,terrain_pred_arr,terrain_gt_arr)

    print("Test accuracy (contact) is: %.4f" % test_acc['contact'])
    print("Test accuracy (terrain) is: %.4f" % test_acc['terrain'])
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
    print("Terrain confusion matrix: ")
    # print(conf_mat['terrain'])
    np.savetxt(sys.stdout.buffer, conf_mat['terrain'])

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
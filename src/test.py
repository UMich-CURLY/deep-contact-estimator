import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm

import torch.optim as optim

from contact_cnn import *
from utils.data_handler import *

def compute_accuracy(dataloader, model):
    # compute accuracy in batch

    num_correct = 0
    num_data = 0
    correct_per_leg = np.zeros(4)
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_label = sample['label']

            output = model(input_data)

            normalized_output = (output-torch.min(output))/(torch.max(output)-torch.min(output))
            normalized_output = normalized_output/normalized_output.sum()

            _, prediction = torch.max(normalized_output,1)
            top2_val, top2_idx = torch.topk(normalized_output,2,dim=1)
            top2_ratio = top2_val[0,1]/top2_val[0,0]

            

            bin_pred = decimal2binary(prediction)
            bin_2ndbest = decimal2binary(top2_idx[0,1])
            bin_gt = decimal2binary(gt_label)
            
            
            if top2_ratio > 0.92:
                # print("----------------")
                # print(bin_pred[0])
                # print(bin_2ndbest)
                new_bin_pred = torch.logical_and(bin_pred,bin_2ndbest).type(torch.uint8)
                bin_pred = new_bin_pred

            correct_per_leg += (bin_pred==bin_gt).sum(axis=0).cpu().numpy()
            num_data += input_data.size(0)
            num_correct += (prediction==gt_label).sum().item()


    return num_correct/num_data, correct_per_leg/num_data

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

    test_acc, acc_per_leg = compute_accuracy(test_dataloader, model)

    print("Test accuracy in terms of class is: %.4f" % test_acc)
    print("Accuracy of leg 0 is: %.4f" % acc_per_leg[0])
    print("Accuracy of leg 1 is: %.4f" % acc_per_leg[1])
    print("Accuracy of leg 2 is: %.4f" % acc_per_leg[2])
    print("Accuracy of leg 3 is: %.4f" % acc_per_leg[3])
    print("Accuracy is: %.4f" % (np.sum(acc_per_leg)/4.0))

    print(test_acc)
    print(acc_per_leg[0])
    print(acc_per_leg[1])
    print(acc_per_leg[2])
    print(acc_per_leg[3])
    print((np.sum(acc_per_leg)/4.0))

if __name__ == '__main__':
    main()
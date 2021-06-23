import os
import argparse
import glob
import sys
sys.path.append('..')
import yaml
from tqdm import tqdm

import torch.optim as optim

from contact_cnn import *
from utils.data_handler import *

def compute_accuracy(dataloader, model):

    num_correct = 0
    num_data = 0
    with torch.no_grad():
        for sample in tqdm(dataloader): # tqdm shows a smart progress meter
            input_data = sample['data']
            gt_label = sample['label']

            output = model(input_data)
            _, prediction = torch.max(output,1)
            # print("gt_label: ", gt_label)
            # print("prediction: " , prediction)

            num_data += input_data.size(0)
            num_correct += (prediction==gt_label).sum().item()

    return num_correct/num_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    parser = argparse.ArgumentParser(description='Test the contcat network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/test_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name))

    test_data = contact_dataset(data_path=config['data_folder']+"test.npy",\
                                label_path=config['data_folder']+"test_label.npy",\
                                window_size=config['window_size'], device=device)
    test_dataloader = DataLoader(dataset=test_data, batch_size=config['batch_size'])
    # test_dataloader = DataLoader(dataset=test_data, batch_size=1)



    # init network
    model = contact_cnn()
    checkpoint = torch.load(config['model_load_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    test_acc = compute_accuracy(test_dataloader, model)

    print("Test accuracy is: %.4f" % test_acc)


    

if __name__ == '__main__':
    main()
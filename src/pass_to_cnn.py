import os
import argparse
import glob
import sys
sys.path.append('..')
import yaml
from tqdm import tqdm

import torch.optim as optim

from src.contact_cnn import *
# from utils.data_handler_np import *


def compute_prediction(dataloader, model, gt_label):
    with torch.no_grad():
        output = model(dataloader)
        _, prediction = torch.max(output, 1)
        print("gt_label: ", gt_label)
        print("Prediction: ", prediction)
        return prediction


def receive_input(feature_matrix, input_rows, input_cols, label_matrix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('Using ', device)

    parser = argparse.ArgumentParser(description='Test the contcat network')
    parser.add_argument('--config_name', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)) + '/../config/test_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name))
    test_data = torch.from_numpy(feature_matrix).type('torch.FloatTensor').to(device)
    label_data = torch.from_numpy(label_matrix).type('torch.FloatTensor').to(device)
    gt_label = label_data[149]

    # test_data = contact_dataset(feature_matrix, device=device)
    test_dataloader = test_data.view(-1, input_rows, input_cols)

    # init network
    model = contact_cnn()
    checkpoint = torch.load(config['model_load_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    test_predict = compute_prediction(test_dataloader, model, gt_label)

    # print("Test prediction is: %.4f" % test_predict)
    return test_predict

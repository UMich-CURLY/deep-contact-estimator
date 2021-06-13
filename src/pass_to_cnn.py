import os
import argparse
import glob
import sys
sys.path.append('..')
import yaml
from tqdm import tqdm

import torch.optim as optim

from src.contact_cnn import *
from utils.data_handler_np import *


def compute_prediction(dataloader, model):
    num_correct = 0
    num_data = 0
    with torch.no_grad():
        for input_data in dataloader:  # tqdm shows a smart progress meter, but here we don't need it

            output = model(input_data)
            _, prediction = torch.max(output, 1)
            print("Prediction: ", prediction)
            return prediction


def receive_input(feature_matrix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    parser = argparse.ArgumentParser(description='Test the contcat network')
    parser.add_argument('--config_name', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)) + '/../config/test_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name))

    test_data = contact_dataset(feature_matrix, device=device)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1)

    # init network
    model = contact_cnn()
    checkpoint = torch.load(config['model_load_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    test_predict = compute_prediction(test_dataloader, model)

    print("Test prediction is: %.4f" % test_predict)
    return test_predict

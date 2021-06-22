import os
import argparse
import glob
import sys
sys.path.append('..')
import yaml
import torch.optim as optim
from src.contact_cnn import *


def compute_prediction(dataloader, model, gt_label):
    with torch.no_grad():
        output = model(dataloader)
        _, prediction = torch.max(output, 1)
        print("gt_label: ", gt_label)
        print("Prediction: ", prediction)
        return prediction


def receive_input(feature_matrix, input_rows, input_cols, label_matrix, model):
    # print('Using ', device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = torch.from_numpy(feature_matrix).type('torch.FloatTensor').to(device)
    label_data = torch.from_numpy(label_matrix).type('torch.FloatTensor').to(device)
    gt_label = label_data[-1]

    test_data = (test_data-torch.mean(test_data,dim=0))\
                            /torch.std(test_data,dim=0)
    test_dataloader = test_data.view(-1, input_rows, input_cols)
    

    # prediction takes about 0.0012 s
    test_predict = compute_prediction(test_dataloader, model, gt_label)

    # print("Test prediction is: %.4f" % test_predict)
    return test_predict

import sys
sys.path.append("..")
import torchvision
import torch
from torch.autograd import Variable
import onnx
import argparse
from src.contact_cnn import *
from utils.data_handler import *
import os
import yaml
print(torch.__version__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Test the contcat network')
parser.add_argument('--config_name', type=str,
                    default=os.path.dirname(os.path.abspath(__file__)) + '/../config/test_params.yaml')
args = parser.parse_args()
config = yaml.load(open(args.config_name))

model = contact_cnn()
checkpoint = torch.load(config['model_load_path'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.eval().to(device)


test_data = contact_dataset(data_path=config['data_folder']+"test_lcm.npy",\
                            label_path=config['data_folder']+"test_label_lcm.npy",\
                            window_size=config['window_size'], device=device)
test_dataloader = DataLoader(dataset=test_data, batch_size=1)
for i in test_dataloader:
    print(i['data'].shape)
    input = i['data']
    output = model(input)

ONNX_FILE_PATH = '/home/tingjun/Desktop/Cheetah_code/deep-contact-estimator/results/0412_1dcnn_64_128_no_tao_GRF.onnx'
torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=['input'], output_names=['output'], export_params=True)

onnx_model = onnx.load(ONNX_FILE_PATH)
onnx.checker.check_model(onnx_model)
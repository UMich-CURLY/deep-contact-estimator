import sys
sys.path.append("..")
import torchvision
import torch
from torch.autograd import Variable
import onnx
import argparse
from src.contact_cnn import *
import torch
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


input_name = ['input']
output_name = ['output']
input = Variable(torch.randn(1, 3, 224, 224)).cuda()
# model = torchvision.models.resnet50(pretrained=True).cuda()
torch.onnx.export(model, input, 'resnet50.onnx', input_names=input_name, output_names=output_name, verbose=True)
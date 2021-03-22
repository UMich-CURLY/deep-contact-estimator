import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm
import scipy.io as sio

import torch.optim as optim

from contact_cnn import *
from utils.data_handler import *

def inference(dataloader, model, device):

    num_correct = 0
    num_data = 0
    infer_results = torch.empty(0,4,dtype=torch.uint8).to(device)
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_label = sample['label']

            output = model(input_data)
            _, prediction = torch.max(output,1)

            bin_pred = decimal2binary(prediction)
            infer_results = torch.cat((infer_results, bin_pred), 0)

            # print(prediction)
            # print(bin_pred)

            num_data += input_data.size(0)
            num_correct += (prediction==gt_label).sum().item()
            
    return infer_results, num_correct/num_data


def decimal2binary(x):
    mask = 2**torch.arange(4-1,-1,-1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def save2mat(pred, config):
    data = np.load(config['data_path'])
    label_deci_np = np.load(config['label_path'])
    label_deci = torch.from_numpy(label_deci_np)
    label = decimal2binary(label_deci).reshape(-1,4)

    print(type(label))
    print(label.shape)
    print(label[149])
    print(type(pred))
    print(pred.shape)
    print(pred[0])

    out = {}
    out['contacts'] = pred.cpu().numpy()
    out['ground_truth'] = label[config['window_size']-1:,:].numpy()
    out['q'] = data[config['window_size']-1:,:12]
    out['qd'] = data[config['window_size']-1:,12:24]
    out['imu_acc'] = data[config['window_size']-1:,24:27]
    out['imu_omega'] = data[config['window_size']-1:,27:30]
    out['p'] = data[config['window_size']-1:,30:42]
    out['v'] = data[config['window_size']-1:,42:54]
    out['tau_est'] = data[config['window_size']-1:,54:66]
    
    sio.savemat(config['mat_save_path'],out)




def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    parser = argparse.ArgumentParser(description='Test the contcat network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/inference_one_seq_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name))
    
    dataset = contact_dataset(data_path=config['data_path'],\
                                label_path=config['label_path'],\
                                window_size=config['window_size'],device=device)
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'])

    model = contact_cnn()
    checkpoint = torch.load(config['model_load_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    pred, acc = inference(dataloader, model, device)

    print("accuracy is: %.4f"%acc)

    if(config['save_mat']):
        save2mat(pred,config)

    if(config['save_lcm']):
        pass

if __name__ == '__main__':
    main()
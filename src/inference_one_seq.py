import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm
import scipy.io as sio

import lcm
from lcm_types.python import contact_t, leg_control_data_lcmt, microstrain_lcmt
import time

import torch.optim as optim

from contact_cnn import *
from utils.data_handler import *

def inference(dataloader, model, device):
    
    infer_results = torch.empty(0,4,dtype=torch.uint8).to(device)
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            output = model(input_data)
            _, prediction = torch.max(output,1)
            bin_pred = decimal2binary(prediction)
            infer_results = torch.cat((infer_results, bin_pred), 0)

    return infer_results


def inference_and_compute_acc(dataloader, model, device):

    num_correct = 0
    num_data = 0
    correct_per_leg = np.zeros(4)
    infer_results = torch.empty(0,4,dtype=torch.uint8).to(device)
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_label = sample['label']


            output = model(input_data)
            _, prediction = torch.max(output,1)
            bin_pred = decimal2binary(prediction)
            bin_gt = decimal2binary(gt_label).view(-1,4)
            infer_results = torch.cat((infer_results, bin_pred), 0)


            correct_per_leg += (bin_pred==bin_gt).sum(axis=0).cpu().numpy()
            num_data += input_data.size(0)
            num_correct += (prediction==gt_label).sum().item()

    # return infer_results
    return infer_results, num_correct/num_data,  correct_per_leg/num_data

def decimal2binary(x):
    mask = 2**torch.arange(4-1,-1,-1).to(x.device, x.dtype)

    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def save2mat(pred, config):
    mat_raw_data = sio.loadmat(config['mat_data_path'])
    data = np.load(config['data_path'])
    label_deci_np = np.load(config['label_path'])
    label_deci = torch.from_numpy(label_deci_np)
    label = decimal2binary(label_deci).reshape(-1,4)

    out = {}
    out['contacts_est'] = pred.cpu().numpy()
    out['contacts_gt'] = label[config['window_size']-1:,:].numpy()
    out['q'] = data[config['window_size']-1:,:12]
    out['qd'] = data[config['window_size']-1:,12:24]
    out['imu_acc'] = data[config['window_size']-1:,24:27]
    out['imu_omega'] = data[config['window_size']-1:,27:30]
    out['p'] = data[config['window_size']-1:,30:42]
    out['v'] = data[config['window_size']-1:,42:54]

    # data not used in the network but needed for visualization.
    out['control_time'] = mat_raw_data['control_time'].flatten().tolist()[config['window_size']-1:]
    out['imu_time'] = mat_raw_data['imu_time'].flatten().tolist()[config['window_size']-1:]
    out['tau_est'] = mat_raw_data['tau_est'][config['window_size']-1:]
    out['F'] = mat_raw_data['F'][config['window_size']-1:]

    sio.savemat(config['mat_save_path'],out)

    print("Saved data to mat!")

def save2lcm(pred, config):
    mat_data = sio.loadmat(config['mat_data_path'])
    log = lcm.EventLog(config['lcm_save_path'], mode='w', overwrite=True)
    
    utime = int(time.time() * 10**6)

    imu_time = mat_data['imu_time'].flatten().tolist()


    
    for idx,_ in enumerate(imu_time[config['window_size']-1:]):

        data_idx = idx + config['window_size']-1
        
        leg_control_data_msg = leg_control_data_lcmt()
        leg_control_data_msg.q = mat_data['q'][data_idx]
        leg_control_data_msg.p = mat_data['p'][data_idx]
        leg_control_data_msg.qd = mat_data['qd'][data_idx]
        leg_control_data_msg.v = mat_data['v'][data_idx]
        leg_control_data_msg.tau_est = mat_data['tau_est'][data_idx]
        log.write_event(utime + int(10**6 * imu_time[data_idx]),\
                    'leg_control_data', leg_control_data_msg.encode())
        
        contact_msg = contact_t()
        contact_msg.num_legs = 4
        contact_msg.timestamp = imu_time[data_idx]
        contact_msg.contact = pred[idx]

        # if we want to use GT contact for verification
        # contact_msg.contact = mat_data['contacts'][data_idx]
        
        log.write_event(utime + int(10**6 * imu_time[data_idx]),\
                        'contact', contact_msg.encode())
        
        imu_msg = microstrain_lcmt()
        imu_msg.acc = mat_data['imu_acc'][data_idx]
        imu_msg.omega = mat_data['imu_omega'][data_idx]
        imu_msg.rpy = mat_data['imu_rpy'][data_idx]
        imu_msg.quat = mat_data['imu_quat'][data_idx]
        log.write_event(utime + int(10**6 * imu_time[data_idx]),\
                        'microstrain', imu_msg.encode())
        
    print("Saved data to lcm!")



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

    pred = []
    if(config['calculate_accuracy']):
        pred, acc, acc_per_leg = inference_and_compute_acc(dataloader, model, device)
        print("Accuracy in terms of class: %.4f" % acc)
        print("Accuracy of leg 0 is: %.4f" % acc_per_leg[0])
        print("Accuracy of leg 1 is: %.4f" % acc_per_leg[1])
        print("Accuracy of leg 2 is: %.4f" % acc_per_leg[2])
        print("Accuracy of leg 3 is: %.4f" % acc_per_leg[3])
        print("Accuracy is: %.4f" % (np.sum(acc_per_leg)/4.0))
    else:
        pred = inference(dataloader, model, device)

    

    if(config['save_mat']):
        save2mat(pred,config)

    if(config['save_lcm']):
        save2lcm(pred,config)

if __name__ == '__main__':
    main()
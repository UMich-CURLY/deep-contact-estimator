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

    num_correct = 0
    num_data = 0
    for sample in tqdm(dataloader):
        input_data = sample['data']
        gt_label = sample['label']

        output = model(input_data)
        _, prediction = torch.max(output,1)

        num_data += input_data.size(0)
        num_correct += (prediction==gt_label).sum().item()

    return num_correct/num_data

def train(model, train_dataloader, val_dataloader, config):

    # let's try fixed lr first
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])

    loss_history = []
    for epoch in range(config['num_epoch']):
        cur_loss_history = []
        for i, samples in tqdm(enumerate(train_dataloader, start=0)):
            input_data = samples['data'] 
            label = samples['label'] 

            optimizer.zero_grad()
            output = model(input_data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            cur_loss_history.append(loss.item())

            if i % config['print_every'] == 0:
                print("epoch %d / %d, iteration %d / %d, loss: %.8f" %\
                    (epoch, config['num_epoch'], i, len(train_dataloader), loss))
        
        loss_history.append(cur_loss_history)

        train_acc = compute_accuracy(train_dataloader, model)
        val_acc = compute_accuracy(val_dataloader, model)

        print("Finished epoch %d / %d, training acc: %.4f, validation acc: %.4f" %\
            (epoch, config['num_epoch'], train_acc, val_acc))  
    
    # save model     
    torch.save(model, config['model_save_path'])
    
    # log loss historys
    loss_history = np.array(loss_history) 
    np.save(config['loss_history_path'],loss_history)       

def main():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)


    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/network_params.yaml')
    parser.add_argument('--mode', type=str, help='train: training mode. test: testing mode.')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name))


    if args.mode is None:
        print('Please provide mode. --mode train or --mode test')
        sys.exit(2)


    # load data   
    data = load_data_from_mat(config['data_folder'],0.7,0.15)
    
    # separate data
    train_data = contact_dataset(data=data['train'],label=data['train_label'],device=device)
    train_dataloader = DataLoader(dataset=train_data, batch_size=config['batch_size'])
    val_data = contact_dataset(data=data['val'],label=data['val_label'],device=device)
    val_dataloader = DataLoader(dataset=val_data, batch_size=config['batch_size'])
    test_data = contact_dataset(data=data['test'],label=data['test_label'],device=device)
    test_dataloader = DataLoader(dataset=test_data, batch_size=config['batch_size'])

    # init network
    model = contact_net()
    model = model.to(device)

    if args.mode == 'train':
        train(model, train_dataloader, val_dataloader, config)
    elif args.mode == 'test':
        pass
    else:
        print('Your requested mode does not exist. :o')
        sys.exit(2)

if __name__ == '__main__':
    main()
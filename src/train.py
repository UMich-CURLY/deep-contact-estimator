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

def compute_accuracy_and_loss(dataloader, model, criterion):

    num_correct = 0
    num_data = 0
    loss_history = []
    for sample in tqdm(dataloader):
        input_data = sample['data']
        gt_label = sample['label']

        output = model(input_data)
        _, prediction = torch.max(output,1)

        loss = criterion(output, gt_label)

        num_data += input_data.size(0)
        num_correct += (prediction==gt_label).sum().item()

        loss_history.append(loss.item())

    return num_correct/num_data, loss_history

def train(model, train_dataloader, val_dataloader, config):

    # let's try fixed lr first
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])

    history = {}
    val_loss_history = []
    training_loss_history = []
    train_acc_history = []
    val_acc_history = []
    for epoch in range(config['num_epoch']):
        cur_training_loss_history = []
        for i, samples in tqdm(enumerate(train_dataloader, start=0)):
            input_data = samples['data'] 
            label = samples['label'] 

            optimizer.zero_grad()
            output = model(input_data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            cur_training_loss_history.append(loss.item())

            if i % config['print_every'] == 0:
                print("epoch %d / %d, iteration %d / %d, loss: %.8f" %\
                    (epoch, config['num_epoch'], i, len(train_dataloader), loss))
        
        train_acc = compute_accuracy(train_dataloader, model)
        val_acc, cur_val_loss_history = compute_accuracy_and_loss(val_dataloader, model, criterion)

        training_loss_history.append(cur_training_loss_history)
        val_loss_history.append(cur_val_loss_history)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)    

        print("Finished epoch %d / %d, training acc: %.4f, validation acc: %.4f" %\
            (epoch, config['num_epoch'], train_acc, val_acc))  
    
    # save model     
    torch.save(model, config['model_save_path'])
    
    # log loss historys
    history = {"train_acc":train_acc_history,"val_acc":val_acc_history,\
                "train_loss_history":training_loss_history,"val_loss_history":val_loss_history}
    np.save(config['loss_history_path'],history)       

def main():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)


    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/network_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name))

    # load data   
    # data = load_data_from_mat(config['data_folder'],config['window_size'],0.7,0.15)
    
    # separate data
    train_data = contact_dataset(data_path=config['data_folder']+"train.npy",\
                                label_path=config['data_folder']+"train_label.npy",\
                                window_size=config['window_size'],device=device)
    train_dataloader = DataLoader(dataset=train_data, batch_size=config['batch_size'])
    val_data = contact_dataset(data_path=config['data_folder']+"val.npy",\
                                label_path=config['data_folder']+"val_label.npy",\
                                window_size=config['window_size'],device=device)
    val_dataloader = DataLoader(dataset=val_data, batch_size=config['batch_size'])
    # test_data = contact_dataset(data_path=config['data_folder']+"test.npy",\
    #                             label_path=config['data_folder']+"test_label.npy",\
    #                             window_size=config['window_size'],device=device)
    # test_dataloader = DataLoader(dataset=test_data, batch_size=config['batch_size'])

    # init network
    model = contact_cnn()
    model = model.to(device)

    # if args.mode == 'train':

    train(model, train_dataloader, val_dataloader, config)

    # elif args.mode == 'test':
    #     pass
    # else:
    #     print('Your requested mode does not exist. :o')
    #     sys.exit(2)

if __name__ == '__main__':
    main()
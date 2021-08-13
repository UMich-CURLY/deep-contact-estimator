import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm

import torch.optim as optim

from mtl_models import MultiTaskModel
from loss import MultiTaskLoss
from contact_cnn import *
from utils.data_handler import *
from sklearn.metrics import confusion_matrix

from torch.utils.tensorboard import SummaryWriter
import wandb

def compute_accuracy(dataloader, model):

    num_correct_contact = 0
    num_correct_terrain = 0
    num_data = 0
    correct_per_leg = np.zeros(4)
    prediction = {}
    for sample in tqdm(dataloader):
        input_data = sample['data']
        labels = {
              "contact" : sample['contact_label'] ,
              "terrain" : sample['terrain_label'] 
            }

        output = model(input_data)
        _, prediction['contact'] = torch.max(output['contact'],1)
        _, prediction['terrain'] = torch.max(output['terrain'],1)

        bin_contact_pred = decimal2binary(prediction['contact'])
        bin_contact_gt = decimal2binary(labels['contact'])

        correct_per_leg += (bin_contact_pred==bin_contact_gt).sum(axis=0).cpu().numpy()
        num_data += input_data.size(0)
        num_correct_contact += (prediction['contact']==labels['contact']).sum().item()
        num_correct_terrain += (prediction['terrain']==labels['terrain']).sum().item()

    return num_correct_contact/num_data, correct_per_leg/num_data, num_correct_terrain/num_data

def compute_accuracy_and_loss(model, tasks_dict, dataloader, criterion):

    num_correct_contact = 0
    num_correct_terrain = 0
    num_data = 0
    loss_sum = {}
    loss_sum = {task: 0.0 for task, _ in tasks_dict.items()}
    loss_sum['total'] = 0.0
    correct_per_leg = np.zeros(4)
    prediction = {}
    confusion_mat = {'leg_rf': np.zeros((2,2)),'leg_lf': np.zeros((2,2)),'leg_rh': np.zeros((2,2)),'leg_lh': np.zeros((2,2)),\
                    'terrain': np.zeros((tasks_dict['terrain'],tasks_dict['terrain']))}
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            labels = {
              "contact" : sample['contact_label'] ,
              "terrain" : sample['terrain_label'] 
            }

            output = model(input_data)
            _, prediction['contact'] = torch.max(output['contact'],1)
            _, prediction['terrain'] = torch.max(output['terrain'],1)

            loss = criterion(output, labels)

            bin_pred = decimal2binary(prediction['contact'])
            bin_gt = decimal2binary(labels['contact'])

            correct_per_leg += (bin_pred==bin_gt).sum(axis=0).cpu().numpy()
            num_data += input_data.size(0)

            num_correct_contact += (prediction['contact']==labels['contact']).sum().item()
            num_correct_terrain += (prediction['terrain']==labels['terrain']).sum().item()
            
            bin_pred_cpu = bin_pred.cpu()
            bin_gt_cpu = bin_gt.cpu()
            ter_labels_cpu = labels['terrain'].cpu().numpy()
            ter_pred_cpu = prediction['terrain'].cpu().numpy()

            # print('labels')
            # print(np.shape(ter_labels_cpu))
            # print('pred')
            # print(np.shape(ter_pred_cpu))
            confusion_mat['leg_rf'] += confusion_matrix(bin_gt_cpu[:,0],bin_pred_cpu[:,0], labels=[0,1])
            confusion_mat['leg_lf'] += confusion_matrix(bin_gt_cpu[:,1],bin_pred_cpu[:,1], labels=[0,1])
            confusion_mat['leg_rh'] += confusion_matrix(bin_gt_cpu[:,2],bin_pred_cpu[:,2], labels=[0,1])
            confusion_mat['leg_lh'] += confusion_matrix(bin_gt_cpu[:,3],bin_pred_cpu[:,3], labels=[0,1])

            # print('conf_mat')
            # print(np.shape(confusion_matrix(ter_labels_cpu,ter_pred_cpu)))
            # print(confusion_mat['terrain'])
            confusion_mat['terrain'] += confusion_matrix(ter_labels_cpu,ter_pred_cpu, labels=[0,1,2,3,4,5,6,7,8])

            loss_sum = {loss_type: cur_loss_sum+loss[loss_type].item() for loss_type, cur_loss_sum in loss_sum.items()}

    return num_correct_contact/num_data, correct_per_leg/num_data, num_correct_terrain/num_data, {loss_type: cur_loss_sum/len(dataloader) for loss_type, cur_loss_sum in loss_sum.items()}, confusion_mat

def decimal2binary(x):
    mask = 2**torch.arange(4-1,-1,-1).to(x.device, x.dtype)

    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def train(model, tasks_dict, train_dataloader, val_dataloader, config):
  
    writer = SummaryWriter(config['log_writer_path'],comment=config['model_description'])
    writer.add_text("data_folder: ",config['data_folder'])
    writer.add_text("model_save_path: ",config['model_save_path'])
    writer.add_text("log_writer_path: ",config['log_writer_path'])
    writer.add_text("window_size: ",str(config['window_size']))
    writer.add_text("shuffle: ",str(config['shuffle']))
    writer.add_text("batch_size: ",str(config['batch_size']))
    writer.add_text("init_lr: ",str(config['init_lr']))
    writer.add_text("num_epoch: ",str(config['num_epoch']))

    wandb.init(project=config['wandb_project_name'],name=config['wandb_run_name'],\
                                                    config={"window_size": config['window_size'],\
                                                            "shuffle": config['shuffle'],\
                                                            "batch_size": config['batch_size'],\
                                                            "init_lr": config['init_lr'],\
                                                            "num_epoch": config['num_epoch'],\
                                                            "dropout": config['dropout_rate']})
    
    
    criterion = MultiTaskLoss(tasks_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['init_lr'])

    model.train()

    best_acc = {}
    best_loss = {}
    best_acc['contact'] = 0.0
    best_acc['leg'] = 0.0
    best_acc['terrain'] = 0.0
    best_loss = {task: 1000000000 for task, _ in tasks_dict.items()}
    best_loss['total'] = 1000000000
    for epoch in range(config['num_epoch']):
        running_loss = {}
        running_loss = {task: 0.0 for task, _ in tasks_dict.items()}
        running_loss['total'] = 0.0
        loss_sum = {}
        loss_sum = {task: 0.0 for task, _ in tasks_dict.items()}
        loss_sum['total'] = 0.0
        for i, samples in tqdm(enumerate(train_dataloader, start=0)):
            input_data = samples['data'] 
            labels = {
              "contact" : samples['contact_label'],
              "terrain" : samples['terrain_label'] 
            }

            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, labels)
            loss['total'].backward()
            optimizer.step()


            running_loss = {loss_type: cur_running_loss+loss[loss_type].item() for loss_type, cur_running_loss in running_loss.items()}
            loss_sum = {loss_type: cur_loss_sum+loss[loss_type].item() for loss_type, cur_loss_sum in loss_sum.items()}

            if i % config['print_every'] == 0:
                print("epoch %d / %d, iteration %d / %d, contact loss: %.8f, terrain loss: %.8f, total loss: %.8f," %\
                    (epoch, config['num_epoch'], i, len(train_dataloader), running_loss['contact']/config['print_every']\
                      , running_loss['terrain']/config['print_every'], running_loss['total']/config['print_every']))
                running_loss = {loss_type: 0.0 for loss_type, _ in running_loss.items()}

        # calculate training and validation accuracy
        model.eval()
        train_acc = {}
        train_acc['contact'], train_acc_per_leg, train_acc['terrain'] = compute_accuracy(train_dataloader, model)
        train_loss_avg = {}
        train_loss_avg = {loss_type: cur_loss_sum/len(train_dataloader) for loss_type, cur_loss_sum in loss_sum.items()}

        val_acc = {}
        val_acc['contact'], val_acc_per_leg, val_acc['terrain'], val_loss_avg, val_conf_mat\
            = compute_accuracy_and_loss(model, tasks_dict, val_dataloader, criterion)


        terrain_table_col = ['0: Air', '1: Asphalt Road', '2: Concrete', '3: Forest', '4: Grass',\
                        '5: Middle Pebbles', '6: Small Pebbles', '7: Rock Road', '8: Sidewalk']
        conf_terrain_table = wandb.Table(columns=terrain_table_col, data=val_conf_mat['terrain'].tolist())
        conf_rf_table = wandb.Table(columns=[0, 1],data=val_conf_mat['leg_rf'].tolist())
        conf_lf_table = wandb.Table(columns=[0, 1],data=val_conf_mat['leg_lf'].tolist())
        conf_rh_table = wandb.Table(columns=[0, 1],data=val_conf_mat['leg_rh'].tolist())
        conf_lh_table = wandb.Table(columns=[0, 1],data=val_conf_mat['leg_lh'].tolist())

        train_acc_per_leg_avg = (np.sum(train_acc_per_leg)/4.0)
        val_acc_per_leg_avg = (np.sum(val_acc_per_leg)/4.0)

        # log down info in tensorboard
        writer.add_scalar('training loss (contact)', train_loss_avg['contact'], epoch)
        writer.add_scalar('training loss (terain)', train_loss_avg['terrain'], epoch)
        writer.add_scalar('training loss (total)', train_loss_avg['total'], epoch)
        writer.add_scalar('training accuracy (contact)', train_acc['contact'], epoch)
        writer.add_scalar('training accuracy (terrain)', train_acc['terrain'], epoch)
        writer.add_scalar('training acc leg0', train_acc_per_leg[0], epoch)
        writer.add_scalar('training acc leg1', train_acc_per_leg[1], epoch)
        writer.add_scalar('training acc leg2', train_acc_per_leg[2], epoch)
        writer.add_scalar('training acc leg3', train_acc_per_leg[3], epoch)
        writer.add_scalar('training acc leg avg', train_acc_per_leg_avg, epoch)
        writer.add_scalar('validation loss (contact)', val_loss_avg['contact'], epoch)
        writer.add_scalar('validation loss (terrain)', val_loss_avg['terrain'], epoch)
        writer.add_scalar('validation loss (total)', val_loss_avg['total'], epoch)
        writer.add_scalar('validation accuracy (contact)', val_acc['contact'], epoch)
        writer.add_scalar('validation accuracy (terrain)', val_acc['terrain'], epoch)
        writer.add_scalar('validation acc leg0', val_acc_per_leg[0], epoch)
        writer.add_scalar('validation acc leg1', val_acc_per_leg[1], epoch)
        writer.add_scalar('validation acc leg2', val_acc_per_leg[2], epoch)
        writer.add_scalar('validation acc leg3', val_acc_per_leg[3], epoch)
        writer.add_scalar('validation acc leg avg', val_acc_per_leg_avg, epoch)

        wandb.log({'training loss (contact)': train_loss_avg['contact'],
                   'training loss (terrain)': train_loss_avg['terrain'],
                   'training loss (total)': train_loss_avg['total'],
                   'training accuracy (contact)': train_acc['contact'],
                   'training accuracy (terrain)': train_acc['terrain'],
                   'training acc leg0': train_acc_per_leg[0],
                   'training acc leg1': train_acc_per_leg[1],
                   'training acc leg2': train_acc_per_leg[2],
                   'training acc leg3': train_acc_per_leg[3],
                   'training acc leg avg': train_acc_per_leg_avg,
                   'validation loss (contact)': val_loss_avg['contact'],
                   'validation loss (terrain)': val_loss_avg['terrain'],
                   'validation loss (total)': val_loss_avg['total'],
                   'validation accuracy (contact)': val_acc['contact'],
                   'validation accuracy (terrain)': val_acc['terrain'],
                   'validation acc leg0': val_acc_per_leg[0],
                   'validation acc leg1': val_acc_per_leg[1],
                   'validation acc leg2': val_acc_per_leg[2],
                   'validation acc leg3': val_acc_per_leg[3],
                   'validation acc leg avg': val_acc_per_leg_avg,
                   'epoch': epoch,
                   'terrain confusion matrix': conf_terrain_table,
                   'leg rf confusion matrix': conf_rf_table,
                   'leg lf confusion matrix': conf_lf_table,
                   'leg rh confusion matrix': conf_rh_table,
                   'leg lh confusion matrix': conf_lh_table
                    })

        # if we achieve best val contact acc, save the model.
        if val_acc['contact'] > best_acc['contact']:
            best_acc['contact'] = val_acc['contact']
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            torch.save(state, config['model_save_path']+'_best_val_contact_acc.pt')

        if val_acc['terrain'] > best_acc['terrain']:
            best_acc['terrain'] = val_acc['terrain']
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            torch.save(state, config['model_save_path']+'_best_val_terrain_acc.pt')
        
        # if we achieve best val acc, save the model.
        if val_acc_per_leg_avg > best_acc['leg']:
            best_acc['leg'] = val_acc_per_leg_avg
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            torch.save(state, config['model_save_path']+'_best_val_leg_acc.pt')

        # if we achieve best val loss, save the model
        if val_loss_avg['contact'] < best_loss['contact']:
            best_loss['contact'] = val_loss_avg['contact']
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            torch.save(state, config['model_save_path']+'_best_val_contact_loss.pt')

        # if we achieve best val loss, save the model
        if val_loss_avg['terrain'] < best_loss['terrain']:
            best_loss['terrain'] = val_loss_avg['terrain']
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            torch.save(state, config['model_save_path']+'_best_val_terrain_loss.pt')

        # if we achieve best val loss, save the model
        if val_loss_avg['total'] < best_loss['total']:
            best_loss['total'] = val_loss_avg['total']
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            torch.save(state, config['model_save_path']+'_best_val_total_loss.pt')


        print("Finished epoch %d / %d, training contact acc: %.4f, validation contact acc: %.4f" %\
            (epoch, config['num_epoch'], train_acc['contact'], val_acc['contact'])) 
        print("training terrain acc: %.4f, validation terrain acc: %.4f" %\
            (train_acc['terrain'], val_acc['terrain']))
        print("train leg0 acc: %.4f, train leg1 acc: %.4f, train leg2 acc: %.4f, train leg3 acc: %.4f, train leg acc avg: %.4f" %\
            (train_acc_per_leg[0],train_acc_per_leg[1],train_acc_per_leg[2],train_acc_per_leg[3],train_acc_per_leg_avg))    
        print("val leg0 acc: %.4f, val leg1 acc: %.4f, val leg2 acc: %.4f, val leg3 acc: %.4f, val leg acc avg: %.4f" %\
            (val_acc_per_leg[0],val_acc_per_leg[1],val_acc_per_leg[2],val_acc_per_leg[3],val_acc_per_leg_avg))
        print("===================================================================================================================")
    
    # save model     
    state = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss_avg,
            'acc': train_acc,
            'val_loss': val_loss_avg,
            'val_acc': val_acc}

    torch.save(state, config['model_save_path']+'_final_epo.pt')
    
    writer.close()
    wandb.finish

def main():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)


    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/network_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name))

    print("Using the following params: ")
    print("-------------path-------------")
    print("data_folder: ",config['data_folder'])
    print("model_save_path: ",config['model_save_path'])
    print("log_writer_path: ",config['log_writer_path'])
    # print("loss_history_path: ",config['loss_history_path'])
    print("--------network params--------")
    print("window_size: ",config['window_size'])
    print("shuffle: ",config['shuffle'])
    print("batch_size: ",config['batch_size'])
    print("init_lr: ",config['init_lr'])
    print("num_epoch: ",config['num_epoch'])


    # load data   
    # data = load_data_from_mat(config['data_folder'],config['window_size'],0.7,0.15)
    
    # separate data
    train_data = contact_dataset(data_path=config['data_folder']+"train.npy",\
                                window_size=config['window_size'],device=device)
    train_dataloader = DataLoader(dataset=train_data, batch_size=config['batch_size'],\
                                  shuffle=config['shuffle'])
    val_data = contact_dataset(data_path=config['data_folder']+"val.npy",\
                                window_size=config['window_size'],device=device)
    val_dataloader = DataLoader(dataset=val_data, batch_size=config['batch_size'])
    # test_data = contact_dataset(data_path=config['data_folder']+"test.npy",\
    #                             label_path=config['data_folder']+"test_label.npy",\
    #                             window_size=config['window_size'],device=device)
    # test_dataloader = DataLoader(dataset=test_data, batch_size=config['batch_size'])
    
    # init network
    # Initialize multi-task models and losses
    tasks_dict = {
      "contact" : 16,
      "terrain" : 9
    }
    model = MultiTaskModel(tasks_dict, config).to(device)

    train(model, tasks_dict, train_dataloader, val_dataloader, config)

   

if __name__ == '__main__':
    main()

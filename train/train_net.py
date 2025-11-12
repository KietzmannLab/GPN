# Code to train the GPNs

import argparse
parser = argparse.ArgumentParser(description='Obtaining hyps')

parser.add_argument('--network_id', type=int, default=1) # unique id for network
parser.add_argument('--gaze_type', type=str, default='dg3') # dg3/random/dg3p/dg3r
parser.add_argument('--recurrence', type=int, default=1) # 0 if no recurrence, 1 if recurrence
parser.add_argument('--provide_loc', type=int, default=0) # 1 if saccade is input, 0 if not

parser.add_argument('--network_type', type=str, default='lstm') # lstm
parser.add_argument('--timesteps', type=int, default=6) # how many gazes to be provided as input to lstm
parser.add_argument('--timestep_multiplier', type=int, default=3) # how many rnn layers
parser.add_argument('--n_rnn', type=int, default=1024) # number of neurons in each rnn layer
parser.add_argument('--regularisation', type=int, default=1) # whether to turn on regularisation in net (details in net definition)
parser.add_argument('--input_dropout', type=float, default=0.25)
parser.add_argument('--rnn_dropout', type=float, default=0.1)
parser.add_argument('--input_split', type=int, default=0) # 0 if provide image and saccade at the same time, 1 if provide image and saccade separately
parser.add_argument('--glimpse_loss', type=int, default=1) # 0 if not required, 1 if yes - CPC, 2 if yes - CPC but with no equal split of in-sequences and out-sequence pairs
parser.add_argument('--semantic_loss', type=int, default=0) # 0 if not required, 1 if yes - contrastive
parser.add_argument('--scene_loss', type=int, default=0) # 0 if not required, 1 if yes - BCE with logits (multihot object labels)
parser.add_argument('--gazeloc_loss', type=int, default=0) # 0 if not required, 1 if yes - MSE

parser.add_argument('--trainer', type=str, default='train_515') # train/train_515
parser.add_argument('--n_epochs', type=int, default=-1) # -1 if end on convergence, >0 if #epochs desired
parser.add_argument('--dva_dataset', type=str, default='NSD') # NSD
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--show_progress_bar', type=int, default=0)
parser.add_argument('--mix_pres', type=int, default=0)
parser.add_argument('--in_memory', type=int, default=1)

parser.add_argument('--save_nets', type=int, default=1) # save networks

parser.add_argument('--bbv', type=int, default=6) # 0 is RN50-init, 1/2 is RN50-IN trained, 3 is RN50-Barlowtwins, 4 is RN50-DVD-B, 5 is DINOv2B, 6 is RN50-simclr

args = parser.parse_args()

import torch
import numpy as np
import time
import random

from helpers.helper_funcs import get_Dataset_loaders, create_folders_logging, create_cpc_matrix, LinearFitScheduler
from models.helper_funcs import get_network_model, weights_init, get_optimizer, compute_losses

##################
## Hyperparameters
##################

base_lr = args.learning_rate
warmup_epochs = 5 # lr starts at base_lr/(lr scale factor) and scales up linearly for these many epochs
lr_scale_factor = 100 # lr scale factor

show_progress_bar = args.show_progress_bar # if you want to print the training progress bar
if args.semantic_loss or args.scene_loss or args.glimpse_loss:
    compute_contrastive_floor = 1
else:
    compute_contrastive_floor = 0

hyp = {
    'dataset': {
        'dataset_path': '/share/klab/datasets/GPN/', # Folder where dataset exists (end with '/')
        'in_memory': args.in_memory, # should we load the entire dataset in memory?
        'bbv': args.bbv, # which backbone version to use
        'dva_dataset': args.dva_dataset, # extent of glimpse decided given NSD/AVS parameters
    },
    'network': {
        'model': args.network_type, # model to be used
        'identifier': args.network_id, # identifier in case we run multiple versions of the net
        'timestep_multiplier': args.timestep_multiplier, # number of LSTM layers / number of iterations RNN runs for each glimpse with lateral connections
        'timesteps': args.timesteps, # how many gazes to be provided as input to lstm
        'gaze_type': args.gaze_type, # which gaze patterns to use - dg3/random/dg3p (permuted dg3)/dg3s (swap) (central biased random sampling)
        'n_rnn': args.n_rnn, # number of neurons in rnn layer
        'regularisation': args.regularisation,
        'input_dropout': args.input_dropout,
        'rnn_dropout': args.rnn_dropout,
        'analysis_mode': 0, # this will only return outputs
        'input_split': args.input_split, # 0 if provide image and saccade at the same time, 1 if provide image and saccade separately
        'recurrence': args.recurrence # 0 if no recurrence, 1 if recurrence
    },
    'optimizer': { # we do not need normalise data as RN50 already used BN before avgpool
        'type': 'adam', # optimizer to be used
        'lr': args.learning_rate, # learning rate - scheduler takes care of this!
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs, # number of epochs (full cycle through the dataset)
        'trainer': args.trainer, # train/train_1000/train_515
        'device': 'cuda', # device to train the network on: 'cuda', 'mps', 'cpu'
        'dataloader': { # request 10 cores at least
            'num_workers_train': 6, # number of cpu workers processing the batches 
            'num_workers_val_test': 3, # don't need as many workers for val/test 
            'prefetch_factor': 4, # number of batches kept in memory by each worker (providing quick access for the gpu)
        },
        'losses': {
            'glimpse_loss': args.glimpse_loss,
            'semantic_loss': args.semantic_loss,
            'scene_loss': args.scene_loss,
            'gazeloc_loss': args.gazeloc_loss,
            'provide_loc': args.provide_loc,
        }
    },
    'misc': {
        'use_amp': args.mix_pres==1, # use automatic mixed precision during training - forward pass .half(), backward full
        'save_logs': 1, # after how many epochs should we save a copy of the logs
        'save_net': 1 # after how many epochs should we save a copy of the net
    }
}

torch.manual_seed(hyp['network']['identifier'])
random.seed(hyp['network']['identifier'])
np.random.seed(hyp['network']['identifier'])

##########################
## Training and evaluation
##########################

if __name__ == '__main__':

    # load the dataset loaders to iterate over for training and eval (CS MAGIC)
    print(f'Loading data and preparing dataloaders for {args.trainer} and validation...')
    if args.in_memory == 1:
        print('Loading datasets in memory!')
    train_loader = get_Dataset_loaders(hyp, args.trainer)
    val_loader = get_Dataset_loaders(hyp, 'val')
    print(f'{args.trainer} and validation dataloaders are ready!\n')

    # create the network and initialize it
    print('Loading network...')
    net, net_name = get_network_model(hyp)
    net.apply(weights_init)
    net = net.float()
    net.to(hyp['optimizer']['device'])
    print('Network is ready!\n')

    # criterion and optimizer setup
    optimizer = get_optimizer(hyp,net) # optimizers for the entire network or the 4 modules - glimpse, semantic, scene, gazeloc
    scaler = torch.cuda.amp.GradScaler(enabled=hyp['misc']['use_amp']) # this is in service of mixed precision training
    if hyp['optimizer']['n_epochs'] == -1:
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=1e-2, verbose=True) # usual scheduler
        scheduler = LinearFitScheduler(optimizer, num_epochs=3, factor=1./5, min_percent_change=1.0, verbose=True) # 1% change necessary in 3 epochs else drop lr by 1/5
    
    # Warm-up scheduler - this already initialises the lr to base_lr/lr_scale_factor - has an internal counter which it uses to do its updates when .step() is called!
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch_h: ((base_lr - base_lr/lr_scale_factor) / (warmup_epochs-1)) * epoch_h + base_lr/lr_scale_factor)

    # logging losses and accuracies

    train_losses = []
    val_losses = []

    # creating folders for logging losses/acc and network weights
    if args.save_nets:
        log_path, net_path = create_folders_logging(net_name)
        print(f'Log_folders: {log_path} -- {net_path}')

    # saving the randomly initialized network
    if args.save_nets:
        torch.save(net.state_dict(), f'{net_path}/{net_name}_epoch_{-1}.pth') # saving random weights

    print('\nTraining begins here!\n')

    epoch = 1
    training_not_finished = 1

    while training_not_finished:

        start = time.time()

        torch.cuda.synchronize()

        if not show_progress_bar:
            print(f'Epoch: {epoch}')

        if epoch > 1:
            compute_contrastive_floor = 0
        
        train_loss_running = 0.0
        if compute_contrastive_floor:
            train_contrastive_loss_floor_running = 0.0 
            train_contrastive_loss_floor = 0.0
        batch = 0

        print('LR_main now: ',optimizer.param_groups[0]['lr'])

        for actvs,next_fix_rel_coords,fix_coords,semantic_embed,_,scene_multihot,_,_ in train_loader: 

            cpc_mask = create_cpc_matrix(actvs.shape[0]*(actvs.shape[1]-1),actvs.shape[1]-1).to(hyp['optimizer']['device'])

            actvs = actvs.to(hyp['optimizer']['device'])
            next_fix_rel_coords = next_fix_rel_coords.to(hyp['optimizer']['device'])

            fix_coords = fix_coords.to(hyp['optimizer']['device'])
            semantic_embed = semantic_embed.to(hyp['optimizer']['device'])
            scene_multihot = scene_multihot.to(hyp['optimizer']['device'])

            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):

                outputs = net(actvs[:,:-1,:],args.provide_loc*next_fix_rel_coords) # only feed actvs[:,:-1,:] as input as the last fixation isn't used as an input

                loss, contrastive_loss_floor = compute_losses(outputs,actvs,fix_coords,semantic_embed,scene_multihot,cpc_mask,hyp,compute_contrastive_floor)
            
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()

            train_loss_running += loss.item()
            if compute_contrastive_floor:
                train_contrastive_loss_floor_running += contrastive_loss_floor.item()

            batch += 1
            if show_progress_bar:
                print(f'Training Epoch {epoch}: Batch {batch} of {len(train_loader)}', end="\r")
            
        train_losses.append(train_loss_running/len(train_loader))
        if compute_contrastive_floor:
            train_contrastive_loss_floor = train_contrastive_loss_floor_running/len(train_loader)
        
        # getting validation loss and acc
        net.eval()
        val_loss_running = 0.0
        if compute_contrastive_floor:
            val_contrastive_loss_floor_running = 0.0
            val_contrastive_loss_floor = 0.0

        for actvs,next_fix_rel_coords,fix_coords,semantic_embed,_,scene_multihot,_,_ in val_loader:

            cpc_mask = create_cpc_matrix(actvs.shape[0]*(actvs.shape[1]-1),actvs.shape[1]-1).to(hyp['optimizer']['device'])

            actvs = actvs.to(hyp['optimizer']['device'])
            next_fix_rel_coords = next_fix_rel_coords.to(hyp['optimizer']['device'])

            fix_coords = fix_coords.to(hyp['optimizer']['device'])
            semantic_embed = semantic_embed.to(hyp['optimizer']['device'])
            scene_multihot = scene_multihot.to(hyp['optimizer']['device'])
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):

                outputs = net(actvs[:,:-1,:],args.provide_loc*next_fix_rel_coords) # only feed actvs[:,:-1,:] as input as the last fixation isn't used as an input

                loss, contrastive_loss_floor = compute_losses(outputs,actvs,fix_coords,semantic_embed,scene_multihot,cpc_mask,hyp,compute_contrastive_floor)

            val_loss_running += loss.item()
            if compute_contrastive_floor:
                val_contrastive_loss_floor_running += contrastive_loss_floor.item()

        net.train()

        val_losses.append(val_loss_running/len(val_loader))
        if compute_contrastive_floor:
            val_contrastive_loss_floor = val_contrastive_loss_floor_running/len(val_loader)

        print('Epoch time: ', "{:.2f}".format(time.time() - start), ' seconds\n')
        
        print(f'Train loss: {train_losses[-1]:.3f}')
        print(f'Val loss: {val_losses[-1]:.3f}\n')
        if compute_contrastive_floor:
            print(f'Train contrastive loss floor: {train_contrastive_loss_floor:.3f}')
            print(f'Val contrastive loss floor: {val_contrastive_loss_floor:.3f}\n')

        if (epoch) < warmup_epochs: # updating for next epoch's use!
            warmup_scheduler.step()
        elif hyp['optimizer']['n_epochs'] == -1:
            scheduler.step(train_losses[-1])
        
        if args.save_nets:
            if (epoch) % hyp['misc']['save_logs'] == 0:
                np.savez(log_path+'/loss_'+net_name+'.npz', train_loss=train_losses, val_loss=val_losses)
            if (epoch) % hyp['misc']['save_net'] == 0:
                if epoch > 1: # save only if val loss has decreased or stayed same (preference to later epochs)
                    if val_losses[-1] <= min(val_losses[:-1]):
                        print(f'Val loss decreased, saving network at epoch {epoch}...\n')
                        torch.save(net.state_dict(), f'{net_path}/{net_name}.pth')
                    # torch.save(net.state_dict(), f'{net_path}/{net_name}_epoch_{epoch}.pth')
                else:
                    print(f'Saving network at epoch {epoch}...\n')
                    torch.save(net.state_dict(), f'{net_path}/{net_name}.pth')

        if epoch > 1:
            # check if val loss has increased by more than 1% of min, if yes exit training
            if val_losses[-1] > 0.99*min(val_losses[:-1]):
                print(f'Val loss increased by more than 1% of min val loss, stopping training to prevent overfitting...\n')
                training_not_finished = 0

        max_mem_allocated = 0
        for i in range(torch.cuda.device_count()):
            device = f'cuda:{i}'
            max_mem_allocated += torch.cuda.max_memory_allocated(device) / (1024**3) * torch.cuda.device_count()
        print(f'Max GPU(s) memory used: {max_mem_allocated} Gb\n')

        epoch += 1
        if hyp['optimizer']['n_epochs'] > 0:
            if epoch > hyp['optimizer']['n_epochs']:
                training_not_finished = 0
                print('\n Done training! - #epochs completed\n')
        elif hyp['optimizer']['n_epochs'] == -1:
            if optimizer.param_groups[0]['lr'] <= 1e-8:
                training_not_finished = 0
                print('\n Done training! - LR reached 1e-8 i.e. converged\n')

    print('\n Done training!\n')

    # 
    if args.save_nets:
        print('Min val loss:',min(val_losses))
    else:
        print('Min val loss:',min(val_losses))
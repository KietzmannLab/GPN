import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
    
##################################
## Importing the network
##################################

def get_network_model(hyp):
    # import the req. network

    timestep_multiplier = hyp['network']['timestep_multiplier']
    timesteps = hyp['network']['timesteps']
    gaze_type = hyp['network']['gaze_type']
    network_id = hyp['network']['identifier']
    n_rnn = hyp['network']['n_rnn']
    regularisation = hyp['network']['regularisation']
    input_dropout = hyp['network']['input_dropout']
    rnn_dropout = hyp['network']['rnn_dropout']
    analysis_mode = hyp['network']['analysis_mode']
    input_split = hyp['network']['input_split']
    recurrence = hyp['network']['recurrence']

    semantic_loss = hyp['optimizer']['losses']['semantic_loss']
    scene_loss = hyp['optimizer']['losses']['scene_loss']
    glimpse_loss = hyp['optimizer']['losses']['glimpse_loss']
    gazeloc_loss = hyp['optimizer']['losses']['gazeloc_loss']
    provide_loc = hyp['optimizer']['losses']['provide_loc']

    r50v = hyp['dataset']['r50v']
    dva_dataset = hyp['dataset']['dva_dataset']

    trainer = hyp['optimizer']['trainer']
    lr = hyp['optimizer']['lr']

    if hyp['network']['model'] == 'lstm':

        from .GPN import lstm_gpn

        net = lstm_gpn(timestep_multiplier=timestep_multiplier,glimpse_loss=glimpse_loss,semantic_loss=semantic_loss,scene_loss=scene_loss,gazeloc_loss=gazeloc_loss,n_rnn=n_rnn,regularisation=regularisation,input_dropout=input_dropout,rnn_dropout=rnn_dropout,return_all_actvs=analysis_mode, input_split=input_split, recurrence=recurrence) 

        if timesteps == 6 and input_split == 0 and glimpse_loss == 1:
            net_name = f'gpn_n_{n_rnn}_tm_{timestep_multiplier}_recurrence_{recurrence}_loc_{provide_loc}_gaze_{gaze_type}_gdva_{dva_dataset}_num_{network_id}'
        else:
            net_name = f'gpn_lstm_n_{n_rnn}_tm_{timestep_multiplier}_t_{timesteps}_insplit_{input_split}_recurrence_{recurrence}_loc_{provide_loc}_reg_{regularisation}_indp_{input_dropout}_rnndp_{rnn_dropout}_gaze_{gaze_type}_gcpc_{glimpse_loss}_semc_{semantic_loss}_scc_{scene_loss}_locmse_{gazeloc_loss}_tr_{trainer}_r50v{r50v}_gdva_{dva_dataset}_lr_{lr}_num_{network_id}'

    print(f'\nNetwork name: {net_name}')

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"\nThe network has {params} trainable parameters\n")

    return net, net_name

def weights_init(m):
    # Xavier intialisation for conv and linear - LSTM has self-initialisation
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def get_optimizer(hyp,net):
    # selecting the optimizer

    if hyp['optimizer']['type'] == 'adam': # write an optimizer with access to the entire net and another for finetuning desired outputs
        return optim.Adam(net.parameters(),lr=1.)
    
def compute_losses(outputs,actvs,fix_coords,semantic_embed,scene_embed,cpc_mask,hyp,compute_contrastive_floor):

    loss_combined = 0.
    contrastive_loss_floor = 0.

    if hyp['optimizer']['losses']['glimpse_loss'] == 1:

        A = actvs[:,1:,:].reshape(-1, outputs[0].shape[2])
        B = outputs[0].reshape(-1, outputs[0].shape[2])
        
        similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (B / B.norm(dim=1, keepdim=True)).T
        A_filter_map = ((A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T) < 0.999 # places where activations are not repeated or EXTREMELY similar - else they make the loss inelegant
        # MSE: similarity_matrix = -torch.sqrt(torch.sum(A**2, dim=1, keepdim=True) + torch.sum(B**2, dim=1).unsqueeze(0) - 2 * torch.mm(A, B.t()))

        loss_combined += (similarity_matrix[(cpc_mask==2)&A_filter_map].mean() + similarity_matrix[cpc_mask==3].mean())/2 - similarity_matrix[cpc_mask==1].mean() 

        if compute_contrastive_floor:

            similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T
            contrastive_loss_floor += (similarity_matrix[(cpc_mask==2)&(similarity_matrix<0.999)].mean() + similarity_matrix[cpc_mask==3].mean())/2 - similarity_matrix[cpc_mask==1].mean() 

    elif hyp['optimizer']['losses']['glimpse_loss'] == 2:

        A = actvs[:,1:,:].reshape(-1, outputs[0].shape[2])
        B = outputs[0].reshape(-1, outputs[0].shape[2])
        
        similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (B / B.norm(dim=1, keepdim=True)).T
        # MSE: similarity_matrix = -torch.sqrt(torch.sum(A**2, dim=1, keepdim=True) + torch.sum(B**2, dim=1).unsqueeze(0) - 2 * torch.mm(A, B.t()))

        loss_combined += -similarity_matrix[cpc_mask==1].mean() + similarity_matrix[(cpc_mask == 2) | (cpc_mask == 3)].mean()

        if compute_contrastive_floor:

            similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T
            contrastive_loss_floor += -similarity_matrix[cpc_mask==1].mean() + similarity_matrix[(cpc_mask == 2) | (cpc_mask == 3)].mean()

    if hyp['optimizer']['losses']['semantic_loss'] == 1:

        A = semantic_embed.unsqueeze(1).repeat(1, outputs[1].shape[1], 1).reshape(-1, outputs[1].shape[2])
        B = outputs[1].reshape(-1, outputs[1].shape[2])
        
        similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (B / B.norm(dim=1, keepdim=True)).T

        loss_combined += similarity_matrix[cpc_mask==3].mean() - similarity_matrix[cpc_mask==1].mean()

        if compute_contrastive_floor:

            similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T
            contrastive_loss_floor += similarity_matrix[cpc_mask==3].mean() - similarity_matrix[cpc_mask==1].mean() 

    if hyp['optimizer']['losses']['scene_loss'] == 1:

        A = scene_embed.unsqueeze(1).repeat(1, outputs[2].shape[1], 1).reshape(-1, outputs[2].shape[2])
        B = outputs[2].reshape(-1, outputs[2].shape[2])
        
        similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (B / B.norm(dim=1, keepdim=True)).T

        loss_combined += similarity_matrix[cpc_mask==3].mean() - similarity_matrix[cpc_mask==1].mean()

        if compute_contrastive_floor:

            similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T
            contrastive_loss_floor += similarity_matrix[cpc_mask==3].mean() - similarity_matrix[cpc_mask==1].mean()

    elif hyp['optimizer']['losses']['scene_loss'] == 2:

        A = outputs[2].reshape(-1, outputs[2].shape[2])
        
        similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T

        loss_combined += similarity_matrix[cpc_mask==3].mean() - similarity_matrix[cpc_mask==2].mean()

        if compute_contrastive_floor: # crazy LLM upper bound!

            A = semantic_embed.unsqueeze(1).repeat(1, outputs[2].shape[1], 1).reshape(-1, semantic_embed.shape[1])

            similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T
            contrastive_loss_floor += similarity_matrix[cpc_mask==3].mean() - similarity_matrix[cpc_mask==2].mean()

    if hyp['optimizer']['losses']['gazeloc_loss'] == 1:

        if hyp['optimizer']['losses']['glimpse_loss'] == 1: # then ofc only relative coords are provided so have to predict next frame
            loss_combined += torch.mean((outputs[3] - fix_coords[:,1:,:])**2)**0.5
        else: # absolute coords are provided so predict them
            loss_combined += torch.mean((outputs[3] - fix_coords[:,:-1,:])**2)**0.5
    
    return loss_combined, contrastive_loss_floor


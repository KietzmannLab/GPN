### Global package imports

import os, h5py, sys
import numpy as np
from train.helpers.helper_funcs import create_cpc_matrix
import torch
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import pdist
import pickle
from scipy.stats import spearmanr, pearsonr, zscore
from matplotlib.lines import Line2D
import seaborn as sns
import statsmodels.api as sm
from matplotlib.patches import Patch
import tables
import cv2
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
plt.style.use('seaborn-v0_8-colorblind')
import pprint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random
from collections import defaultdict
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import re

from scipy.special import logsumexp
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from scipy.ndimage import zoom
PYTORCH_ENABLE_MPS_FALLBACK=1
import deepgaze_pytorch
from pysaliency.models import sample_from_logdensity

### Network name abbrv

def get_net_params(net_abbrv):
    # returns recurrence, provide_loc, bbv, n_rnn, timestep_multiplier, rnn_dropout, input_dropout, gaze_type

    # GPN-RS/S/B-ILSVRC: 0.1, 0.1 
    # GPN-R-ILSVRC: 0.25, 0.25
    # GPN-R-SimCLR: 0.25, 0.1
    # GPN-S/B/RS-SimCLR: 0.1, 0.1
    # GPN-B/S/R/RS-Dinov2B: 0.1, 0.1

    # c/sGSN-R-SimCLR: 0.5, 0.5
    # cGSN-RS-SimCLR: 0.5, 0.1
    # sGSN-RS-SimCLR: 0.5, 0.5

    # bbv: 0 if rn50-init, 1/2 if rn50-IN, 3 if barlow-twins-rn50-IN, 4 if DVD-B, 5 if DINOv2b, 6 if RN50-simclr
    # gaze_type: dg3/random/dg3p/dg3r 

    print(f'Getting hyperparameters for {net_abbrv}')
    r_value = None
    s_value = None
    bbv_value = None
    n_rnn_value = 1024
    l_rnn_value = 3
    dpout_in_value = None
    dpout_rnn_value = None
    gaze_type_value = 'random' if '-random' in net_abbrv else ('dg3p' if '-dg3p' in net_abbrv else ('dg3r' if '-dg3r' in net_abbrv else 'dg3'))
    dva_dataset = 'AVS' if '-gsmall' in net_abbrv else 'NSD'

    if '-R-' in net_abbrv or '-RS-' in net_abbrv:
        r_value = 1
    else:
        r_value = 0
    if '-S-' in net_abbrv or '-RS-' in net_abbrv:
        s_value = 1
    else:
        s_value = 0
    if 'SimCLR' in net_abbrv:
        bbv_value = 6
    elif 'Dinov2B' in net_abbrv:
        bbv_value = 5
    elif 'ILSVRC' in net_abbrv:
        bbv_value = 1
    if 'random' in net_abbrv:
        gaze_type_value = 'random'
    elif 'dg3p' in net_abbrv:
        gaze_type_value = 'dg3p'
    elif 'dg3s' in net_abbrv:
        gaze_type_value = 'dg3s'
    if  net_abbrv == 'sGSN-R-SimCLR' or net_abbrv == 'sGSN-RS-SimCLR':
        dpout_in_value = 0.5
        dpout_rnn_value = 0.5
    elif net_abbrv == 'cGSN-R-SimCLR' or net_abbrv == 'cGSN-RS-SimCLR':
        dpout_in_value = 0.5
        dpout_rnn_value = 0.25
    elif net_abbrv == 'GPN-R-SimCLR' or net_abbrv == 'GPN-R-SimCLR-dg3p' or net_abbrv == 'GPN-R-SimCLR-dg3r' or net_abbrv == 'GPN-R-SimCLR-random' or net_abbrv == 'GPN-R-SimCLR-gsmall':
        dpout_in_value = 0.25
        dpout_rnn_value = 0.1
    elif net_abbrv == 'GPN-RS-ILSVRC' or net_abbrv == 'GPN-S-ILSVRC' or net_abbrv == 'GPN-B-ILSVRC' or net_abbrv == 'GPN-RS-SimCLR' or net_abbrv == 'GPN-S-SimCLR' or net_abbrv == 'GPN-B-SimCLR' or net_abbrv == 'GPN-R-Dinov2B' or net_abbrv == 'GPN-RS-Dinov2B' or net_abbrv == 'GPN-S-Dinov2B' or net_abbrv == 'GPN-B-Dinov2B':
        dpout_in_value = 0.1
        dpout_rnn_value = 0.1
    elif net_abbrv == 'GPN-R-ILSVRC':
        dpout_in_value = 0.25
        dpout_rnn_value = 0.25
    elif net_abbrv == 'sGSN-R-SimCLR-opt':
        dpout_in_value = 0.5
        dpout_rnn_value = 0.1
        n_rnn_value = 512
        l_rnn_value = 1
    return r_value, s_value, bbv_value, n_rnn_value, l_rnn_value, dpout_in_value, dpout_rnn_value, gaze_type_value, dva_dataset

### Package the required hyperparameters for the model

def get_hyp(dataset_path, gaze_type, dva_dataset, n_rnn, timestep_multiplier, rnn_dropout, input_dropout, net_id, load_epoch, recurrence, provide_loc, bbv, net_abbrv):

    hyp = {
        'dataset': {
            'dataset_path': dataset_path, # Folder where dataset exists (end with '/')
            'in_memory': 1, # should we load the entire dataset in memory?
            'bbv': bbv, # which r50 version to use
            'dva_dataset': dva_dataset, # extent of glimpse decided given NSD/AVS parameters
        },
        'network': {
            'model': 'rn18-lstm' if 'e2e' in net_abbrv else 'lstm', # model to be used
            'identifier': net_id, # identifier in case we run multiple versions of the net
            'timestep_multiplier': timestep_multiplier, # number of LSTM layers / number of iterations RNN runs for each glimpse with lateral connections
            'timesteps': 6, # how many gazes to be provided as input to lstm
            'gaze_type': gaze_type, # which gaze patterns to use - dg3/random/dg3p (permuted dg3)/cb_rand (central bias random)
            'n_rnn': n_rnn, # number of neurons in rnn layer
            'regularisation': 1,
            'input_dropout': input_dropout,
            'rnn_dropout': rnn_dropout,
            'analysis_mode': 1, # 0 will only return outputs, 1 internals
            'input_split': 0, # 0 if provide image and saccade at the same time, 1 if provide image and saccade separately
            'recurrence': recurrence, # 0 if no recurrence, 1 if recurrence
            'load_epoch': load_epoch, # which epoch to load the model from
            'analysis_mode': 1, # returns all activations
        },
        'optimizer': { # we do not need normalise data as RN50 already used BN before avgpool
            'batch_size': 206,  # this is for local code run - such that we get 25 (or 206) batches on test_515 - we can do stats over them
            'device': 'cpu',
            'dataloader': { 
                'num_workers_train': 1, # number of cpu workers processing the batches 
                'num_workers_val_test': 1, # don't need as many workers for val/test 
                'prefetch_factor': 1, # number of batches kept in memory by each worker (providing quick access for the gpu)
            },
            'losses': {
                'glimpse_loss': 1 if not 'GSN' in net_abbrv else 0, # 1 if we want to have glimpse prediction loss
                'semantic_loss': 0 if not 'sGSN' in net_abbrv else 1, # 1 if we want to have semantic prediction loss
                'scene_loss': 0 if not 'cGSN' in net_abbrv else 1, # 1 if we want to have scene prediction loss
                'gazeloc_loss': 0,
                'provide_loc': provide_loc
            },
            'trainer': 'train_515', # train/train_515
            'lr': 0.0001, # learning rate - scheduler takes care of this!
        }
    }

    return hyp

### Data loader for the test_515 dataset

def get_test_515_loader(hyp,shuffler=False):

    dataset_path = hyp['dataset']['dataset_path']
    gaze_type = hyp['network']['gaze_type']
    # gaze_type = 'dg3' # use dg3 glimpses to make comparisons fair
    timesteps = hyp['network']['timesteps']
    dva_dataset = hyp['dataset']['dva_dataset']
    bbv = hyp['dataset']['bbv']

    split_data = CocoGaze_t515(dataset_path=dataset_path, gaze_type=gaze_type, timesteps=timesteps, in_memory=hyp['dataset']['in_memory'], dva_dataset=dva_dataset, bbv=bbv)

    data_loader = torch.utils.data.DataLoader(split_data, batch_size=hyp['optimizer']['batch_size'], shuffle=shuffler,num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor'])

    return data_loader

def get_test_515_loader_e2e(hyp,shuffler=False):

    dataset_path = hyp['dataset']['dataset_path']
    timesteps = hyp['network']['timesteps']

    split_data = CocoGaze_t515_e2e(dataset_path=dataset_path, timesteps=timesteps, in_memory=hyp['dataset']['in_memory'], dva_dataset=hyp['dataset']['dva_dataset'])

    data_loader = torch.utils.data.DataLoader(split_data, batch_size=hyp['optimizer']['batch_size'], shuffle=shuffler, num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'], prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor'])
        
    return data_loader

class CocoGaze_t515_e2e(torch.utils.data.Dataset):
    #Import dataset splitwise

    def __init__(self, dataset_path, timesteps, in_memory, dva_dataset='NSD', img_scale=91): # only in_memory, dg3 agze_type, implemented here!

        self.gaze_type = 'dg3'
        self.in_memory = in_memory
        self.timesteps = timesteps
        self.dva_dataset = dva_dataset
        if self.dva_dataset == 'NSD':
            self.glimpse_size = 91
        elif self.dva_dataset == 'AVS':
            self.glimpse_size = 36
        self.preprocess = transforms.Compose([
            transforms.Resize(self.glimpse_size, antialias=True), # Resize shortest side to glimpse_size
            transforms.ConvertImageDtype(torch.float), # Convert image to float,
        ])
        self.img_scale = img_scale

        f_dg3_fixations = h5py.File(dataset_path+'coco_dg3_fixations_test_515.h5', 'r')
        f_coco = h5py.File(dataset_path+'coco-test-515.h5', 'r')

        self.coco_imgs = torch.from_numpy(f_coco['images'][()])
        self.dg3_fixs = torch.from_numpy(f_dg3_fixations['dg3_fixations'][()])

    def __len__(self):
        return (self.dg3_fixs.shape[0]*self.dg3_fixs.shape[1]) # each image * gaze trace is considered a sample
    
    def __getitem__(self, idx): # accepts ids for img/trace and returns for glimpse sequences to the Dataloader

        trace_n = idx//self.dg3_fixs.shape[0]
        img_n = idx%self.dg3_fixs.shape[0]

        glimpse_ext = self.glimpse_size

        img = self.coco_imgs[img_n,:,:,:]
        im_h = torch.zeros((int(img.shape[0]+glimpse_ext),int(img.shape[1]+glimpse_ext),3),dtype=img.dtype)
        im_h[int(glimpse_ext/2):int(glimpse_ext/2)+img.shape[0],int(glimpse_ext/2):int(glimpse_ext/2)+img.shape[1],:] = img
        img = im_h
        
        dg3_fixs = self.dg3_fixs[img_n,trace_n,:,:]
    
        glimpse_seq = torch.zeros((self.timesteps+1, 3, self.img_scale, self.img_scale))
        next_fix_rel_coords = torch.zeros((self.timesteps, 2))
        fix_coords = torch.zeros((self.timesteps+1, 2))

        for gli_idx in range(self.timesteps+1):
            
            x_ext_low = int(dg3_fixs[gli_idx,0])
            x_ext_high = int(dg3_fixs[gli_idx,0]+glimpse_ext)
            y_ext_low = int(dg3_fixs[gli_idx,1])
            y_ext_high = int(dg3_fixs[gli_idx,1]+glimpse_ext)

            glimpse = img[y_ext_low:y_ext_high,x_ext_low:x_ext_high,:]
            glimpse = glimpse.permute(2,0,1)
            if self.preprocess is not None:
                glimpse = self.preprocess(glimpse)
            glimpse_seq[gli_idx,:,:,:] = glimpse

            if gli_idx < self.timesteps:
                next_fix_rel_coords[gli_idx,:] = dg3_fixs[gli_idx+1,:] - dg3_fixs[gli_idx,:]
            fix_coords[gli_idx,:] = dg3_fixs[gli_idx,:] - 128

        img_n = torch.Tensor([img_n])
        trace_n = torch.Tensor([trace_n])

        return glimpse_seq, next_fix_rel_coords, fix_coords, img_n, trace_n

class CocoGaze_t515(torch.utils.data.Dataset):

    def __init__(self, dataset_path, gaze_type, timesteps, in_memory, dva_dataset='NSD', bbv=1):

        self.root_dir = dataset_path
        gaze_map = {'dg3': 0, 'random': 1, 'dg3p': 2, 'dg3r': 3}
        self.gaze_type = gaze_map[gaze_type] # 0 if dg3, 1 if random, 2 if dg3_permuted, 3 if dg3r
        self.in_memory = in_memory
        self.timesteps = timesteps
        self.bbv = bbv

        self.dataset_str = f'coco_{dva_dataset}_dg3fix91_r50v{self.bbv}ap_7fix'
        print(f'Loading dataset: {self.dataset_str} with gaze type: {gaze_type} and bbv: {self.bbv}')

        path_here = dataset_path + f'{self.dataset_str}_test_515.h5'
        path_coco_h = dataset_path + 'coco-test-515.h5'

        if in_memory == 1:
            with h5py.File(path_here, "r") as f:
                self.actvs = torch.from_numpy(f['dg3_fix_actvs'][:,:,:timesteps+1,self.gaze_type,:][()]).float()
                self.len = self.actvs.shape
                # self.gist_actvs = torch.from_numpy(f['test']['dg3_gist_actvs'][:,:,:timesteps+1,self.gaze_type,:][()]).float()
                self.next_fix_rel_coords = torch.from_numpy(f['next_fix_rel_coords'][:,:,:timesteps,self.gaze_type,:][()]).float()
                self.next_fix_coords = torch.from_numpy(f['next_fix_coords'][:,:,:timesteps,self.gaze_type,:][()]).float()
                self.semantic_embed = torch.from_numpy(f['mpnet_embeddings'][()]).float()
                self.scene_embed = torch.from_numpy(f['full_image_actvs'][()]).float()
            with h5py.File(path_coco_h, "r") as f:
                self.scene_multihot = torch.from_numpy(f['multihot_labels'][()]).float()
        else:
            with h5py.File(path_here, "r") as f:
                self.len = f['dg3_fix_actvs'][:,:,:timesteps+1,self.gaze_type,:].shape

    def __len__(self):
        return (self.len[0]*self.len[1]) # each image * gaze trace is considered a sample
    
    def __getitem__(self, idx): # accepts ids for img/trace and returns for fixations, the actvs, fixs_coords, to the Dataloader

        img_n = idx//self.len[1]
        trace_n = idx%self.len[1]

        if self.in_memory == 1:

            actvs = self.actvs[img_n,trace_n,:,:]
            # gist_actvs = self.gist_actvs[img_n,trace_n,:,:]
            next_fix_rel_coords = self.next_fix_rel_coords[img_n,trace_n,:,:]
            fix_coords = self.next_fix_coords[img_n,trace_n,:,:]
            fix_coords = torch.cat((torch.Tensor([[0, 0]]), fix_coords), dim=0)
            semantic_embed = self.semantic_embed[img_n,:]
            scene_embed = self.scene_embed[img_n,:]
            scene_multihot = self.scene_multihot[img_n,:]

        else:
            
            path_here = self.root_dir + f'{self.dataset_str}_test_515.h5'
            path_coco_h = self.root_dir + 'coco-test-515.h5'
            with h5py.File(path_here, "r") as f:

                actvs = torch.from_numpy(f['dg3_fix_actvs'][img_n,trace_n,:self.timesteps+1,self.gaze_type,:][()]).float()
                # gist_actvs = torch.from_numpy(f['dg3_gist_actvs'][img_n,trace_n,:self.timesteps+1,self.gaze_type,:][()]).float()
                next_fix_rel_coords = torch.from_numpy(f['next_fix_rel_coords'][img_n,trace_n,:self.timesteps,self.gaze_type,:][()]).float()
                fix_coords = torch.from_numpy(f['next_fix_coords'][img_n,trace_n,:self.timesteps,self.gaze_type,:][()]).float()
                fix_coords = torch.cat((torch.Tensor([[0, 0]]), fix_coords), dim=0)
                semantic_embed = torch.from_numpy(f['mpnet_embeddings'][img_n,:][()]).float()
                scene_embed = torch.from_numpy(f['full_image_actvs'][img_n,:][()]).float()
            with h5py.File(path_coco_h, "r") as f:
                scene_multihot = torch.from_numpy(f['multihot_labels'][img_n,:][()]).float()

        img_n = torch.Tensor([img_n])
        trace_n = torch.Tensor([trace_n])

        # return actvs, gist_actvs, next_fix_rel_coords, fix_coords, semantic_embed, scene_embed, img_n, trace_n
        return actvs, next_fix_rel_coords, fix_coords, semantic_embed, scene_embed, scene_multihot, img_n, trace_n

### Get dg3 glimpse cutouts and scene starter for COCO-515 dataset

def get_coco515_dg3_glimpse_cutouts_and_scene(scene_ids,trace_ids,glimpse_ids,glimpse_ext=91,bbv=1,gaze_type='dg3'):

    f_scene = h5py.File('datasets/coco-test-515.h5', 'r')
    f_glimpse_coord = h5py.File(f'datasets/coco_NSD_dg3fix91_r50v{bbv}ap_7fix_test_515.h5', 'r')
    gaze_map = {'dg3': 0, 'random': 1, 'dg3p': 2, 'dg3r': 3}
    glimpse_cutouts = []
    scene_starter = None

    for scene_id,trace_id,glimpse_id in zip(scene_ids,trace_ids,glimpse_ids):

        img = f_scene['images'][scene_id][()]
        if len(glimpse_cutouts) == 0:
            scene_starter = img
        glimpse_coord = f_glimpse_coord['next_fix_coords'][scene_id,trace_id,glimpse_id-1,gaze_map[gaze_type],:][()] if glimpse_id > 0 else [0,0]
        x = int(glimpse_coord[0]) + 128
        y = int(glimpse_coord[1]) + 128

        im_size = img.shape[0]

        im_h = np.zeros([int(im_size+glimpse_ext),int(im_size+glimpse_ext),3],dtype=img.dtype)
        im_h[int(glimpse_ext/2):int(glimpse_ext/2)+im_size,int(glimpse_ext/2):int(glimpse_ext/2)+im_size,:] = img
        
        glimpse_cutout = im_h[y:y+glimpse_ext,x:x+glimpse_ext,:]
        glimpse_cutouts.append(glimpse_cutout)

    return glimpse_cutouts, scene_starter

def get_coco515_dg3_glimpse_cutouts_and_scene_e2e(scene_ids,trace_ids,glimpse_ids,glimpse_ext=91):

    f_scene = h5py.File('datasets/coco-test-515.h5', 'r')
    f_glimpse_coord = h5py.File(f'datasets/coco_dg3_fixations_test_515.h5', 'r')
    glimpse_cutouts = []

    for scene_id,trace_id,glimpse_id in zip(scene_ids,trace_ids,glimpse_ids):

        img = f_scene['images'][scene_id][()]
        if len(glimpse_cutouts) == 0:
            scene_starter = img
        glimpse_coord = f_glimpse_coord['dg3_fixations'][scene_id,trace_id,glimpse_id,:][()]
        x = int(glimpse_coord[0]) 
        y = int(glimpse_coord[1]) 

        im_size = img.shape[0]

        im_h = np.zeros([int(im_size+glimpse_ext),int(im_size+glimpse_ext),3],dtype=img.dtype)
        im_h[int(glimpse_ext/2):int(glimpse_ext/2)+im_size,int(glimpse_ext/2):int(glimpse_ext/2)+im_size,:] = img
        
        glimpse_cutout = im_h[y:y+glimpse_ext,x:x+glimpse_ext,:]
        glimpse_cutouts.append(glimpse_cutout)

    return glimpse_cutouts, scene_starter

### CPC loss elements extraction

def compute_cossim(outputs,actvs,semantic_embed=None,cpc_mask=None,hyp=None):

    if hyp['optimizer']['losses']['glimpse_loss']:
        dim_h = 0
        A = actvs[:,1:,:].reshape(-1, actvs.shape[2])
        A_now  = actvs[:,:-1,:].reshape(-1, actvs.shape[2])
        B = outputs[0].reshape(-1, actvs.shape[2])
    elif hyp['optimizer']['losses']['semantic_loss']:
        dim_h = 1
        A = semantic_embed.unsqueeze(1).repeat(1, outputs[1].shape[1], 1).reshape(-1, outputs[1].shape[2])
        B = outputs[1].reshape(-1, outputs[1].shape[2])

    cos_sim_target = [0.0 for idx in range(outputs[dim_h].shape[1])]
    rank_target = [0.0 for idx in range(outputs[dim_h].shape[1])]
    if hyp['optimizer']['losses']['glimpse_loss']:
        cos_sim_other_fix = [0.0 for idx in range(outputs[dim_h].shape[1])]
        cos_sim_current_fix = [0.0 for idx in range(outputs[dim_h].shape[1])]
        rank_other_fix = [0.0 for idx in range(outputs[dim_h].shape[1])]
    cos_sim_other_sc = [0.0 for idx in range(outputs[dim_h].shape[1])]
    cos_sim_diff = [0.0 for idx in range(outputs[dim_h].shape[1])]
    rank_other_sc = [0.0 for idx in range(outputs[dim_h].shape[1])]

    cosine_similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (B / B.norm(dim=1, keepdim=True)).T
    A_filter_map = ((A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T) < 0.999
    if hyp['optimizer']['losses']['glimpse_loss']:
        cosine_similarity_current = (A_now / A_now.norm(dim=1, keepdim=True)) * (B / B.norm(dim=1, keepdim=True))

    for idx in range(outputs[dim_h].shape[1]):
        idxs_h = outputs[dim_h].shape[1]*np.arange(outputs[dim_h].shape[0]) + idx
        cs_sm_mt_h = cosine_similarity_matrix[:,idxs_h]
        cs_sm_mt_h_ranks = cs_sm_mt_h.detach().numpy().argsort(axis=0).argsort(axis=0)
        cpc_ms_h = cpc_mask[:,idxs_h]
        A_filter_map_h = A_filter_map[:,idxs_h]
        cos_sim_target[idx] += cs_sm_mt_h[cpc_ms_h==1].mean().detach().numpy()
        rank_target[idx] += cs_sm_mt_h_ranks[cpc_ms_h==1].mean()
        if hyp['optimizer']['losses']['glimpse_loss']:
            cos_sim_other_fix[idx] += cs_sm_mt_h[(cpc_ms_h==2)&A_filter_map_h].mean().detach().numpy()
            cos_sim_current_fix[idx] += cosine_similarity_current[idxs_h].sum(dim=1).mean().detach().numpy()
            rank_other_fix[idx] += cs_sm_mt_h_ranks[(cpc_ms_h==2)&A_filter_map_h].mean()
        cos_sim_other_sc[idx] += cs_sm_mt_h[cpc_ms_h==3].mean().detach().numpy()
        rank_other_sc[idx] += cs_sm_mt_h_ranks[cpc_ms_h==3].mean()
        if hyp['optimizer']['losses']['glimpse_loss']:
            cos_sim_diff[idx] += cos_sim_target[idx] - (cos_sim_other_fix[idx]+cos_sim_other_sc[idx])/2
        elif hyp['optimizer']['losses']['semantic_loss']:
            cos_sim_diff[idx] += cos_sim_target[idx] - cos_sim_other_sc[idx]

    if hyp['optimizer']['losses']['glimpse_loss']:
        return cos_sim_target, cos_sim_other_fix, cos_sim_other_sc, cos_sim_diff, cos_sim_current_fix, rank_target, rank_other_sc, rank_other_fix
    else:
        return cos_sim_target, cos_sim_other_sc, cos_sim_diff, rank_target, rank_other_sc

### Extracting the activations and metrics from GPN

def extract_activations_metrics(dataloader, net, hyp):

    timesteps = hyp['network']['timesteps']
    glimpse_loss = hyp['optimizer']['losses']['glimpse_loss']
    semantic_loss = hyp['optimizer']['losses']['semantic_loss']

    n_samples_batch = [0.0 for _ in range(len(dataloader))]
    if glimpse_loss or semantic_loss:
        cos_sim_target = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        rank_target = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        cos_sim_other_sc = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        cos_sim_diff = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
    else:
        bce_loss = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
    if glimpse_loss:
        cos_sim_other_fix = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        cos_sim_other_fix_in = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        cos_sim_current_fix = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        cos_sim_current_fix_in = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        cos_sim_target_in = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        cos_sim_other_sc_in = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        cos_sim_diff_in = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]

    lstm_reps = {}
    output_reps = []
    scene_reps = []
    scene_multihot_reps = []
    llm_reps = []
    actv_reps = []
    fix_coords_all = []
    next_fix_rel_coords_all = []
    img_n_all = []
    trace_n_all = []

    count = 0
    
    for actvs,next_fix_rel_coords,fix_coords,semantic_embed,scene_embed,scene_multihot,img_n,trace_n in dataloader:

        n_samples_batch[count] = actvs.shape[0]

        cpc_mask = create_cpc_matrix(actvs.shape[0]*(actvs.shape[1]-1),actvs.shape[1]-1).to(hyp['optimizer']['device'])

        actvs = actvs.to(hyp['optimizer']['device'])
        next_fix_rel_coords = next_fix_rel_coords.to(hyp['optimizer']['device'])

        fix_coords = fix_coords.to(hyp['optimizer']['device']) # these are fixations' coords relative to the central fixation, computed on 256px images by deepgaze3! first fixation is [0,0]
        semantic_embed = semantic_embed.to(hyp['optimizer']['device'])
        scene_embed = scene_embed.to(hyp['optimizer']['device'])
        scene_multihot = scene_multihot.to(hyp['optimizer']['device'])
        
        activations, outputs= net(actvs[:,:-1,:],hyp['optimizer']['losses']['provide_loc']*next_fix_rel_coords) # only feed actvs[:,:-1,:] as input as the last fixation isn't used as an input

        for key in activations.keys():
            if key not in ['rn50_glimpse','joint_proj','glimpse_hidden','glimpse_output']:
                if key not in lstm_reps:
                    lstm_reps[key] = []
                lstm_reps[key].append(activations[key].detach().cpu().numpy())
        scene_reps.append(scene_embed.detach().numpy())
        scene_multihot_reps.append(scene_multihot.detach().numpy())
        fix_coords_all.append(fix_coords.detach().numpy())
        llm_reps.append(semantic_embed.detach().numpy())
        actv_reps.append(activations['rn50_glimpse'].detach().numpy())
        next_fix_rel_coords_all.append(next_fix_rel_coords.detach().numpy())
        img_n_all.append(img_n.numpy())
        trace_n_all.append(trace_n.numpy())
        idx_output_h = 0 if glimpse_loss else 1 if semantic_loss else 2
        output_reps.append(outputs[idx_output_h].detach().cpu().numpy())

        if glimpse_loss:
            cos_sim_target_h, cos_sim_other_fix_h, cos_sim_other_sc_h, cos_sim_diff_h, cos_sim_current_fix_h, rank_target_h, _, _ = compute_cossim(outputs,actvs,semantic_embed,cpc_mask,hyp)
            cos_sim_target_h_in, cos_sim_other_fix_h_in, cos_sim_other_sc_h_in, cos_sim_diff_h_in, cos_sim_current_fix_h_in, _, _, _ = compute_cossim([actvs[:,:-1,:]],actvs,semantic_embed,cpc_mask,hyp)
        elif semantic_loss:
            cos_sim_target_h, cos_sim_other_sc_h, cos_sim_diff_h, rank_target_h, _ = compute_cossim(outputs,actvs,semantic_embed,cpc_mask,hyp)
        else:
            bce_loss_h = []
            for t in range(timesteps):
                bce_loss_h.append(torch.nn.BCEWithLogitsLoss()(outputs[2][:,t,:], scene_multihot).detach().cpu().numpy())
            bce_loss_h = np.array(bce_loss_h)

        if glimpse_loss or semantic_loss:
            cos_sim_target[count] = np.array(cos_sim_target_h)
            rank_target[count] = np.array(rank_target_h)
            cos_sim_other_sc[count] = np.array(cos_sim_other_sc_h)
            cos_sim_diff[count] = np.array(cos_sim_diff_h)
        else:
            bce_loss[count] = np.array(bce_loss_h)
        if glimpse_loss:
            cos_sim_other_fix[count] = np.array(cos_sim_other_fix_h)
            cos_sim_other_fix_in[count] = np.array(cos_sim_other_fix_h_in)
            cos_sim_current_fix[count] += np.array(cos_sim_current_fix_h)
            cos_sim_current_fix_in[count] += np.array(cos_sim_current_fix_h_in)
            cos_sim_target_in[count] = np.array(cos_sim_target_h_in)
            cos_sim_other_sc_in[count] = np.array(cos_sim_other_sc_h_in)
            cos_sim_diff_in[count] = np.array(cos_sim_diff_h_in)

        print(f'Batch {count} of {len(dataloader)}')
        count += 1

    if glimpse_loss:
        return cos_sim_target, cos_sim_other_fix, cos_sim_other_sc, cos_sim_diff, cos_sim_current_fix, rank_target, lstm_reps, scene_reps, scene_multihot_reps, llm_reps, actv_reps, fix_coords_all, next_fix_rel_coords_all, img_n_all, trace_n_all, n_samples_batch, output_reps, cos_sim_target_in, cos_sim_other_fix_in, cos_sim_other_sc_in, cos_sim_diff_in, cos_sim_current_fix_in
    elif semantic_loss:
        return cos_sim_target, cos_sim_other_sc, cos_sim_diff, rank_target, lstm_reps, scene_reps, scene_multihot_reps, llm_reps, actv_reps, fix_coords_all, next_fix_rel_coords_all, img_n_all, trace_n_all, n_samples_batch, output_reps
    else:
        return bce_loss, lstm_reps, scene_reps, scene_multihot_reps, llm_reps, actv_reps, fix_coords_all, next_fix_rel_coords_all, img_n_all, trace_n_all, n_samples_batch, output_reps
    
def extract_activations_metrics_e2e(dataloader, net, hyp):

    timesteps = hyp['network']['timesteps']
    glimpse_loss = hyp['optimizer']['losses']['glimpse_loss']

    n_samples_batch = [0.0 for _ in range(len(dataloader))]
    cos_sim_target = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
    cos_sim_target_in = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
    rank_target = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
    if glimpse_loss:
        cos_sim_other_fix = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        cos_sim_other_fix_in = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        cos_sim_current_fix = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
        cos_sim_current_fix_in = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
    cos_sim_other_sc = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
    cos_sim_other_sc_in = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
    cos_sim_diff = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]
    cos_sim_diff_in = [np.array([0.0 for idx in range(timesteps)]) for _ in range(len(dataloader))]

    lstm_reps = {}
    output_reps = []
    scene_reps = []
    llm_reps = []
    actv_reps = []
    fix_coords_all = []
    next_fix_rel_coords_all = []
    img_n_all = []
    trace_n_all = []

    count = 0
    
    for glimpse_seq, next_fix_rel_coords, fix_coords, img_n, trace_n in dataloader:

        activations, outputs, actvs = net(glimpse_seq, hyp['optimizer']['losses']['provide_loc']*next_fix_rel_coords)

        n_samples_batch[count] = actvs.shape[0]

        cpc_mask = create_cpc_matrix(actvs.shape[0]*(actvs.shape[1]-1),actvs.shape[1]-1).to(hyp['optimizer']['device'])

        actvs = actvs.to(hyp['optimizer']['device'])
        next_fix_rel_coords = next_fix_rel_coords.to(hyp['optimizer']['device'])

        fix_coords = fix_coords.to(hyp['optimizer']['device']) # these are fixations' coords relative to the central fixation, computed on 256px images by deepgaze3! first fixation is [0,0]

        for key in activations.keys():
            if key not in ['rn50_glimpse','joint_proj','glimpse_hidden','glimpse_output']:
                if key not in lstm_reps:
                    lstm_reps[key] = []
                lstm_reps[key].append(activations[key].detach().cpu().numpy())
        fix_coords_all.append(fix_coords.detach().numpy())
        actv_reps.append(activations['rn50_glimpse'].detach().numpy())
        next_fix_rel_coords_all.append(next_fix_rel_coords.detach().numpy())
        img_n_all.append(img_n.numpy())
        trace_n_all.append(trace_n.numpy())
        output_reps.append(outputs[0].detach().cpu().numpy())

        cos_sim_target_h, cos_sim_other_fix_h, cos_sim_other_sc_h, cos_sim_diff_h, cos_sim_current_fix_h, rank_target_h, _, _ = compute_cossim(outputs,actvs,semantic_embed=None,cpc_mask=cpc_mask,hyp=hyp)
        cos_sim_target_h_in, cos_sim_other_fix_h_in, cos_sim_other_sc_h_in, cos_sim_diff_h_in, cos_sim_current_fix_h_in, _, _, _ = compute_cossim([actvs[:,:-1,:]],actvs,semantic_embed=None,cpc_mask=cpc_mask,hyp=hyp)

        cos_sim_target[count] = np.array(cos_sim_target_h)
        cos_sim_target_in[count] = np.array(cos_sim_target_h_in)
        rank_target[count] = np.array(rank_target_h)
        if glimpse_loss:
            cos_sim_other_fix[count] = np.array(cos_sim_other_fix_h)
            cos_sim_other_fix_in[count] = np.array(cos_sim_other_fix_h_in)
            cos_sim_current_fix[count] += np.array(cos_sim_current_fix_h)
            cos_sim_current_fix_in[count] += np.array(cos_sim_current_fix_h_in)
        cos_sim_other_sc[count] = np.array(cos_sim_other_sc_h)
        cos_sim_other_sc_in[count] = np.array(cos_sim_other_sc_h_in)
        cos_sim_diff[count] = np.array(cos_sim_diff_h)
        cos_sim_diff_in[count] = np.array(cos_sim_diff_h_in)

        print(f'Batch {count} of {len(dataloader)}')
        count += 1

    return cos_sim_target, cos_sim_other_fix, cos_sim_other_sc, cos_sim_diff, cos_sim_current_fix, rank_target, lstm_reps, scene_reps, llm_reps, actv_reps, fix_coords_all, next_fix_rel_coords_all, img_n_all, trace_n_all, n_samples_batch, output_reps, cos_sim_target_in, cos_sim_other_fix_in, cos_sim_other_sc_in, cos_sim_diff_in, cos_sim_current_fix_in

### Extracting the RN50 embeddings

def get_RN50_embeddings(images, bbv=1):

    # Load model
    preprocess = transforms.Compose([
        transforms.Resize(224, antialias=True), # resize to 224 as expected by IMAGENET1K_V1/2
        transforms.ConvertImageDtype(torch.float), # Convert image to float
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if bbv == 1:
        weights=ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
    elif bbv == 3:
        model = torch.hub.load(
            'facebookresearch/barlowtwins:main',  # repo@branch
            'resnet50',                           # entry-point in hubconf.py
            pretrained=True,                      # pulls the 1 000-epoch weights
            verbose=False
            )
    elif bbv == 4:
        model = resnet50()
        model.fc = torch.nn.Linear(2048, 565) # DVD-B has 565-dim output (ecoset)
        model_path = 'dvd-b-565.pth'
        state_dict = torch.load(model_path, map_location='cpu')['state_dict']
        state_dict = {k.replace('module._orig_mod.', ''): v for k, v in state_dict.items()}  # remove 'module._orig_mod.' prefix
        model.load_state_dict(state_dict)
        preprocess = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.ConvertImageDtype(torch.float)
        ])
    elif bbv == 5: # not rn50 actually - dinov2b
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    elif bbv == 6: # SimCLR
        model = resnet50(weights=None)
        state_dict_h = torch.load('ResNet50 1x.pth', map_location='cpu') # your path to the simclr weights
        model.load_state_dict(state_dict_h['state_dict'])
        preprocess = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.ConvertImageDtype(torch.float),
        ])

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if bbv == 5:
        device = 'cpu'  # DINOv2 models are not supported on MPS
    # print('Device:',device)
    model.to(device)
    if bbv != 5:
        activation = {} # remember to empty the list after each forward pass
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        model.avgpool.register_forward_hook(get_activation('avgpool'))
    model.eval()

    # Extract the avgpool layer features
    images = preprocess(torch.from_numpy(images).permute(0, 3, 1, 2)).to(device) 
    with torch.no_grad():
        if bbv != 5:
            _ = model(images)
            embeddings = activation['avgpool'].detach().cpu().numpy().squeeze()
        else:
            embeddings = model(images).detach().cpu().numpy().squeeze()

    return embeddings

### Resize and center crop the image

def resize_and_center_crop(img, target_size=256):
    # Step 1: Resize so that the shorter side becomes target_size (256)
    w, h = img.size
    if w < h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Step 2: Center crop to target_size × target_size
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    img = img.crop((left, top, right, bottom))

    return img

### Cosine similarity function

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

### Analysis helper functions for NSD

def load_pickle(filepath):
    """Load a pickled file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def compute_noise_ceiling(neural_rdms, metric='regression', standardize=True, positive=True):
    """
    Compute noise ceiling for a set of neural RDMs.
    For each subject, the correlation is computed against the average of all other subjects.
    """
    ceilings = []
    n = len(neural_rdms)
    for i in range(n):
        # Compute the average RDM from all other subjects
        others = np.delete(neural_rdms, i, axis=0).mean(axis=0)
        if standardize:
            subj_rdm = zscore(neural_rdms[i])
            others = zscore(others)
        else:
            subj_rdm = neural_rdms[i]
        # Compute correlation
        if metric == 'spearman' or metric == 'pearson':
            corr = spearmanr(subj_rdm, others).statistic if metric == 'spearman' else \
                pearsonr(subj_rdm, others).statistic
        elif metric == 'regression':
            corr = LinearRegression(positive=positive).fit(others.reshape(-1, 1), subj_rdm.reshape(-1, 1)).score(others.reshape(-1, 1), subj_rdm.reshape(-1, 1))
        ceilings.append(corr)
    return np.mean(ceilings)

def compute_gpn_correlations(neural_rdms, net, layers, n_glimpse=6, measure='regression', standardize=True, positive=True):
    """
    Compute Spearman correlations between each subject's neural RDM
    and each "glimpse" output from a given network.
    
    Parameters:
        neural_rdms: array of neural RDMs (n_subjects x RDM_vector)
        net: dictionary with network outputs (e.g., gp_net, gs_net)
        layers: list of layer names to extract correlations from
        n_glimpse: number of glimpse outputs (assumed along axis 0 in net[layer][0])
        measure: 'spearman' or 'pearson' or 'regression'
        standardize: whether to z-score the RDMs
        positive: if True, use positive coefficients in regression
        
    Returns:
        Array of shape (n_subjects, n_glimpse, len(layers)) with correlations.
    """
    n_subjects = len(neural_rdms)
    corr_array = np.zeros((n_subjects, n_glimpse, len(layers)))
    for subj_idx in range(n_subjects):
        for layer_idx, layer in enumerate(layers):
            for gli in range(n_glimpse):
                # print(f'Computing correlation for subject {subj_idx+1}/{n_subjects}, layer {layer} ({layer_idx+1}/{len(layers)}), glimpse {gli+1}/{n_glimpse}')
                if standardize:
                    neural_rdm_h = zscore(neural_rdms[subj_idx])
                    net_rdm_h = zscore(net[layer][gli])
                else:
                    neural_rdm_h = neural_rdms[subj_idx]
                    net_rdm_h = net[layer][gli]
                if measure == 'spearman':
                    corr = spearmanr(neural_rdm_h, net_rdm_h).statistic
                elif measure == 'pearson':
                    corr = pearsonr(neural_rdm_h, net_rdm_h).statistic
                elif measure == 'regression':
                    corr = LinearRegression(positive=positive).fit(net_rdm_h.reshape(-1, 1), neural_rdm_h.reshape(-1, 1)).score(net_rdm_h.reshape(-1, 1), neural_rdm_h.reshape(-1, 1))
                corr_array[subj_idx, gli, layer_idx] = corr
    return corr_array

def compute_network_correlations_all(neural_rdms, net_names, metric='regression', standardize=True, positive=True, verbose=False):

    net_count = 0
    for rdm_file in net_names:
        with open(rdm_file, 'rb') as f:
            rdms = pickle.load(f)
        net_count += len(rdms)

    n_subjects = len(neural_rdms)
    corr_array = np.zeros((n_subjects, net_count))
    network_names = []
    network_layer_names = []
    
    net_count = 0
    for rdm_file in net_names:
        with open(rdm_file, 'rb') as f:
            rdms = pickle.load(f)
        network_name = rdm_file.split('/')[-1].split('_test')[0]
        for name in rdms:
            if rdms[name] is None:
                if verbose:
                    print(f"Warning: RDM for {network_name} - {name} is None. Skipping.")
                continue
            if rdms[name].std() < 1e-8:
                if verbose:
                    print(f"Warning: RDM for {network_name} - {name} has zero variance. Skipping.")
                continue
            if standardize:
                net_rdm_h = zscore(rdms[name])
            else:
                net_rdm_h = rdms[name]
            for subj_idx in range(n_subjects):
                if rdms[name].shape != neural_rdms[subj_idx].shape:
                    if verbose:
                        print(f"Warning: Shape mismatch for {network_name} - {name}: network RDM shape {rdms[name].shape}, neural RDM shape {neural_rdms[subj_idx].shape}")
                    continue
                if standardize:
                    neural_rdm_h = zscore(neural_rdms[subj_idx])
                else:
                    neural_rdm_h = neural_rdms[subj_idx]
                if metric != 'regression':
                    corr_array[subj_idx, net_count] = spearmanr(net_rdm_h, neural_rdm_h).statistic if metric == 'spearman' else pearsonr(net_rdm_h, neural_rdm_h).statistic
                else:
                    corr_array[subj_idx, net_count] = LinearRegression(positive=positive).fit(net_rdm_h.reshape(-1, 1), neural_rdm_h.reshape(-1, 1)).score(net_rdm_h.reshape(-1, 1), neural_rdm_h.reshape(-1, 1))
            network_names.append(network_name)
            network_layer_names.append(name)
            net_count += 1
    return corr_array, network_names, network_layer_names

def compute_network_correlations(neural_rdms, network_rdms, metric='regression', standardize=True, positive=True):
    """
    Compute Spearman correlations between each subject's neural RDM and each network RDM.
    
    Parameters:
        neural_rdms: array of neural RDMs (n_subjects x RDM_vector)
        network_rdms: dict mapping network name to its RDM; can be 2-level
        metric: 'spearman' or 'pearson' or 'regression'
        standardize: whether to z-score the RDMs
        positive: if True, use positive coefficients in regression
        
    Returns:
        Tuple of (corr_array, network_names) where:
          - corr_array has shape (n_subjects, n_networks)
          - network_names is a list of network names (keys of network_rdms)
    """
    n_subjects = len(neural_rdms)
    n_networks = np.sum([len(network_rdms[key]) for key in network_rdms])
    corr_array = np.zeros((n_subjects, n_networks))
    network_names = []
    for subj_idx in range(n_subjects):
        net_count = 0
        for key in network_rdms.keys():
            for name in network_rdms[key].keys():
                if standardize:
                    neural_rdm_h = zscore(neural_rdms[subj_idx])
                    net_rdm_h = zscore(network_rdms[key][name])
                else:
                    neural_rdm_h = neural_rdms[subj_idx]
                    net_rdm_h = network_rdms[key][name]
                if metric != 'regression':
                    corr_array[subj_idx, net_count] = spearmanr(net_rdm_h, neural_rdm_h).statistic if metric == 'spearman' else pearsonr(net_rdm_h, neural_rdm_h).statistic
                else:
                    corr_array[subj_idx, net_count] = LinearRegression(positive=positive).fit(net_rdm_h.reshape(-1, 1), neural_rdm_h.reshape(-1, 1)).score(net_rdm_h.reshape(-1, 1), neural_rdm_h.reshape(-1, 1))
                net_count += 1
                if subj_idx == 0:
                    network_names.append(f"{name}")
    return corr_array, network_names

# Extracting the RDMs for all layers of GPN
def extract_GPN_RDMs(dataloader, net, hyp, distance_metric='correlation', averaging=False, trace_return=0, return_actvs=False):

    n_samples_batch = [0.0 for _ in range(len(dataloader))]

    gpn_actvs = {}
    semantic_embeds = []
    img_n_all = []
    trace_n_all = []

    count = 0
    
    for actvs,next_fix_rel_coords,fix_coords,semantic_embed,scene_embed,scene_multihot,img_n,trace_n in dataloader:

        n_samples_batch[count] = actvs.shape[0]

        actvs = actvs.to(hyp['optimizer']['device'])
        next_fix_rel_coords = next_fix_rel_coords.to(hyp['optimizer']['device'])

        fix_coords = fix_coords.to(hyp['optimizer']['device']) # these are next fixations' coords relative to the central fixation, computed on 256px images by deepgaze3! first fixation is [0,0]
        semantic_embed = semantic_embed.to(hyp['optimizer']['device'])
        scene_embed = scene_embed.to(hyp['optimizer']['device'])
        scene_multihot = scene_multihot.to(hyp['optimizer']['device'])
        
        activations, _ = net(actvs[:,:-1,:],hyp['optimizer']['losses']['provide_loc']*next_fix_rel_coords) # only feed actvs[:,:-1,:] as input as the last fixation isn't used as an input

        for layer in activations.keys():
            if layer not in gpn_actvs:
                gpn_actvs[layer] = []
            gpn_actvs[layer].append(activations[layer].detach().cpu().numpy())
        img_n_all.append(img_n.numpy())
        trace_n_all.append(trace_n.numpy())
        semantic_embeds.append(semantic_embed.detach().cpu().numpy())

        print(f'Batch {count+1} of {len(dataloader)}')
        count += 1
    
    gpn_rdms = {}
    trace_n_all = np.vstack(trace_n_all)
    for layer in gpn_actvs.keys():
        gpn_actvs[layer] = np.vstack(gpn_actvs[layer])
        if not averaging:
            gpn_actvs_use = gpn_actvs[layer][trace_n_all[:,0] == trace_return, :, :]
            gpn_rdms[layer] = np.vstack([pdist(gpn_actvs_use[:,t,:], metric=distance_metric) for t in range(gpn_actvs_use.shape[1])])
        else:
            gpn_rdms[layer] = []
            for t in range(gpn_actvs[layer].shape[1]):
                gpn_actvs_use = gpn_actvs[layer][trace_n_all[:,0] == 0,t,:]
                for trace_use in range(1,int(np.max(trace_n_all))+1):
                    gpn_actvs_use += gpn_actvs[layer][trace_n_all[:,0] == trace_use, t, :]
                gpn_actvs_use /= (np.max(trace_n_all)+1)
                gpn_rdms[layer].append(pdist(gpn_actvs_use, metric=distance_metric))
            gpn_rdms[layer] = np.vstack(gpn_rdms[layer])
        print(f'RDM extraction for layer {layer} done! Shape: {gpn_rdms[layer].shape}')

    gpn_rdms['RN50_gl_seqadd'] = []
    for t in range(gpn_actvs_use.shape[1]):
        if not averaging:
            rn50_actvs_use = gpn_actvs['rn50_glimpse'][trace_n_all[:,0] == trace_return, :t+1, :].sum(axis=1)
            gpn_rdms['RN50_gl_seqadd'].append(pdist(rn50_actvs_use, metric=distance_metric))
        else:
            rn50_actvs_use = gpn_actvs['rn50_glimpse'][trace_n_all[:,0] == 0, :t+1, :].sum(axis=1)
            for trace_use in range(1,int(np.max(trace_n_all))+1):
                rn50_actvs_use += gpn_actvs['rn50_glimpse'][trace_n_all[:,0] == trace_use, :t+1, :].sum(axis=1)
            rn50_actvs_use /= (np.max(trace_n_all)+1)
            gpn_rdms['RN50_gl_seqadd'].append(pdist(rn50_actvs_use, metric=distance_metric))
    gpn_rdms['RN50_gl_seqadd'] = np.vstack(gpn_rdms['RN50_gl_seqadd'])
    print(f'RDM extraction for RN50_gl_seqadd done! Shape: {gpn_rdms["RN50_gl_seqadd"].shape}')

    semantic_embeds = np.vstack(semantic_embeds)
    gpn_rdms['semantic_embed'] = pdist(semantic_embeds[trace_n_all[:,0] == 0, :], metric=distance_metric)
    print(f'RDM extraction for semantic_embed done! Shape: {gpn_rdms["semantic_embed"].shape}')

    if return_actvs:
        return gpn_rdms, gpn_actvs, trace_n_all
    else:
        return gpn_rdms
    
def extract_GPN_RDMs_e2e(dataloader, net, hyp, distance_metric='correlation', averaging=False, trace_return=0, return_actvs=False): 

    gpn_actvs = {}
    img_n_all = []
    trace_n_all = []

    count = 0
    
    for glimpse_seq, next_fix_rel_coords, _, img_n, trace_n in dataloader:

        activations, _, _ = net(glimpse_seq, hyp['optimizer']['losses']['provide_loc']*next_fix_rel_coords)

        for layer in activations.keys():
            if layer not in gpn_actvs:
                gpn_actvs[layer] = []
            gpn_actvs[layer].append(activations[layer].detach().cpu().numpy())
        img_n_all.append(img_n.numpy())
        trace_n_all.append(trace_n.numpy())

        print(f'Batch {count+1} of {len(dataloader)}')
        count += 1
    
    gpn_rdms = {}
    trace_n_all = np.vstack(trace_n_all)
    for layer in gpn_actvs.keys():
        gpn_actvs[layer] = np.vstack(gpn_actvs[layer])
        if not averaging:
            gpn_actvs_use = gpn_actvs[layer][trace_n_all[:,0] == trace_return, :, :]
            gpn_rdms[layer] = np.vstack([pdist(gpn_actvs_use[:,t,:], metric=distance_metric) for t in range(gpn_actvs_use.shape[1])])
        else:
            gpn_rdms[layer] = []
            for t in range(gpn_actvs[layer].shape[1]):
                gpn_actvs_use = gpn_actvs[layer][trace_n_all[:,0] == 0,t,:]
                for trace_use in range(1,int(np.max(trace_n_all))+1):
                    gpn_actvs_use += gpn_actvs[layer][trace_n_all[:,0] == trace_use, t, :]
                gpn_actvs_use /= (np.max(trace_n_all)+1)
                gpn_rdms[layer].append(pdist(gpn_actvs_use, metric=distance_metric))
            gpn_rdms[layer] = np.vstack(gpn_rdms[layer])
        print(f'RDM extraction for layer {layer} done! Shape: {gpn_rdms[layer].shape}')

    gpn_rdms['RN50_gl_seqadd'] = []
    for t in range(gpn_actvs_use.shape[1]):
        if not averaging:
            rn50_actvs_use = gpn_actvs['rn50_glimpse'][trace_n_all[:,0] == trace_return, :t+1, :].sum(axis=1)
            gpn_rdms['RN50_gl_seqadd'].append(pdist(rn50_actvs_use, metric=distance_metric))
        else:
            rn50_actvs_use = gpn_actvs['rn50_glimpse'][trace_n_all[:,0] == 0, :t+1, :].sum(axis=1)
            for trace_use in range(1,int(np.max(trace_n_all))+1):
                rn50_actvs_use += gpn_actvs['rn50_glimpse'][trace_n_all[:,0] == trace_use, :t+1, :].sum(axis=1)
            rn50_actvs_use /= (np.max(trace_n_all)+1)
            gpn_rdms['RN50_gl_seqadd'].append(pdist(rn50_actvs_use, metric=distance_metric))
    gpn_rdms['RN50_gl_seqadd'] = np.vstack(gpn_rdms['RN50_gl_seqadd'])
    print(f'RDM extraction for RN50_gl_seqadd done! Shape: {gpn_rdms["RN50_gl_seqadd"].shape}')

    if return_actvs:
        return gpn_rdms, gpn_actvs, trace_n_all
    else:
        return gpn_rdms

# 2D matrix linear regression loop
def linear_regression_general(X, y, fit_intercept=True, rcond=None):
    """
    Parameters
    ----------
    X : ndarray, shape (B, T, D1)
        Predictor tensor.
    y : ndarray, shape (B, D2)
        Targets.
    fit_intercept : bool, default True
        If True, a column of ones is prepended so each output gets an intercept.
    rcond : float or None
        Cut-off for small singular values in `np.linalg.lstsq`.

    Returns
    -------
    betas : ndarray, shape (T*D1 + fit_intercept, D2)
        Coefficient matrix.  Row 0 holds the intercepts if requested.
    """
    B, T, D1 = X.shape
    B_y, D2 = y.shape
    if B_y != B:
        raise ValueError("X and y must have the same first dimension B")

    # ---- build 2-D design matrix -------------------------------------------
    X_flat = X.reshape(B, T * D1)                 # (B, T*D1)
    if fit_intercept:
        X_design = np.hstack([np.ones((B, 1)), X_flat])   # (B, 1+T*D1)
    else:
        X_design = X_flat                         # (B, T*D1)
    Tp = X_design.shape[1]

    # ---- solve in one shot -------------------------------------------------
    # np.linalg.lstsq happily takes (B, Tp) by (B, D2) and returns (Tp, D2).
    betas, *_ = np.linalg.lstsq(X_design, y, rcond=rcond)
    return betas

def predict_linear_regression_general(X, betas, fit_intercept=True):
    B, T, D1 = X.shape
    X_flat = X.reshape(B, T * D1)
    if fit_intercept:
        return betas[0] + X_flat @ betas[1:]
    else:
        return X_flat @ betas

# Variance Partitioning Analysis

def variance_partitioning_analysis(
    neural_rdm,
    A_rdm,
    B_rdm,
    *,
    neural_name="Observed RDM",
    A_name="Model 1 RDM",
    B_name="Model 2 RDM",
    positive=True,          # keep or flip the NNLS constraint
    random_state=42,        # for reproducible jitter only
    output_data=False  # if True, return the data for further processing and not the plot
):
    """
    Variance–partitioning (commonality) analysis with non-negative fits.

    Parameters
    ----------
    neural_rdm : array (n_subjects, n_pairs)
        One RDM per subject, already vectorised.
    A_rdm, B_rdm : array (n_pairs,)
        Model RDMs (vectorised).
    positive : bool, default True
        If True, slopes are constrained to be ≥ 0 (NNLS).
    random_state : int, default 42
        Controls the point-jitter in the plot.
    """
    n_subjects = neural_rdm.shape[0]

    # ------------------------------------------------------------------
    # Helper – R² from (optionally) non-negative linear regression
    # ------------------------------------------------------------------
    def _r2_lr(X, y):
        """
        Fits LinearRegression to X, y and returns R².
        X is 1-D or 2-D; y is 1-D.
        """
        # standardise vectors first
        X = zscore(X, axis=0)
        y = zscore(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        model = LinearRegression(fit_intercept=True, positive=positive)
        model.fit(X, y)
        return model.score(X, y)           # standard R²

    # Build once, reuse
    X_full = np.column_stack((A_rdm, B_rdm))
    X_A    = A_rdm
    X_B    = B_rdm

    unique_A_all, unique_B_all, shared_all, total_explained_all = [], [], [], []

    # ------------------------------------------------------------------
    # Main loop over subjects
    # ------------------------------------------------------------------
    for subj_idx in range(n_subjects):
        y = neural_rdm[subj_idx]

        R2_full = _r2_lr(X_full, y)
        R2_A    = _r2_lr(X_A,    y)
        R2_B    = _r2_lr(X_B,    y)

        unique_A = max(R2_full - R2_B, 0)
        unique_B = max(R2_full - R2_A, 0)
        shared   = max(R2_A + R2_B - R2_full, 0)

        unique_A_all.append(unique_A)
        unique_B_all.append(unique_B)
        shared_all.append(shared)
        total_explained_all.append(R2_full)

    unique_A_all        = np.asarray(unique_A_all)
    unique_B_all        = np.asarray(unique_B_all)
    shared_all          = np.asarray(shared_all)
    total_explained_all = np.asarray(total_explained_all)

    # ------------------------------------------------------------------
    # Visualisation — identical to your original
    # ------------------------------------------------------------------
    data   = [unique_A_all, unique_B_all, shared_all, total_explained_all]
    labels = [f"Unique {A_name}", f"Unique {B_name}", "Shared", "Total explained"]

    if output_data:
        return data, labels
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        parts   = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=False)

        for pc in parts["bodies"]:
            pc.set_facecolor("lightgray")
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

        rng            = np.random.default_rng(random_state)
        subject_jitter = rng.normal(0, 0.04, n_subjects)   # same jitter across measures

        for subj in range(n_subjects):
            xs = [i + 1 + subject_jitter[subj] for i in range(len(data))]
            ys = [d[subj] for d in data]
            ax.plot(xs, ys, color="gray", alpha=0.4, linewidth=0.5)

        for i, d in enumerate(data):
            xs = 1 + i + subject_jitter
            ax.scatter(xs, d, color="k", s=20, alpha=0.8)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Explained Variance (R²)")
        ax.set_title(f"Variance Partitioning in {neural_name}\n({A_name} & {B_name})")
        ax.set_ylim(0, 0.25)
        plt.tight_layout()
        plt.show()


# Permutation paired t-test

def perm_test_scegram(
    D1_act: np.ndarray,
    D1_base: np.ndarray,
    D2_act: np.ndarray,
    D2_base: np.ndarray,
    *,
    n_perm: int = 10000,
    rng: Union[int, np.random.Generator, None] = 42,
    alpha: float = 0.05,
    test_output: str = 'hierarchical'
) -> Dict[str, Dict[str, float]]:
    """
    Hierarchical sign-permutation test for a 2×2 within-subjects design.

    Multiple-comparisons control:
      • Level 2: Holm–Bonferroni across {Δ1_change, Δ2_change} (if interaction is significant).
      • Level 3 (pooled branch): Holm across {pooled_act, pooled_base} (if pooled_change is significant).
      • Level 3 (per-Δ branch): 
          - If only one Δ is open → Holm across its pair {Δk_act, Δk_base}.
          - If BOTH Δ1 and Δ2 are open → Holm across the UNION of all four
            {Δ1_act, Δ1_base, Δ2_act, Δ2_base}.

    Gatekeeping:
      Level 1 interaction @ α → if not significant, test pooled_change; if significant, test Δ1 & Δ2.
      Level 3 follow-ups open only if their parent test passed (unless test_output='flat').

    test_output:
      'hierarchical' → follow gatekeeping and only open lower levels if gates pass.
      'flat'         → return all contrasts; Level-3 per-Δ follow-ups use union-of-4 adjustment.

    Returns nested dict with keys 'level1', 'level2', 'level3'. Each entry has:
      • obs   : observed mean statistic
      • p_raw : permutation p-value (two-sided via sign flips)
      • p_adj : adjusted p within its family (or equals p_raw for single tests)
    """
    # ——— Input checks —————————————————————————————————————————————
    D1a, D1b, D2a, D2b = map(np.asarray, (D1_act, D1_base, D2_act, D2_base))
    if not (D1a.ndim == D1b.ndim == D2a.ndim == D2b.ndim == 1):
        raise ValueError("All inputs must be 1D arrays of equal length")
    if not (D1a.shape == D1b.shape == D2a.shape == D2b.shape):
        raise ValueError("All inputs must have the same length")
    n = D1a.shape[0]

    # ——— Observed contrasts ——————————————————————————————————————
    # Level 1 interaction
    c1 = (D1a - D1b) - (D2a - D2b)
    # Level 2
    c2_pool = 0.5 * ((D1a - D1b) + (D2a - D2b))
    c2_1    = D1a - D1b
    c2_2    = D2a - D2b
    # Level 3 pooled follow-ups
    c3p_act  = 0.5 * (D1a + D2a)
    c3p_base = 0.5 * (D1b + D2b)
    # Level 3 simple follow-ups per condition
    c3_1_act,  c3_1_base = D1a, D1b
    c3_2_act,  c3_2_base = D2a, D2b

    obs = np.array([
        c1.mean(axis=0),          # 0
        c2_pool.mean(axis=0),     # 1
        c2_1.mean(axis=0),        # 2
        c2_2.mean(axis=0),        # 3
        c3p_act.mean(axis=0),     # 4
        c3p_base.mean(axis=0),    # 5
        c3_1_act.mean(axis=0),    # 6
        c3_1_base.mean(axis=0),   # 7
        c3_2_act.mean(axis=0),    # 8
        c3_2_base.mean(axis=0),   # 9
    ])

    # ——— Permutations —————————————————————————————————————————————
    rng = np.random.default_rng(rng)
    signs = rng.choice([-1, 1], size=(n_perm, n))  # each row is one sign-flip vector

    # stack all 10 contrasts for vectorized flipping: shape (10, n)
    data = np.stack([
        c1, c2_pool, c2_1, c2_2,
        c3p_act, c3p_base,
        c3_1_act, c3_1_base,
        c3_2_act, c3_2_base
    ], axis=0)

    # apply sign flips: (n_perm, 10, n) → means: (n_perm, 10)
    perm = (signs[:, None, :] * data[None, :, :]).mean(axis=2)

    # ——— Unadjusted p-values (two-sided) ————————————————————————
    p_raw = np.mean(np.abs(perm) >= np.abs(obs)[None, :], axis=0)
    # (Optional finite-sample tweak: use (count+1)/(n_perm+1).)

    # ——— Helper: Holm step-down (returns adjusted p in the same order) ———
    def holm_stepdown(pvals):
        p = np.asarray(pvals, float)
        m = p.size
        if m <= 1:
            return p.copy()
        order = np.argsort(p)              # ascending
        p_sorted = p[order]
        adj_sorted = (m - np.arange(m)) * p_sorted
        # Holm is step-DOWN: cumulative max in the FORWARD direction
        adj_sorted = np.maximum.accumulate(adj_sorted)
        adj_sorted = np.clip(adj_sorted, 0, 1)
        out = np.empty_like(adj_sorted)
        out[order] = adj_sorted
        return out

    # ——— Assemble results with gatekeeping ————————————————————————
    results = {
        'level1':  {'obs': obs[0], 'p_raw': p_raw[0], 'p_adj': p_raw[0]},
        'level2':  {},
        'level3':  {}
    }

    # LEVEL 2
    # Branch A: interaction NOT significant → test pooled_change (single test)
    if results['level1']['p_adj'] >= alpha or test_output == 'flat':
        results['level2']['pooled_change'] = {
            'obs':   obs[1],
            'p_raw': p_raw[1],
            'p_adj': p_raw[1],  # single test
        }

    # Branch B: interaction significant → test Δ1 and Δ2 with Holm (family of 2)
    if results['level1']['p_adj'] < alpha or test_output == 'flat':
        p_raw_L2 = np.array([p_raw[2], p_raw[3]])
        p_adj_L2 = holm_stepdown(p_raw_L2)
        results['level2']['Δ1_change'] = {'obs': obs[2], 'p_raw': p_raw[2], 'p_adj': p_adj_L2[0]}
        results['level2']['Δ2_change'] = {'obs': obs[3], 'p_raw': p_raw[3], 'p_adj': p_adj_L2[1]}

    # LEVEL 3
    # 3a) pooled branch (if tested and significant OR flat): Holm across {pooled_act, pooled_base}
    if ('pooled_change' in results['level2']) and \
       (results['level2']['pooled_change']['p_adj'] < alpha or test_output == 'flat'):
        p_raw_pair = np.array([p_raw[4], p_raw[5]])  # pooled_act, pooled_base
        p_adj_pair = holm_stepdown(p_raw_pair)
        results['level3']['pooled_act']  = {'obs': obs[4], 'p_raw': p_raw[4], 'p_adj': p_adj_pair[0]}
        results['level3']['pooled_base'] = {'obs': obs[5], 'p_raw': p_raw[5], 'p_adj': p_adj_pair[1]}

    # 3b) per-Δ branch:
    # Collect open Δ-pairs; if both open → Holm across the union of 4; else Holm within the single pair.
    open_pairs = []
    if ('Δ1_change' in results['level2']) and (results['level2']['Δ1_change']['p_adj'] < alpha or test_output == 'flat'):
        open_pairs += [('Δ1_change', 6)]  # indices 6,7
    if ('Δ2_change' in results['level2']) and (results['level2']['Δ2_change']['p_adj'] < alpha or test_output == 'flat'):
        open_pairs += [('Δ2_change', 8)]  # indices 8,9

    if open_pairs:
        idxs, labels = [], []
        for lab, start in open_pairs:
            idxs.extend([start, start + 1])
            labels.extend([f'{lab}_act', f'{lab}_base'])

        p_adj_union = holm_stepdown(np.array([p_raw[i] for i in idxs]))
        for i, name in enumerate(labels):
            j = idxs[i]
            results['level3'][name] = {'obs': obs[j], 'p_raw': p_raw[j], 'p_adj': p_adj_union[i]}

    print("Results of the hierarchical sign-permutation test (Holm-adjusted; union-of-4 at Level-3 if both Δ's open):")
    pprint.pprint(results)
    return results

def get_fixation_history(fixation_coordinates, model):
    """
    Given a Python list `fixation_coordinates` of (x,y) tuples,
    returns a list of length `len(model.included_fixations)` where
    positions beyond the available history are filled with nan.
    """
    history = []
    for idx in model.included_fixations:
        try:
            history.append(fixation_coordinates[idx])
        except IndexError:
            # history.append(np.nan) # this is how the original code did it
            history.append(-1) # to align with Varun's code; works well
    return history

class OnTheFlyExtractor(nn.Module):
    """
    Given a batch of raw scene images (uint8, 256×256x3), produces:
      • fix_actvs   : float16 tensor [B, T, 2048]
      • next_abs    : int16  tensor [B, T-1, 2]
      • next_rel    : int16  tensor [B, T-1, 2]
      • scene_embed : float16 tensor [B, 2048]
    All heavy conv passes (DeepGazeIII & ResNet-50) are micro-batched
    and run under no_grad(), then their activations are freed to save GPU RAM.
    """

    def __init__(
        self,
        dataset: str = "NSD",           # "NSD" or "AVS"
        bbv: int = 1,                  # ResNet50 version 0/1/2
        n_fixs: int = 7,                # number of fixations to sample
        device: str = "mps",           # where GPN will live
        dg3_device: str = "cpu",       # where DG3 will run
        centerbias_path: str = "centerbias_mit1003.npy",
        seed: int = 42,
        rn50_mb: int = 100,              # micro-batch size for ResNet50
        dg3_mb: int = 50,                # micro-batch size for DeepGazeIII - 90 max on 1 L40S
        gaze_type: str = "dg3",      # "dg3","random","dg3p" - currently defaulting to dg3 only - to-do other cases
        dg3_res: int = 256,         # resolution of DG3 - 1024/256
    ):
        super().__init__()
        self.device   = device
        self.dg3_dev  = dg3_device
        self.n_fixs   = n_fixs
        self.im_size  = 256
        self.glimpse_ext = 91 if dataset == "NSD" else 36
        self.pad      = self.glimpse_ext // 2
        self.rn50_mb  = rn50_mb
        self.dg3_mb   = dg3_mb
        self.dg3_res = dg3_res

        # ─── DeepGazeIII setup (FP32) ─────────────────────────────────────
        self.dg3 = deepgaze_pytorch.DeepGazeIII(pretrained=True) \
                      .float().to(self.dg3_dev).eval()
        cb = np.load(centerbias_path)                        # [1024×1024]
        cb -=  logsumexp(cb)                              # normalize log‐density
        # rescale to dg3_res
        cb = zoom(cb, (dg3_res/cb.shape[0], dg3_res/cb.shape[0]), order=0, mode='nearest') # [dg3_res×dg3_res] - as done in DG3 example
        # will broadcast inside DG3
        self.cb = torch.tensor([cb]).float().to(self.dg3_dev) # [1,dg3_res,dg3_res]
        self._rng = np.random.RandomState(seed)
        self.dg3_prep = transforms.Resize(dg3_res, antialias=True)

        # ─── ResNet-50 setup (FP16) ────────────────────────────────────────
        weights = {0: None,
                   1: ResNet50_Weights.IMAGENET1K_V1,
                   2: ResNet50_Weights.IMAGENET1K_V2}[bbv]
        self.net = resnet50(weights=weights)
        self._acts = {} # remember to empty the list after each forward pass
        def get_activation(name):
            def hook(model, input, output):
                self._acts[name] = output.detach()
            return hook
        self.net.avgpool.register_forward_hook(get_activation('avgpool'))
        self.net = self.net.float().to(device).eval()
        self.prep = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])

    def forward(self, scenes_uint8: torch.Tensor):
        """
        Input:
          scenes_uint8: uint8 array [B,256,256,3], values in [0,255]
        Returns:
          fix_actvs   : float16 [B, T, 2048]
          next_abs    : int16   [B, T-1, 2]
          next_rel    : int16   [B, T-1, 2]
          scene_embed : float16 [B, 2048]
        """
        B = scenes_uint8.shape[0]

        ########################################
        # 1) Full‐scene embedding via ResNet50 #
        ########################################

        scene_float = torch.from_numpy(scenes_uint8).permute(0, 3, 1, 2)
        embeds = []

        for i in range(0, B, self.rn50_mb):
            chunk = scene_float[i : i + self.rn50_mb]
            inp = self.prep(chunk).float().to(self.device)   
            with torch.no_grad():
                _ = self.net(inp)
                h = self._acts.pop("avgpool")                # [b,2048,1,1]
            embeds.append(h.squeeze(-1).squeeze(-1))
            # free everything except model params
        del chunk, inp
        torch.cuda.empty_cache()
        scene_embed = torch.cat(embeds, 0).cpu().numpy()            # [B,2048]
        # print(scene_embed.shape)

        ##############################################
        # 2) DG3 sampling of fixations (FP32, no_grad) #
        ##############################################
        
        # upsample to dg3_res in chunks
        scenes_f32 = torch.tensor(scenes_uint8.transpose(0, 3, 1, 2))
        coords_all = []

        for i in range(0, B, self.dg3_mb):
            sub = scenes_f32[i : i + self.dg3_mb]
            up = self.dg3_prep(sub).float().to(self.dg3_dev)  # [b,3,dg3_res,dg3_res]
            # init histories
            histories = [[[self.dg3_res//2],[self.dg3_res//2]] for _ in range(sub.size(0))]
            # sample T−1 more fixations
            for _step in range(self.n_fixs - 1):
                # build history tensors
                x_list, y_list = [], []
                for hist in histories:
                    hx = get_fixation_history(hist[0], self.dg3)
                    hy = get_fixation_history(hist[1], self.dg3)
                    x_list.append(hx); y_list.append(hy)
                x_hist = torch.tensor(x_list, device=self.dg3_dev)
                y_hist = torch.tensor(y_list, device=self.dg3_dev)
                with torch.no_grad():
                    logd = self.dg3(up, self.cb, x_hist, y_hist)   # [b,1,dg3_res,dg3_res]
                # print(logd.shape)
                logd_np = logd.cpu().numpy().squeeze(1)         # [b,dg3_res,dg3_res]
                # print(logd_np.shape)
                # sample one next fixation per item
                for bi in range(len(histories)):
                    # print(logd_np[bi].shape)
                    xn, yn = sample_from_logdensity(logd_np[bi], rst=self._rng)
                    histories[bi][0].append(float(xn))
                    histories[bi][1].append(float(yn))

            coords_all.extend(histories)

            print(f"DG3 Processed {i + self.dg3_mb} / {B} scenes")

        # free
        del x_hist, y_hist, logd, logd_np
        torch.cuda.empty_cache()
        del sub, up, histories
        torch.cuda.empty_cache()
        
        # print(np.array(coords_all).shape)
        coords = np.array(coords_all).transpose((0,2,1))  # [B, T, 2]

        ###################################################
        # 3) Build next_abs / next_rel in 256-space (CPU) #
        ###################################################
        coords256 = coords * (self.im_size / self.dg3_res)
        abs_np = coords256[:,1:,:] - coords256[:,0:1,:]
        rel_np = coords256[:,1:,:] - coords256[:,:-1,:]
        next_abs = abs_np.astype(np.int16)
        next_rel = rel_np.astype(np.int16) 

        ##############################################
        # 4) Crop glimpses + ResNet50 activations     #
        ##############################################
        # pad once
        padded = F.pad(scene_float,
                       (self.pad,)*4, mode="constant", value=0)     # [B,3,256+g,256+g]
        crops = []
        for b in range(B):
            for t in range(self.n_fixs):
                x_c, y_c = coords256[b,t]
                x0, y0 = int(x_c), int(y_c)
                crop = padded[b, :,
                              y0 : y0 + self.glimpse_ext,
                              x0 : x0 + self.glimpse_ext]
                crops.append(crop)
        # now micro-batch through RN50 again
        all_actvs = []
        crops = torch.stack(crops, 0)  # [B*T,3,E,E]
        for i in range(0, crops.size(0), self.rn50_mb):
            c = crops[i : i + self.rn50_mb]
            inp = self.prep(c).float().to(self.device)
            with torch.no_grad():
                _ = self.net(inp)
                h = self._acts.pop("avgpool")  # [b,2048,1,1]
            all_actvs.append(h.squeeze(-1).squeeze(-1))

            print(f"RN50 Processed {i + self.rn50_mb} / {crops.size(0)} crops")

        all_actvs = torch.cat(all_actvs, 0)     # [B*T,2048]
        fix_actvs = all_actvs.view(B, self.n_fixs, 2048).cpu().numpy() 
        # print(fix_actvs.shape)

        return fix_actvs, next_abs, next_rel, scene_embed

# Sign permutation test

def sign_perm_test(corrs, n_perm=10000, two_tailed=True):
    obs_stat = corrs.mean()
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        signs = np.random.choice([+1, -1], size=corrs.shape)
        perm_stats[i] = (signs * corrs).mean()
    if two_tailed:
        p_val = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))
    else:
        p_val = np.mean(perm_stats >= obs_stat)
    return obs_stat, p_val, perm_stats

# Image loader for miniplaces

def load_miniplaces_images(folder, labels_file, n_images=None, seed=42, crop=True):
    """
    Read images listed in labels_file (lines: '<relative/path> <int_label>').
    If n_images is not None, sample up to n_images per label; else include all.
    Returns:
        images: np.ndarray [N, H, W, 3] (RGB)
        labels: np.ndarray [N]          (int)
        crop:  scale image to 256px (smaller side) and take 91px center crop
    """
    rng = random.Random(seed)
    label_to_files = defaultdict(list)

    with open(labels_file, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # robust to spaces in path: split on last whitespace only
            try:
                img_rel, label_str = line.rsplit(maxsplit=1)
            except ValueError:
                raise ValueError(f"Bad line (expected 'path label'): {line!r}")
            label = int(label_str)
            img_path = os.path.join(folder, img_rel)
            label_to_files[label].append(img_path)

    if not label_to_files:
        raise ValueError("No images found from labels_file.")

    images, labels = [], []

    for label in sorted(label_to_files.keys()):
        files = label_to_files[label]
        if n_images is None:
            chosen = files
        else:
            take = min(n_images, len(files))
            if take < n_images:
                print(f"Warning: label {label} has only {take} images (< {n_images}).")
            chosen = rng.sample(files, take)

        for img_path in chosen:
            try:
                with Image.open(img_path) as im:
                    im = im.convert('RGB')
                    if crop:
                        # scale smaller side to 256px
                        w, h = im.size
                        if w < h:
                            new_w = 256
                            new_h = int(h * (256 / w))
                        else:
                            new_h = 256
                            new_w = int(w * (256 / h))
                        im = im.resize((new_w, new_h), Image.LANCZOS)
                        # center crop 91x91
                        left = (new_w - 91) // 2
                        top = (new_h - 91) // 2
                        im = im.crop((left, top, left + 91, top + 91))
                    images.append(np.array(im))
                labels.append(label)
            except FileNotFoundError:
                print(f"Warning: file not found: {img_path}")
            except Exception as e:
                print(f"Warning: could not load {img_path}: {e}")

    if not images:
        raise ValueError("After sampling, no images were loaded.")

    print(f"Loaded {len(images)} images across {len(label_to_files)} labels.")

    return np.stack(images), np.array(labels, dtype=np.int64)

def mean_and_bootstrap_ci(
    A: np.ndarray,
    n_boot: int = 10000,
    ci: float = 95.0,
    random_state: int = 42,
    batch_size: int = 100000,
):
    """
    Compute per-network mean across subjects and a two-sided bootstrap CI.

    Parameters
    ----------
    A : array-like, shape (n_subjects, n_networks)
        Rows are subjects, columns are networks. Can contain NaNs.
    n_boot : int, default=10000
        Number of bootstrap resamples (with replacement) over subjects.
    ci : float, default=95.0
        Confidence level for the percentile CI.
    random_state : int or None, default=None
        Seed for reproducibility.
    batch_size : int or None, default=None
        If set, compute bootstrap in chunks of this many resamples
        to reduce peak memory.

    Returns
    -------
    mean : np.ndarray, shape (n_networks,)
        Mean across subjects (nan-robust).
    ci_low : np.ndarray, shape (n_networks,)
        Lower bound of the bootstrap percentile CI.
    ci_high : np.ndarray, shape (n_networks,)
        Upper bound of the bootstrap percentile CI.
    err_low : np.ndarray, shape (n_networks,)
        Asymmetric lower error bar (mean - ci_low).
    err_high : np.ndarray, shape (n_networks,)
        Asymmetric upper error bar (ci_high - mean).

    Notes
    -----
    - Bootstrap resamples subjects with replacement and recomputes
      the column-wise mean for each resample using `np.nanmean`.
    - If an entire column is NaN, outputs for that column will be NaN.
    """
    X = np.asarray(A, dtype=float)
    if X.ndim != 2:
        raise ValueError("A must be 2D: shape (n_subjects, n_networks).")

    n_subj, n_net = X.shape
    rng = np.random.default_rng(random_state)

    # Per-network mean across subjects (nan-robust)
    mean = np.nanmean(X, axis=0)

    # Allocate container for bootstrap means (memory ~ n_boot * n_net)
    boot_means = np.empty((n_boot, n_net), dtype=float)

    if batch_size is None or batch_size >= n_boot:
        # Single batch (fully vectorized)
        idx = rng.integers(0, n_subj, size=(n_boot, n_subj))
        # Shape: (n_boot, n_subj, n_net) -> mean over subjects
        boot_means[:] = np.nanmean(X[idx, :], axis=1)
    else:
        # Chunked to reduce peak memory
        start = 0
        while start < n_boot:
            end = min(start + batch_size, n_boot)
            bs = end - start
            idx = rng.integers(0, n_subj, size=(bs, n_subj))
            boot_means[start:end] = np.nanmean(X[idx, :], axis=1)
            start = end

    alpha = 100.0 - ci
    ci_low = np.percentile(boot_means, alpha / 2.0, axis=0)
    ci_high = np.percentile(boot_means, 100.0 - alpha / 2.0, axis=0)

    err_low = mean - ci_low
    err_high = ci_high - mean
    return mean, ci_low, ci_high, err_low, err_high

#--------------------------------------------------

class _FriedmanWilcoxon2x2Result:
    def __init__(self, fr_stat, fr_p, n, kendall_w, rows):
        self.fr_stat = fr_stat
        self.fr_p = fr_p
        self.n = n
        self.kendall_w = kendall_w
        self.rows = rows  # list of dicts for contrasts (label, n_eff, Wplus, p, p_holm, r_rb)

    def __str__(self):
        out = []
        out.append("Friedman–Wilcoxon (nonparametric 2×2 RM)")
        out.append(f"Friedman χ²(3) = {self.fr_stat:.3f}, p = {self.fr_p:.3e}, n = {self.n}, Kendall's W = {self.kendall_w:.3f}")
        for r in self.rows:
            out.append(f"{r['label']}: n_eff={r['n_eff']}, W={r['W']:.1f}, p={r['p']:.3e}, p_Holm={r['p_holm']:.3e}, r_rb={r['r_rb']:.3f}")
        return "\n".join(out)

def friedman_wilcoxon_2x2(anova_df,
                          subject_col='subject',
                          rec_col='Recurrence',
                          sac_col='Saccade',
                          dv_col='Diff',
                          alternative='two-sided'):
    """
    Expects anova_df with columns: subject, Recurrence ('Yes'/'No'), Saccade ('Yes'/'No'), Diff.
    Returns a printable result object. Use: res = friedman_wilcoxon_2x2(anova_df); print(res)
    """

    # Pivot to subjects × 4 cells in a fixed order: RS, R, S, B
    wide = anova_df.pivot_table(index=subject_col,
                                columns=[rec_col, sac_col],
                                values=dv_col,
                                aggfunc='mean')
    # ensure expected columns exist
    expected_cols = [('Yes','Yes'), ('Yes','No'), ('No','Yes'), ('No','No')]
    missing = [c for c in expected_cols if c not in wide.columns]
    if missing:
        raise ValueError(f"Missing cells in data: {missing}. Each subject must have all 4 conditions.")
    wide = wide[expected_cols].dropna()

    # subjects retained
    n = wide.shape[0]
    if n == 0:
        raise ValueError("No complete subjects (rows) with all 4 conditions after pivot/dropna.")

    # Data matrix D: rows=subjects, cols=[RS, R, S, B]
    D = wide.to_numpy(dtype=float)
    RS, R, S, B = D[:, 0], D[:, 1], D[:, 2], D[:, 3]

    # ---- Omnibus: Friedman across the 4 related conditions ----
    fr_stat, fr_p = friedmanchisquare(RS, R, S, B)
    kendall_w = fr_stat / (n * (4 - 1))  # effect size for Friedman

    # ---- Planned 2×2 contrasts (Wilcoxon; Holm-corrected) ----
    # Interaction: (RS - R) - (S - B) == RS + B - R - S
    I = (RS - R) - (S - B)
    # Main effect (Recurrence): average of levels with R minus without
    A = 0.5*(RS + R) - 0.5*(S + B)
    # Main effect (Saccade): average of levels with S minus without
    Bf = 0.5*(RS + S) - 0.5*(R + B)

    labels = ["Interaction (RS−R)−(S−B)", "Main effect: Recurrence", "Main effect: Saccade"]
    vecs   = [I, A, Bf]

    pvals = []
    Wplus_list = []
    n_eff_list = []
    for v in vecs:
        # Drop exact zeros for Wilcoxon (SciPy does this internally, but we need n_eff for effect size)
        nz = v != 0
        v_nz = v[nz]
        if v_nz.size == 0:
            # Degenerate: all ties
            W_plus, p = 0.0, 1.0
            n_eff = 0
        else:
            try:
                res = wilcoxon(v_nz, alternative=alternative, mode="exact")
            except Exception:
                res = wilcoxon(v_nz, alternative=alternative, mode="approx")
            W_plus = float(res.statistic)  # SciPy actually returns min(W+, W-)!!
            p = float(res.pvalue)
            n_eff = v_nz.size
        pvals.append(p)
        Wplus_list.append(W_plus)
        n_eff_list.append(n_eff)

    # Holm correction across the three planned tests
    rej, p_holm, _, _ = multipletests(pvals, method="holm", alpha=0.05)

    # Rank-biserial correlation for Wilcoxon: r_rb = (2*R_plus/S) - 1, with S=n_eff*(n_eff+1)/2
    rows = []
    for lab, Wp, p, ph, neff in zip(labels, Wplus_list, pvals, p_holm, n_eff_list):
        if neff > 0:
            S_tot = neff * (neff + 1) / 2.0
            r_rb = 2.0 * (Wp / S_tot) - 1.0
        else:
            r_rb = np.nan
        rows.append({"label": lab, "n_eff": neff, "W": Wp, "p": p, "p_holm": ph, "r_rb": r_rb})

    return _FriedmanWilcoxon2x2Result(fr_stat, fr_p, n, kendall_w, rows)

#--------------------------------------------------------------------

@dataclass
class _BlockResult:
    label: str
    stats: Dict[str, Any]

@dataclass
class _FamilyResult:
    family: str
    tests: List[_BlockResult]

@dataclass
class _FriedmanWilcoxon2x2xKResult:
    n_subjects: int
    levels_backbone: List[Any]
    grand_omnibus: Optional[Dict[str, Any]]
    families: List[_FamilyResult]

    def __str__(self):
        lines = []
        lines.append(f"Repeated-measures nonparametric 2×2×{len(self.levels_backbone)} summary")
        lines.append(f"Subjects (complete cases): {self.n_subjects}")
        lines.append(f"Backbone levels: {self.levels_backbone}")
        if self.grand_omnibus is not None:
            g = self.grand_omnibus
            lines.append("\nGrand omnibus (Friedman across all 4×K cells):")
            lines.append(f"  χ²({g['df']})={g['chi2']:.4g}, p={g['p']:.4g}, Kendall's W={g['kendall_w']:.4g}")
        for fam in self.families:
            lines.append(f"\n{fam.family}:")
            if fam.family.startswith("Wilcoxon"):
                lines.append("  label | n_eff | W+ | p | p_holm | r_rb | w")
                for t in fam.tests:
                    s = t.stats
                    lines.append(f"  {t.label} | {s['n_eff']:>5} | {s['W_plus']:.4g} | {s['p']:.4g} | {s['p_holm']:.4g} | {s['r_rb'] if np.isfinite(s['r_rb']) else np.nan:.4g} | {s['W']:.4g}")
            else:
                lines.append("  label | χ² | df | p | p_holm | Kendall's W")
                for t in fam.tests:
                    s = t.stats
                    lines.append(f"  {t.label} | {s['chi2']:.4g} | {s['df']} | {s['p']:.4g} | {s['p_holm']:.4g} | {s['kendall_w']:.4g}")
        return "\n".join(lines)

def _wilcoxon_with_Wplus_and_rrb(v, alternative='two-sided'):
    v = np.asarray(v, float)
    nz = v != 0
    v_nz = v[nz]
    if v_nz.size == 0:
        return dict(n_eff=0, W_plus=0.0, p=1.0, r_rb=np.nan)

    # p-value from SciPy
    try:
        res = wilcoxon(v_nz, alternative=alternative, mode="exact")
    except Exception:
        res = wilcoxon(v_nz, alternative=alternative, mode="approx")

    # T⁺ from ranks of |v|
    ranks = rankdata(np.abs(v_nz), method='average')
    W_plus  = float(np.sum(ranks[v_nz > 0]))
    W_minus = float(np.sum(ranks[v_nz < 0]))
    W_min = min(W_plus, W_minus)
    S_tot   = W_plus + W_minus  # = sum of all ranks (handles ties)
    r_rb    = (W_plus - W_minus) / S_tot if S_tot > 0 else np.nan

    return dict(n_eff=v_nz.size, W_plus=W_plus, p=float(res.pvalue), r_rb=r_rb, W=W_min)

def friedman_wilcoxon_2x2xK(
    anova_df: pd.DataFrame,
    subject_col: str = 'subject',
    rec_col: str = 'Recurrence',      # two levels, e.g., 'Yes'/'No'
    sac_col: str = 'Saccade',         # two levels, e.g., 'Yes'/'No'
    back_col: str = 'Backbone',       # K levels (e.g., 3)
    dv_col: str = 'Diff',
    rec_levels: Tuple[Any, Any] = ('Yes', 'No'),
    sac_levels: Tuple[Any, Any] = ('Yes', 'No'),
    back_levels: Optional[List[Any]] = None,
    alternative: str = 'two-sided',
    do_grand_omnibus: bool = True
) -> _FriedmanWilcoxon2x2xKResult:
    """
    General 2×2×K repeated-measures nonparametric analysis.

    Expects long-format `anova_df` with columns:
      subject_col, rec_col (2 levels), sac_col (2 levels), back_col (K levels), dv_col.

    Returns:
      _FriedmanWilcoxon2x2xKResult with:
        - optional grand omnibus Friedman across all 4×K cells
        - Wilcoxon family (3 tests): Rec main, Sac main, Rec×Sac (collapsed over Backbone)
        - Friedman-K family (4 tests): Backbone main, Rec×Backbone, Sac×Backbone, 3-way Rec×Sac×Backbone
    """

    # determine backbone levels order
    if back_levels is None:
        # preserve order of first appearance
        back_levels = list(pd.unique(anova_df[back_col]))

    # pivot to subjects × (Rec, Sac, Back) cells
    wide = anova_df.pivot_table(
        index=subject_col,
        columns=[rec_col, sac_col, back_col],
        values=dv_col,
        aggfunc='mean'
    )

    # expected columns in fixed order
    expected_cols = []
    for bk in back_levels:
        for r in rec_levels:
            for s in sac_levels:
                expected_cols.append((r, s, bk))

    missing = [c for c in expected_cols if c not in wide.columns]
    if missing:
        raise ValueError(f"Missing cells in data: {missing}. Each subject must have all 4×K conditions.")

    # align, reorder, and drop incomplete subjects
    wide = wide[expected_cols].dropna()
    n = wide.shape[0]
    if n == 0:
        raise ValueError("No complete subjects (rows) with all 4×K conditions after pivot/dropna.")

    # helper to grab a cell vector
    def cell_vec(r, s, bk):
        return wide[(r, s, bk)].to_numpy(dtype=float)

    # Collect RS, R, S, B for each backbone (shape: list of arrays length n)
    RS = [cell_vec(rec_levels[0], sac_levels[0], bk) for bk in back_levels]
    R  = [cell_vec(rec_levels[0], sac_levels[1], bk) for bk in back_levels]
    S  = [cell_vec(rec_levels[1], sac_levels[0], bk) for bk in back_levels]
    B  = [cell_vec(rec_levels[1], sac_levels[1], bk) for bk in back_levels]

    K = len(back_levels)

    # ---------- Grand omnibus across all 4×K cells (optional) ----------
    grand_omnibus = None
    if do_grand_omnibus:
        # Friedman across all 4K repeated measures
        samples = RS + R + S + B  # list of length 4K, each n-vector
        fr = friedmanchisquare(*samples)
        chi2, p = float(fr.statistic), float(fr.pvalue)
        df = 4 * K - 1
        kendall_w = chi2 / (n * df) if n > 0 and df > 0 else np.nan
        grand_omnibus = dict(chi2=chi2, p=p, df=df, kendall_w=kendall_w)

    # ---------- Planned tests ----------

    # (A) Wilcoxon family (3 tests), collapsed over Backbone
    # Main effect: Recurrence (average across Sac & Backbone)
    rec_eff_components = [0.5 * (RS[k] + R[k]) - 0.5 * (S[k] + B[k]) for k in range(K)]
    A_vec = np.mean(np.stack(rec_eff_components, axis=1), axis=1)

    # Main effect: Saccade (average across Rec & Backbone)
    sac_eff_components = [0.5 * (RS[k] + S[k]) - 0.5 * (R[k] + B[k]) for k in range(K)]
    B_vec = np.mean(np.stack(sac_eff_components, axis=1), axis=1)

    # Rec×Sac interaction (collapsed over Backbone)
    int_RS_components = [(RS[k] - R[k]) - (S[k] - B[k]) for k in range(K)]
    I_RS_vec = np.mean(np.stack(int_RS_components, axis=1), axis=1)

    wilcoxon_labels = [
        "Main effect: Recurrence (collapsed over Saccade & Backbone)",
        "Main effect: Saccade (collapsed over Recurrence & Backbone)",
        "Interaction: (RS−R)−(S−B) (collapsed over Backbone)"
    ]
    wilcoxon_vecs = [A_vec, B_vec, I_RS_vec]

    wilcoxon_rows = []
    pvals_w = []
    for lab, v in zip(wilcoxon_labels, wilcoxon_vecs):
        out = _wilcoxon_with_Wplus_and_rrb(v, alternative=alternative)
        pvals_w.append(out["p"])
        wilcoxon_rows.append(_BlockResult(lab, dict(
            n_eff=out["n_eff"], W_plus=out["W_plus"], p=out["p"], r_rb=out["r_rb"], W=out["W"]
        )))

    # Holm within Wilcoxon family
    _, p_holm_w, _, _ = multipletests(pvals_w, method="holm", alpha=0.05)
    for i in range(len(wilcoxon_rows)):
        wilcoxon_rows[i].stats["p_holm"] = float(p_holm_w[i])

    # (B) Friedman-K family (4 tests) — K related samples each
    friedman_tests = []

    def _friedman_K(label: str, series_per_level: List[np.ndarray]) -> _BlockResult:
        # Each element is an n-length vector (one per backbone level)
        fr = friedmanchisquare(*series_per_level)
        chi2, p = float(fr.statistic), float(fr.pvalue)
        df = K - 1
        kendall_w = chi2 / (n * df) if n > 0 and df > 0 else np.nan
        return _BlockResult(label, dict(chi2=chi2, p=p, df=df, kendall_w=kendall_w))

    # Backbone main effect: mean of four cells within each backbone
    means_per_backbone = [0.25 * (RS[k] + R[k] + S[k] + B[k]) for k in range(K)]
    friedman_tests.append(_friedman_K("Main effect: Backbone", means_per_backbone))

    # Rec×Backbone: Recurrence effect within each backbone
    rec_eff_by_backbone = [0.5 * (RS[k] + R[k]) - 0.5 * (S[k] + B[k]) for k in range(K)]
    friedman_tests.append(_friedman_K("Interaction: Recurrence×Backbone", rec_eff_by_backbone))

    # Sac×Backbone: Saccade effect within each backbone
    sac_eff_by_backbone = [0.5 * (RS[k] + S[k]) - 0.5 * (R[k] + B[k]) for k in range(K)]
    friedman_tests.append(_friedman_K("Interaction: Saccade×Backbone", sac_eff_by_backbone))

    # 3-way Rec×Sac×Backbone: 2×2 interaction within each backbone
    int_by_backbone = [(RS[k] - R[k]) - (S[k] - B[k]) for k in range(K)]
    friedman_tests.append(_friedman_K("3-way interaction: Rec×Sac×Backbone", int_by_backbone))

    # Holm within Friedman-K family
    pvals_f = [t.stats["p"] for t in friedman_tests]
    rej_f, p_holm_f, _, _ = multipletests(pvals_f, method="holm", alpha=0.05)
    for i in range(len(friedman_tests)):
        friedman_tests[i].stats["p_holm"] = float(p_holm_f[i])

    families = [
        _FamilyResult("Wilcoxon (paired) — collapsed over Backbone", wilcoxon_rows),
        _FamilyResult(f"Friedman (K={K}) — per-backbone contrasts", friedman_tests)
    ]

    return _FriedmanWilcoxon2x2xKResult(
        n_subjects=n,
        levels_backbone=back_levels,
        grand_omnibus=grand_omnibus,
        families=families
    )

def _parse_gpn_model_name(name: str):
    """
    Parse strings like 'GPN-R-SimCLR', 'GPN-B-ILSVRC', 'GPN-RS-DINOv2-b'.
    Returns: variant in {'B','R','S','RS'} and canonical backbone string.
    """
    m = re.match(r'^GPN-(RS|R|S|B)-(.+)$', name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognized GPN model name format: {name}")
    variant = m.group(1).upper()

    raw_backbone = m.group(2)
    key = raw_backbone.lower().replace('_','-')
    canon = {
        'ilsvrc': 'ILSVRC',
        'simclr': 'SimCLR',
        'dino': 'DINO',
        'dinov2': 'DINOv2',
        'dinov2-b': 'DINOv2-b',
        'dino-v2-b': 'DINOv2-b',
        'dinov2-base': 'DINOv2-b',
    }
    backbone = canon.get(key, raw_backbone)  # fall back to original if unseen
    return variant, backbone

def build_2x2xK_df_from_best(best_gpn_corr: dict,
                             dv_key: str = 'correlation',
                             baseline_key: str = 'None',
                             subject_col: str = 'subject'):
    """
    best_gpn_corr[model_name][dv_key] -> (N,) array-like per model.
    Builds long-format DataFrame for friedman_wilcoxon_2x2xK with columns:
      subject, Recurrence ('Yes'/'No'), Saccade ('Yes'/'No'), Backbone, Correlation, Model
    """
    # infer N from the first model
    some_model = next(iter(best_gpn_corr))
    N = len(np.asarray(best_gpn_corr[some_model][dv_key]))
    rows = []

    for model_name, payload in best_gpn_corr.items():
        vals = np.asarray(payload[dv_key], dtype=float) if baseline_key == 'None' else np.asarray(payload[dv_key]-payload[baseline_key], dtype=float)
        if len(vals) != N:
            raise ValueError(f"Inconsistent subject count in {model_name}: expected {N}, got {len(vals)}")

        variant, backbone = _parse_gpn_model_name(model_name)

        rec = 'Yes' if variant in ('R','RS') else 'No'
        sac = 'Yes' if variant in ('S','RS') else 'No'

        for subj in range(N):
            rows.append({
                subject_col: subj,
                'Recurrence': rec,
                'Saccade': sac,
                'Backbone': backbone,
                'Correlation': float(vals[subj]),
                'Model': model_name
            })

    df = pd.DataFrame(rows)
    return df

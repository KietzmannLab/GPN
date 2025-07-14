import torch
import numpy as np
import h5py
import os
from sklearn.linear_model import LinearRegression

##############################
## Loading the dataset loaders
##############################

def get_Dataset_loaders(hyp,split):

    dataset_path = hyp['dataset']['dataset_path']
    gaze_type = hyp['network']['gaze_type']
    gaze_extract = hyp['network']['gaze_extract']
    timesteps = hyp['network']['timesteps']

    split_data = CocoGaze(split=split, dataset_path=dataset_path, gaze_type=gaze_type, gaze_extract=gaze_extract, timesteps=timesteps, in_memory= hyp['dataset']['in_memory'], r50v=hyp['dataset']['r50v'], dva_dataset=hyp['dataset']['dva_dataset'])

    if 'train' in split:
        data_loader = torch.utils.data.DataLoader(split_data, batch_size=hyp['optimizer']['batch_size'], shuffle=True,
                                                    num_workers=hyp['optimizer']['dataloader']['num_workers_train'],
                                                    prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor'])
    elif 'val' in split or 'test' in split:
        data_loader = torch.utils.data.DataLoader(split_data, batch_size=hyp['optimizer']['batch_size'], shuffle=False,
                                                    num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                    prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor'])
        
    return data_loader

class CocoGaze(torch.utils.data.Dataset):
    #Import dataset splitwise

    def __init__(self, split, dataset_path, gaze_type, timesteps, in_memory, r50v,  gaze_extract='-1', dva_dataset='NSD'):

        self.root_dir = dataset_path
        gaze_map = {'dg3': 0, 'random': 1, 'dg3p': 2, 'dg3s': 3}
        self.gaze_type = gaze_map[gaze_type] # 0 if dg3, 1 if random, 2 if dg3_permuted, 3 if dg3_swap
        self.gaze_extract = gaze_extract
        self.in_memory = in_memory
        self.split = split
        self.timesteps = timesteps
        self.dva_dataset = dva_dataset
        if self.dva_dataset == 'NSD':
            self.dataset_str = f'coco_NSD_dg3fix91_r50v{r50v}ap_7fix' if gaze_extract == '-1' else f'coco_NSD_dg3nfix91_r50v{r50v}ap_7fix'
        elif self.dva_dataset == 'AVS':
            self.dataset_str = f'coco_AVS_dg3fix36_r50v{r50v}ap_7fix' if gaze_extract == '-1' else f'coco_AVS_dg3nfix36_r50v{r50v}ap_7fix'

        if split == 'train' or split == 'val' or split == 'test':

            path_here = dataset_path + f'{self.dataset_str}_{split}.h5'

            if in_memory == 1:
                with h5py.File(path_here, "r") as f:
                    self.actvs = torch.from_numpy(f[split]['dg3_fix_actvs'][:,:,:timesteps+1,self.gaze_type,:][()]).float()
                    self.len = self.actvs.shape
                    self.next_fix_rel_coords = torch.from_numpy(f[split]['next_fix_rel_coords'][:,:,:timesteps,self.gaze_type,:][()]).float()
                    self.next_fix_coords = torch.from_numpy(f[split]['next_fix_coords'][:,:,:timesteps,self.gaze_type,:][()]).float()
                    self.semantic_embed = torch.from_numpy(f[split]['mpnet_embeddings'][()]).float()
                    self.scene_embed = torch.from_numpy(f[split]['full_image_actvs'][()]).float()
            else:
                with h5py.File(path_here, "r") as f:
                    self.len = f[split]['dg3_fix_actvs'][:,:,:timesteps+1,self.gaze_type,:].shape
        
        elif len(split.split('_')) == 2 and split.split('_')[0] == 'train':

            test_case = int(split.split('_')[1])
            
            path_here = dataset_path + f'{self.dataset_str}_test.h5'
            with h5py.File(path_here, "r") as f:
                n_imgs_tt = f['test']['mpnet_embeddings'].shape[0]
            idxs_notuse = np.load(f'/share/klab/datasets/NSD_special_imgs_pythonicDatasetIndices/pythonic_conds{test_case}.npy')

            path_here = dataset_path + f'{self.dataset_str}_train.h5'
            with h5py.File(path_here, "r") as f:
                n_imgs_tr = f['train']['mpnet_embeddings'].shape[0]
                actvs_shape = f['train']['dg3_fix_actvs'][:,:,:timesteps+1,self.gaze_type,:].shape
                next_fix_rel_coords_shape = f['train']['next_fix_rel_coords'][:,:,:timesteps,self.gaze_type,:].shape
                next_fix_coords_shape = f['train']['next_fix_coords'][:,:,:timesteps,self.gaze_type,:].shape
                semantic_embed_shape = f['train']['mpnet_embeddings'].shape
                scene_embed_shape = f['train']['full_image_actvs'].shape

            if in_memory == 1:
                self.actvs = torch.zeros((actvs_shape[0] + n_imgs_tt - len(idxs_notuse), *actvs_shape[1:]))
                self.len = self.actvs.shape
                self.next_fix_rel_coords = torch.zeros((next_fix_rel_coords_shape[0] + n_imgs_tt - len(idxs_notuse), *next_fix_rel_coords_shape[1:]))
                self.next_fix_coords = torch.zeros((next_fix_coords_shape[0] + n_imgs_tt - len(idxs_notuse), *next_fix_coords_shape[1:]))
                self.semantic_embed = torch.zeros((semantic_embed_shape[0] + n_imgs_tt - len(idxs_notuse), *semantic_embed_shape[1:]))
                self.scene_embed = torch.zeros((scene_embed_shape[0] + n_imgs_tt - len(idxs_notuse), *scene_embed_shape[1:]))
                
                path_here = dataset_path + f'{self.dataset_str}_train.h5'
                with h5py.File(path_here, "r") as f:
                    self.actvs[:n_imgs_tr] = torch.from_numpy(f['train']['dg3_fix_actvs'][:,:,:timesteps+1,self.gaze_type,:][()]).float()
                    self.next_fix_rel_coords[:n_imgs_tr] = torch.from_numpy(f['train']['next_fix_rel_coords'][:,:,:timesteps,self.gaze_type,:][()]).float()
                    self.next_fix_coords[:n_imgs_tr] = torch.from_numpy(f['train']['next_fix_coords'][:,:,:timesteps,self.gaze_type,:][()]).float()
                    self.semantic_embed[:n_imgs_tr] = torch.from_numpy(f['train']['mpnet_embeddings'][()]).float()
                    self.scene_embed[:n_imgs_tr] = torch.from_numpy(f['train']['full_image_actvs'][()]).float()

                path_here = dataset_path + f'{self.dataset_str}_test.h5'
                idxs_use = [x for i, x in enumerate([i for i in range(n_imgs_tt)]) if i not in list(idxs_notuse)]
                with h5py.File(path_here, "r") as f:
                    self.actvs[n_imgs_tr:] = torch.from_numpy(f['test']['dg3_fix_actvs'][idxs_use,:,:timesteps+1,self.gaze_type,:][()]).float()
                    self.next_fix_rel_coords[n_imgs_tr:] = torch.from_numpy(f['test']['next_fix_rel_coords'][idxs_use,:,:timesteps,self.gaze_type,:][()]).float()
                    self.next_fix_coords[n_imgs_tr:] = torch.from_numpy(f['test']['next_fix_coords'][idxs_use,:,:timesteps,self.gaze_type,:][()]).float()
                    self.semantic_embed[n_imgs_tr:] = torch.from_numpy(f['test']['mpnet_embeddings'][idxs_use,:][()]).float()
                    self.scene_embed[n_imgs_tr:] = torch.from_numpy(f['test']['full_image_actvs'][idxs_use,:][()]).float()
            else:
                self.len = [actvs_shape[0],actvs_shape[1]]
                self.len[0] = self.len[0] + n_imgs_tt - len(idxs_notuse)


        elif len(split.split('_')) == 2 and split.split('_')[0] == 'test':

            path_here = dataset_path + f'{self.dataset_str}_test.h5'
            test_case = int(split.split('_')[1])
            idxs_use = np.load(f'/share/klab/datasets/NSD_special_imgs_pythonicDatasetIndices/pythonic_conds{test_case}.npy')

            if in_memory == 1:
                with h5py.File(path_here, "r") as f:
                    self.actvs = torch.from_numpy(f['test']['dg3_fix_actvs'][idxs_use,:,:timesteps+1,self.gaze_type,:][()]).float()
                    self.len = self.actvs.shape
                    self.next_fix_rel_coords = torch.from_numpy(f['test']['next_fix_rel_coords'][idxs_use,:,:timesteps,self.gaze_type,:][()]).float()
                    self.next_fix_coords = torch.from_numpy(f['test']['next_fix_coords'][idxs_use,:,:timesteps,self.gaze_type,:][()]).float()
                    self.semantic_embed = torch.from_numpy(f['test']['mpnet_embeddings'][idxs_use,:][()]).float()
                    self.scene_embed = torch.from_numpy(f['test']['full_image_actvs'][idxs_use,:][()]).float()
            else:
                with h5py.File(path_here, "r") as f:
                    actvs_shape = f['test']['dg3_fix_actvs'][idxs_use,:,:timesteps+1,self.gaze_type,:].shape
                    self.len = actvs_shape

    def __len__(self):
        return (self.len[0]*self.len[1]) # each image * gaze trace is considered a sample
    
    def __getitem__(self, idx): # accepts ids for img/trace and returns for fixations, the actvs, fixs_coords, to the Dataloader

        img_n = idx//self.len[1]
        trace_n = idx%self.len[1]

        if self.in_memory == 1:

            actvs = self.actvs[img_n,trace_n,:,:]
            next_fix_rel_coords = self.next_fix_rel_coords[img_n,trace_n,:,:]
            fix_coords = self.next_fix_coords[img_n,trace_n,:,:]
            fix_coords = torch.cat((torch.Tensor([[0, 0]]), fix_coords), dim=0) # adding center coordinates to tensor
            semantic_embed = self.semantic_embed[img_n,:]
            scene_embed = self.scene_embed[img_n,:]

        else:

            if self.split == 'train' or self.split == 'val' or self.split == 'test':

                path_here = self.root_dir + f'{self.dataset_str}_{self.split}.h5'
                with h5py.File(path_here, "r") as f:
                    actvs = torch.from_numpy(f[self.split]['dg3_fix_actvs'][img_n,trace_n,:self.timesteps+1,self.gaze_type,:][()]).float()
                    next_fix_rel_coords = torch.from_numpy(f[self.split]['next_fix_rel_coords'][img_n,trace_n,:self.timesteps,self.gaze_type,:][()]).float()
                    fix_coords = torch.from_numpy(f[self.split]['next_fix_coords'][img_n,trace_n,:self.timesteps,self.gaze_type,:][()]).float()
                    fix_coords = torch.cat((torch.Tensor([[0, 0]]), fix_coords), dim=0) # adding center coordinates to tensor
                    semantic_embed = torch.from_numpy(f[self.split]['mpnet_embeddings'][img_n,:][()]).float()
                    scene_embed = torch.from_numpy(f[self.split]['full_image_actvs'][img_n,:][()]).float()

            elif len(self.split.split('_')) == 2 and self.split.split('_')[0] == 'train':

                test_case = int(self.split.split('_')[1])

                path_here = self.root_dir + f'{self.dataset_str}_test.h5'
                with h5py.File(path_here, "r") as f:
                    n_imgs_tt = f['test']['mpnet_embeddings'].shape[0]
                idxs_notuse = np.load(f'/share/klab/datasets/NSD_special_imgs_pythonicDatasetIndices/pythonic_conds{test_case}.npy')

                path_here = self.root_dir + f'{self.dataset_str}_train.h5'
                with h5py.File(path_here, "r") as f:
                    actvs_shape = f['train']['dg3_fix_actvs'][:,:,:self.timesteps+1,self.gaze_type,:].shape
                
                if img_n < actvs_shape[0]:
                    path_here = self.root_dir + f'{self.dataset_str}_train.h5'
                    with h5py.File(path_here, "r") as f:
                        actvs = torch.from_numpy(f['train']['dg3_fix_actvs'][img_n,trace_n,:self.timesteps+1,self.gaze_type,:][()]).float()
                        next_fix_rel_coords = torch.from_numpy(f['train']['next_fix_rel_coords'][img_n,trace_n,:self.timesteps,self.gaze_type,:][()]).float()
                        fix_coords = torch.from_numpy(f['train']['next_fix_coords'][img_n,trace_n,:self.timesteps,self.gaze_type,:][()]).float()
                        fix_coords = torch.cat((torch.Tensor([[0, 0]]), fix_coords), dim=0) # adding center coordinates to tensor
                        semantic_embed = torch.from_numpy(f['train']['mpnet_embeddings'][img_n,:][()]).float()
                        scene_embed = torch.from_numpy(f['train']['full_image_actvs'][img_n,:][()]).float()
                else:
                    path_here = self.root_dir + f'{self.dataset_str}_test.h5'
                    idxs_use = [x for i, x in enumerate([i for i in range(n_imgs_tt)]) if i not in list(idxs_notuse)]
                    with h5py.File(path_here, "r") as f:
                        actvs = torch.from_numpy(f['test']['dg3_fix_actvs'][idxs_use[img_n-actvs_shape[0]],trace_n,:self.timesteps+1,self.gaze_type,:][()]).float()
                        next_fix_rel_coords = torch.from_numpy(f['test']['next_fix_rel_coords'][idxs_use[img_n-actvs_shape[0]],trace_n,:self.timesteps,self.gaze_type,:][()]).float()
                        fix_coords = torch.from_numpy(f['test']['next_fix_coords'][idxs_use[img_n-actvs_shape[0]],trace_n,:self.timesteps,self.gaze_type,:][()]).float()
                        fix_coords = torch.cat((torch.Tensor([[0, 0]]), fix_coords), dim=0) # adding center coordinates to tensor
                        semantic_embed = torch.from_numpy(f['test']['mpnet_embeddings'][idxs_use[img_n-actvs_shape[0]],:][()]).float()
                        scene_embed = torch.from_numpy(f['test']['full_image_actvs'][idxs_use[img_n-actvs_shape[0]],:][()]).float()

            elif len(self.split.split('_')) == 2 and self.split.split('_')[0] == 'test':

                path_here = self.root_dir + f'{self.dataset_str}_test.h5'
                test_case = int(self.split.split('_')[1])
                idxs_use = np.load(f'/share/klab/datasets/NSD_special_imgs_pythonicDatasetIndices/pythonic_conds{test_case}.npy')

                with h5py.File(path_here, "r") as f:
                    actvs = torch.from_numpy(f['test']['dg3_fix_actvs'][idxs_use[img_n],trace_n,:self.timesteps+1,self.gaze_type,:][()]).float()
                    next_fix_rel_coords = torch.from_numpy(f['test']['next_fix_rel_coords'][idxs_use[img_n],trace_n,:self.timesteps,self.gaze_type,:][()]).float()
                    fix_coords = torch.from_numpy(f['test']['next_fix_coords'][idxs_use[img_n],trace_n,:self.timesteps,self.gaze_type,:][()]).float()
                    fix_coords = torch.cat((torch.Tensor([[0, 0]]), fix_coords), dim=0) # adding center coordinates to tensor
                    semantic_embed = torch.from_numpy(f['test']['mpnet_embeddings'][idxs_use[img_n],:][()]).float()
                    scene_embed = torch.from_numpy(f['test']['full_image_actvs'][idxs_use[img_n],:][()]).float()

        img_n = torch.Tensor([img_n])
        trace_n = torch.Tensor([trace_n])

        return actvs, next_fix_rel_coords, fix_coords, semantic_embed, scene_embed, img_n, trace_n
    
def create_cpc_matrix(N, m, mask=True):
    # mask==True creates a mask matrix; if False creates a matrix that can be used directly for loss

    # Calculate the block value and the rest value
    if mask:
        block_value = 2
        rest_value = 3
        diag_value = 1
    else:
        block_value = 0.5 * (1 / ((N/m) * (m**2 - m)))
        rest_value = 0.5 * (1 / (N**2 - (N/m) * (m**2)))
        diag_value = -1/N
    
    # Initialize the matrix with the rest_value
    matrix = torch.full((N, N), rest_value, requires_grad=False)
    
    # Set the diagonal elements
    torch.diagonal(matrix).fill_(diag_value)
    
    # Fill the blocks along the diagonal
    for i in range(0, N, m):
        matrix[i:i+m, i:i+m] = torch.full((m, m), block_value)
        torch.diagonal(matrix[i:i+m, i:i+m]).fill_(diag_value)

    return matrix

class LinearFitScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, factor=1./2, min_lr=1e-8, min_percent_change=1.0, mode='min', patience=5, last_epoch=-1, verbose=False):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            num_epochs (int): Number of epochs to use for the linear fit.
            factor (float): Factor by which the learning rate will be reduced. Default: 0.1.
            min_lr (float): Minimum learning rate. Default: 1e-6.
            min_percent_change (float): Minimum absolute percentage change in the metric to not trigger a reduction. Default: 1.0.
            mode (str): One of 'min' or 'max'. 'min' will reduce the LR if the metric has not decreased by min_percent_change,
                        'max' will reduce the LR if the metric has not increased by min_percent_change. Default: 'min'.
            patience (int): Number of epochs with no improvement after which learning rate will be reduced. Default: 0.
            last_epoch (int): The index of the last epoch. Default: -1.
            verbose (bool): If True, prints a message to stdout for each update. Default: False.
        """
        self.num_epochs = num_epochs
        self.factor = factor
        self.min_lr = min_lr
        self.min_percent_change = min_percent_change
        self.mode = mode
        self.patience = patience
        self.num_bad_epochs = 0  # Track the number of epochs without improvement
        self.verbose = verbose
        self.metric_history = []
        super(LinearFitScheduler, self).__init__(optimizer, last_epoch, verbose)

    def step(self, metric=None):
        """
        Step should be called after each epoch. Can be called without 'metric' during initialization.
        
        Args:
            metric (float, optional): Current epoch's metric. Default is None.
        """
        # Increment the last_epoch attribute from the base class
        self.last_epoch += 1
        
        if metric is not None:
            # Update metric history
            self.metric_history.append(metric)
            
            # Only perform the check if we have enough history
            if len(self.metric_history) >= self.num_epochs:
                # Perform linear fit
                epochs = np.arange(self.num_epochs).reshape(-1, 1)
                metrics = np.array(self.metric_history[-self.num_epochs:]).reshape(-1, 1)
                
                reg = LinearRegression().fit(epochs, metrics)
                slope = reg.coef_[0, 0]
                intercept = reg.intercept_[0]
                
                # Calculate the predicted metrics
                predicted_start = intercept
                predicted_end = slope * (self.num_epochs - 1) + intercept
                
                # Calculate percent change based on the magnitude of the start value
                if predicted_start != 0:
                    percent_change = 100 * (predicted_end - predicted_start) / abs(predicted_start)
                else:
                    percent_change = float('inf')  # Avoid division by zero
                if self.verbose:
                    print(f"Percent_change in metric: {percent_change:.2f}%")
                    
                # Determine if we should adjust the learning rate based on the mode and percent change
                if self.mode == 'min' and percent_change > -self.min_percent_change:
                    self.num_bad_epochs += 1
                elif self.mode == 'max' and percent_change < self.min_percent_change:
                    self.num_bad_epochs += 1
                else:
                    self.num_bad_epochs = 0  # Reset counter if improvement is observed
                
                # Check if we have hit the patience threshold
                if self.num_bad_epochs > self.patience:
                    self.reduce_lr(percent_change)
                    self.num_bad_epochs = 0  # Reset bad epoch count after reducing LR

    def reduce_lr(self, percent_change):
        """Reduce the learning rate according to the factor and min_lr constraints and print verbose message."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            if self.verbose:
                print(f"Reducing learning rate of group {i} to {new_lr:.4e}. Percent change: {percent_change:.2f}%. Patience exceeded.")
    
    
##############################
## Logging functions
##############################
    
def create_folders_logging(net_name, create=1):

    print('Accessing log folders...')

    log_folder = 'logs/perf_logs'
    net_folder = 'logs/net_params'

    if create:
        isExist = os.path.exists(log_folder)
        if not isExist:
            os.makedirs(log_folder)
            print('Log folder is created!')
        isExist = os.path.exists(net_folder)
        if not isExist:
            os.makedirs(net_folder)
            print('Net folder is created!')

    log_folder_name = log_folder+f'/{net_name}'
    net_folder_name = net_folder+f'/{net_name}'

    if create:
        isExist = os.path.exists(log_folder_name)
        if not isExist:
            os.makedirs(log_folder_name)
            print('Specific log folder is created!')
        isExist = os.path.exists(net_folder_name)
        if not isExist:
            os.makedirs(net_folder_name)
            print('Specific net folder is created!')

    return log_folder_name, net_folder_name 
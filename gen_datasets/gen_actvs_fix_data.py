# Code to extract glimpse crops given the COCO scenes and DG3 fixations, and to extract RN50 activations for those crops.

import argparse
parser = argparse.ArgumentParser(description='Obtaining hyps')
parser.add_argument('--split', type=str, default='test') # train/test/val
parser.add_argument('--r50v', type=int, default=6) # 1/2/0 - 0 is init 3 is barlowtwins (https://github.com/facebookresearch/barlowtwins), 4 is dvd-b, 5 is dinov2b, 6 is simclr
parser.add_argument('--nfixs', type=int, default=7) # 7 is good for now
parser.add_argument('--dataset', type=str, default='NSD') # NSD - decides glimpse size
args = parser.parse_args()

import h5py
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import time

dataset_h = args.dataset # NSD or AVS
n_fixs = args.nfixs # no of fixations to be extracted
rn50_v = args.r50v # 1 or 2 - 1 has better IT brainscore (our choice), 2 has overall better brainscore

original_data = h5py.File('/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze_16_fixations.h5', 'r') # your path to the original dataset with DeepGaze 3 fixations

im_size = 256
fovea = 3
if dataset_h == 'NSD':
    scene_ext = 8.4
    glimpse_ext = 91
elif dataset_h == 'AVS':
    scene_ext = 21.61
    glimpse_ext = 36

output_path = f'/share/klab/datasets/GPN/coco_{dataset_h}_dg3fix{glimpse_ext}_r50v{rn50_v}ap_{n_fixs}fix_{args.split}.h5'

dataset_description = f'This dataset contains ResNet-50-v{rn50_v} avgpool activations for each of the {n_fixs} fixations sampled from DeepGaze 3 on MS-Coco images, and for the full image. Also contains the MPnet embeddings. The Coco images were {im_size}px. The fixations are {glimpse_ext}px as that corresponds to FWHM of retinal cone density (3 degs diameter; from https://journals.plos.org/plosone/article/figure?id=10.1371/journal.pone.0191141.g005), given that in the {dataset_h} dataset, the Coco images were presented {scene_ext} degs wide. The relative and absolute positions (rel. to center) of the next fixations are also recorded.'

device = 'cuda' # 'cuda', 'mps', 'cpu'

actvs_dim = 2048 if rn50_v != 5 else 768 # for RN50 2048, 1024 for dinov2b

# data_splits = ['train','test','val']; train ~ 48k, val ~ 2k, test - 73k
data_splits = [args.split]

# Network setup

print('\nLoading network...\n')

if rn50_v == 1:
    weights=ResNet50_Weights.IMAGENET1K_V1
elif rn50_v == 2:
    weights=ResNet50_Weights.IMAGENET1K_V2
preprocess = transforms.Compose([
    transforms.Resize(224, antialias=True), # resize to 224 as expected by IMAGENET1K_V1/2
    transforms.ConvertImageDtype(torch.float), # Convert image to float
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
if rn50_v == 0:
    net = resnet50()
elif rn50_v == 1 or rn50_v == 2:
    net = resnet50(weights=weights)
elif rn50_v == 3: # Barlow Twins
    net = torch.hub.load(
        'facebookresearch/barlowtwins:main',  # repo@branch
        'resnet50',                           # entry-point in hubconf.py
        pretrained=True,                      # pulls the 1 000-epoch weights
        verbose=False
        )
elif rn50_v == 4: # DVD-B
    net = resnet50()
    net.fc = torch.nn.Linear(2048, 565) # DVD-B has 565-dim output (ecoset)
    model_path = 'dvd-b-565.pth'
    state_dict = torch.load(model_path, map_location='cpu')['state_dict']
    state_dict = {k.replace('module._orig_mod.', ''): v for k, v in state_dict.items()}  # remove 'module._orig_mod.' prefix
    net.load_state_dict(state_dict)
    preprocess = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.ConvertImageDtype(torch.float)
    ])
elif rn50_v == 5: # not rn50 actually - dinov2b
    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
elif rn50_v == 6: # SimCLR
    net = resnet50(weights=None)
    state_dict_h = torch.load('../analysis/ResNet50 1x.pth', map_location='cpu') # your path to the simclr weights
    net.load_state_dict(state_dict_h['state_dict'])
    preprocess = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.ConvertImageDtype(torch.float),
    ])
net.to(device)
if rn50_v != 5:
    activation = {} # remember to empty the list after each forward pass
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    net.avgpool.register_forward_hook(get_activation('avgpool'))
net.eval() # important!! so it uses the trained batchnorm stats

print('Network is loaded. Preparing h5...\n')

# Filling the dataset

with h5py.File(output_path, "w") as f:

    f.create_dataset("dataset_description", data=dataset_description)

    print('\n')

    t0 = time.time()

    for split_id,split in enumerate(data_splits):

        print('\n\n')

        n_imgs = original_data[split]['densenet_deepgaze_fixations'].shape[0]
        n_fix_traces = original_data[split]['densenet_deepgaze_fixations'].shape[1]
        # n_fixs = original_data[split]['densenet_deepgaze_fixations'].shape[2]
        print('Creating fix_dg3 {} dataset with {} images, {} traces, {} fixations...\n'.format(split, int(n_imgs), int(n_fix_traces), int(n_fixs)), end='')

        group = f.create_group(split)
        group.create_dataset("dg3_fix_actvs", shape=(n_imgs,n_fix_traces,n_fixs,4,actvs_dim), dtype=np.float16) # dg3/random/dg3_permute/dg3_random, actvs
        # group.create_dataset("dg3_gist_actvs", shape=(n_imgs,n_fix_traces,n_fixs,2,actvs_dim), dtype=np.float16)
        group.create_dataset("next_fix_coords", shape=(n_imgs,n_fix_traces,n_fixs-1,4,2), dtype=np.int32) # dg3/random, x/y
        group.create_dataset("next_fix_rel_coords", shape=(n_imgs,n_fix_traces,n_fixs-1,4,2), dtype=np.int32) # dg3/random, dx/dy
        group.create_dataset("mpnet_embeddings", data=original_data[split]['all_mpnet_base_v2_mean_embeddings']) # mpnet embeddings
        group.create_dataset("full_image_actvs", shape=(n_imgs,actvs_dim), dtype=np.float16) # full image actvs rn50
        group.create_dataset("dg3_fixation_cover", shape=(n_imgs,n_fix_traces,im_size,im_size), dtype=np.float16) # wherever a fixation happens in a trace, the region is highlighted accumulatively (1 per fixation) to get an overall coverage of the glimpses - important for controls

        for img_id in range(n_imgs):

            print('\rCreating fix_dg3 {} dataset {} / {}...'.format(split, int(img_id+1), int(n_imgs)), end='')
            print('Time elapsed: {}s'.format(int(time.time()-t0)), end='')

            trace_dg3_permute_ids = np.hstack([np.zeros([n_fix_traces,1]),np.vstack([np.random.permutation(n_fixs-1)+1 for _ in range(n_fix_traces)])]) # permuting order of dg3 fixations to randomise them while keeping distribution

            for ftrace_id in range(n_fix_traces):

                im_h = np.zeros([int(im_size+glimpse_ext),int(im_size+glimpse_ext),3],dtype=original_data[split]['data'][img_id].dtype)
                im_h[int(glimpse_ext/2):int(glimpse_ext/2)+im_size,int(glimpse_ext/2):int(glimpse_ext/2)+im_size,:] = original_data[split]['data'][img_id] # because of this padding, the original fixation positions are now at the top-left of the glimpse window, so just need to add glimpse_ext to that for the other positions of the window!

                im_h_mask = np.zeros([int(im_size+glimpse_ext),int(im_size+glimpse_ext)],dtype=np.float16)

                trace_rand = np.random.randint(0,im_size,[n_fixs-1,2])

                dg3_random = original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,:n_fixs,:]
                for i in range(dg3_random.shape[0]-1):
                    dg3_random[i+1,:] = original_data[split]['densenet_deepgaze_fixations'][np.random.randint(n_imgs),np.random.randint(n_fix_traces),np.random.randint(1,n_fixs),:]

                im_h_glimpses = np.zeros([n_fixs*4,int(glimpse_ext),int(glimpse_ext),3],dtype=original_data[split]['data'][img_id].dtype)

                for fix_id in range(n_fixs):
                    for fix_type in range(4): # dg3/random/dg3_permute/dg3_swap - extracting fixations and filling gaze coords info

                        if fix_type == 0: # DG3
                            
                            if fix_id < n_fixs-1: # in fix_coords we are filling coords wrt the centre where first fixation is
                                f[split]['next_fix_coords'][img_id,ftrace_id,fix_id,fix_type,:] = original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,fix_id+1,:] - original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,0,:]
                                f[split]['next_fix_rel_coords'][img_id,ftrace_id,fix_id,fix_type,:] = original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,fix_id+1,:] - original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,fix_id,:]

                            x_ext_low = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,fix_id,0])
                            x_ext_high = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,fix_id,0]+glimpse_ext)
                            y_ext_low = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,fix_id,1])
                            y_ext_high = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,fix_id,1]+glimpse_ext)

                            im_h_glimpses[fix_id+n_fixs*fix_type,:,:,:] = im_h[y_ext_low:y_ext_high,x_ext_low:x_ext_high,:]

                            im_h_mask[y_ext_low:y_ext_high,x_ext_low:x_ext_high] += 1

                        elif fix_type == 2: # DG3 sequence permuted

                            if fix_id < n_fixs-1: # in fix_coords we are filling coords wrt the centre where first fixation is
                                f[split]['next_fix_coords'][img_id,ftrace_id,fix_id,fix_type,:] = original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,int(trace_dg3_permute_ids[ftrace_id,fix_id+1]),:] - original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,int(trace_dg3_permute_ids[ftrace_id,0]),:]
                                f[split]['next_fix_rel_coords'][img_id,ftrace_id,fix_id,fix_type,:] = original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,int(trace_dg3_permute_ids[ftrace_id,fix_id+1]),:] - original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,int(trace_dg3_permute_ids[ftrace_id,fix_id]),:]

                            x_ext_low = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,int(trace_dg3_permute_ids[ftrace_id,fix_id]),0])
                            x_ext_high = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,int(trace_dg3_permute_ids[ftrace_id,fix_id]),0]+glimpse_ext)
                            y_ext_low = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,int(trace_dg3_permute_ids[ftrace_id,fix_id]),1])
                            y_ext_high = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,int(trace_dg3_permute_ids[ftrace_id,fix_id]),1]+glimpse_ext)

                            im_h_glimpses[fix_id+n_fixs*fix_type,:,:,:] = im_h[y_ext_low:y_ext_high,x_ext_low:x_ext_high,:]

                        elif fix_type == 1: # random

                            if fix_id < n_fixs-1:
                                f[split]['next_fix_coords'][img_id,ftrace_id,fix_id,fix_type,:] = trace_rand[fix_id,:] - original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,0,:]
                                if fix_id > 0:
                                    f[split]['next_fix_rel_coords'][img_id,ftrace_id,fix_id,fix_type,:] = trace_rand[fix_id,:] - trace_rand[fix_id-1,:]
                                else:
                                    f[split]['next_fix_rel_coords'][img_id,ftrace_id,fix_id,fix_type,:] = trace_rand[fix_id,:] - original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,0,:]

                            if fix_id > 0:
                                x_ext_low = int(trace_rand[fix_id-1,0])
                                x_ext_high = int(trace_rand[fix_id-1,0]+glimpse_ext)
                                y_ext_low = int(trace_rand[fix_id-1,1])
                                y_ext_high = int(trace_rand[fix_id-1,1]+glimpse_ext)
                            else:
                                x_ext_low = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,0,0])
                                x_ext_high = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,0,0]+glimpse_ext)
                                y_ext_low = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,0,1])
                                y_ext_high = int(original_data[split]['densenet_deepgaze_fixations'][img_id,ftrace_id,0,1]+glimpse_ext)

                            im_h_glimpses[fix_id+n_fixs*fix_type,:,:,:] = im_h[y_ext_low:y_ext_high,x_ext_low:x_ext_high,:]

                        elif fix_type == 3: # DG3 randomly sampled fixations post 0

                            if fix_id < n_fixs-1: # in fix_coords we are filling coords wrt the centre where first fixation is
                                f[split]['next_fix_coords'][img_id,ftrace_id,fix_id,fix_type,:] = dg3_random[fix_id+1,:] - dg3_random[0,:]
                                f[split]['next_fix_rel_coords'][img_id,ftrace_id,fix_id,fix_type,:] = dg3_random[fix_id+1,:] - dg3_random[fix_id,:]

                            x_ext_low = int(dg3_random[fix_id,0])
                            x_ext_high = int(dg3_random[fix_id,0]+glimpse_ext)
                            y_ext_low = int(dg3_random[fix_id,1])
                            y_ext_high = int(dg3_random[fix_id,1]+glimpse_ext)

                            im_h_glimpses[fix_id+n_fixs*fix_type,:,:,:] = im_h[y_ext_low:y_ext_high,x_ext_low:x_ext_high,:]

                            im_h_mask[y_ext_low:y_ext_high,x_ext_low:x_ext_high] += 1

                f[split]['dg3_fixation_cover'][img_id,ftrace_id,:,:] = im_h_mask[int(glimpse_ext/2):int(glimpse_ext/2)+im_size,int(glimpse_ext/2):int(glimpse_ext/2)+im_size]

                # extracting RN50 activations and filling them in

                im_input = preprocess(torch.from_numpy(im_h_glimpses).permute(0, 3, 1, 2)).to(device)
                _ = net(im_input)

                actvs_h = net(im_input).detach().cpu().numpy().squeeze() if rn50_v == 5 else activation['avgpool'].detach().cpu().numpy().squeeze()
                f[split]['dg3_fix_actvs'][img_id,ftrace_id,:,0,:] = actvs_h[:n_fixs,:]
                f[split]['dg3_fix_actvs'][img_id,ftrace_id,:,1,:] = actvs_h[n_fixs:2*n_fixs,:]
                f[split]['dg3_fix_actvs'][img_id,ftrace_id,:,2,:] = actvs_h[2*n_fixs:3*n_fixs,:]
                f[split]['dg3_fix_actvs'][img_id,ftrace_id,:,3,:] = actvs_h[3*n_fixs:4*n_fixs,:]

                activation = {}
                
            # extracting RN50 activation for the full image

            im_input = preprocess(torch.from_numpy(original_data[split]['data'][img_id]).unsqueeze(0).permute(0, 3, 1, 2)).to(device) 
            _ = net(im_input)

            actvs_h = net(im_input).detach().cpu().numpy().squeeze() if rn50_v == 5 else activation['avgpool'].detach().cpu().numpy().squeeze()
            f[split]['full_image_actvs'][img_id,:] = actvs_h

            activation = {}

print('\nDone creating dataset!')


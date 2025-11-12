#!/usr/bin/env python3
"""
extract_RDMs_optimized.py  —  Memory‑efficient extraction of activations & RDMs
--------------------------------------------------------------------------
* Streams activations to an on‑disk HDF5 file in float16 instead of keeping
  every layer × image in RAM.
* Attaches hooks only to the layers of interest:
    • ViT‑like models  → MLP of every transformer block + final normals/proj
    • ResNet‑like      → ReLUs after each residual block + avgpool + fc
* Builds each RDM block‑wise so that ≤ (block × F) elements ever reside
  simultaneously in memory.

Options:
   alexnet
   rn50, rn152, rn50init, rn50bt, rn50simclr
   dinov2_vitg14_reg, dinov2_vitb14, dinov2_vitl14
   dinov3_vitb16, dinov3_convnext_large, dinov3_vit7b16
   dino_webssl_7b, dino_webssl_3b, dino_webssl_1b
   mae_vith, mae_vitb, mae_vitl
   mae_webssl_3b, mae_webssl_1b, mae_webssl_300m
   ijepa_vitg16_22k, ijepa_vith14_1k
   clip_vitl, clip_vitb, clip_RN50
   siglip2_g, siglip2_l, siglip2_b
   DG3
   coco_tr_rn50_multihot, coco_tr_rn50_mpnet,
   levit_128, levit_256, levit_384
   hardcorenas_f, hardcorenas_d, hardcorenas_a
   efficientnet_b1, efficientnet_b3, efficientnet_b7
   mpnet_blt
   convnext_tiny

Usage example:
    python extract_RDMs_optimized.py \
        --model rn50 --split test_515 \
        --batch_size 64 --out_dir ./RDMs
"""

import argparse, os, re, math, gc, pickle, h5py
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
from scipy.spatial.distance import squareform

#############################################################################
# 1.  Argument parsing
#############################################################################

def get_cli_args():
    p = argparse.ArgumentParser(description="Extract activations & RDMs w/ low‑RAM footprint")
    p.add_argument("--model", type=str, default="efficientnet_b3", help="model nickname")
    p.add_argument("--split", type=str, default="test_515", help="dataset split or custom subset")
    p.add_argument("--batch_size", type=int, default=50, help="images per GPU batch")
    p.add_argument("--in_memory", type=int, default=1, help="load images into RAM (0/1)")
    p.add_argument("--out_dir", type=str, default="../rdms", help="directory for .h5 and .pkl output")
    p.add_argument("--extent", type=str, default="NSD_crop", help="image extent: full or crop - NSD_crop or AVS_crop")
    return p.parse_args()

args = get_cli_args()
Path(args.out_dir).mkdir(parents=True, exist_ok=True)

#############################################################################
# 2.  Dataset
#############################################################################

class Coco(torch.utils.data.Dataset):
    """HDF5‑backed COCO subset."""

    def __init__(self, split, dataset_path, in_memory, preprocess=None, extent="full"):
        self.dataset = h5py.File(dataset_path, "r")
        self.in_memory = in_memory
        self.preprocess = preprocess
        self.extent = extent
        if extent != "full":
            assert extent in ["NSD_crop", "AVS_crop"], "Unknown extent option"
            print(f"▶ Using {extent} images")

        if len(split.split("_")) == 2 and split.startswith("test"):
            test_case = int(split.split("_")[1])
            split_grp = "test"
            idxs_use = np.load(f"/share/klab/datasets/NSD_special_imgs_pythonicDatasetIndices/pythonic_conds{test_case}.npy")
            print(f"▶ Using {len(idxs_use)} images from NSD special test case {test_case}")
        else:
            split_grp = split
            idxs_use = np.arange(self.dataset[split_grp]["data"].shape[0])

        imgs = self.dataset[split_grp]["data"]
        self.images = imgs[idxs_use] if not in_memory else imgs[idxs_use][()]
        print("▶ Images {}loaded into RAM".format("" if in_memory else "not "))
        self._len = len(idxs_use)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        img = self.images[idx] if self.in_memory else self.images[idx][()]
        if self.preprocess is not None:
            img = Image.fromarray(img)
            if self.extent == "NSD_crop":
                # center 91px crop
                img = img.crop((128-45, 128-45, 128+45, 128+45))
            elif self.extent == "AVS_crop":
                # center 36px crop
                img = img.crop((128-18, 128-18, 128+18, 128+18))
            return self.preprocess(img)   
        else:
            if self.extent == "NSD_crop":
                img = img[128-45:128+45, 128-45:128+45, :]
            elif self.extent == "AVS_crop":
                img = img[128-18:128+18, 128-18:128+18, :]
            return torch.Tensor(img)

def get_dataloader(split, in_memory, preprocess, extent="full"):
    dset = Coco(split, "/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze_16_fixations.h5", in_memory, preprocess, extent=extent)
    loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False,num_workers=2, prefetch_factor=2)
    return dset, loader

#############################################################################
# 3.  Model zoo — now **includes ResNets**
#############################################################################

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
def preprocess_batch(img_batch,model,device): # for DG3
    from scipy.ndimage import zoom
    from scipy.special import logsumexp
    centerbias_path = '/share/klab/datasets/GPN/centerbias_mit1003.npy'
    cb = np.load(centerbias_path)  
    cb = zoom(cb, (256/cb.shape[0], 256/cb.shape[0]), order=0, mode='nearest')
    cb -=  logsumexp(cb) 
    cb = torch.from_numpy(cb).to(device).unsqueeze(0)
    img_batch = img_batch.to(device)
    histories = [[[256//2],[256//2]] for _ in range(img_batch.shape[0])]
    x_list, y_list = [], []
    for hist in histories:
        hx = get_fixation_history(hist[0], model)
        hy = get_fixation_history(hist[1], model)
        x_list.append(hx); y_list.append(hy)
    x_hist = torch.tensor(x_list, device=device)
    y_hist = torch.tensor(y_list, device=device)
    # _ = model(img_batch, cb, x_hist, y_hist)
    return img_batch, cb, x_hist, y_hist

def load_model_and_preprocess(model_name: str):
    """Return (model, preprocess_fn, device) for the nickname."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- COCO finetuned ----------------------------------------
    if model_name.startswith("coco_"):
        from helper_funcs import get_network_model, create_folders_logging
        output_type = 'multihot' if 'multihot' in model_name else 'mpnet'
        dropout = 0.5 if 'multihot' in model_name else 0.25
        if 'trft' in model_name:
            setting = 'trft'
        elif 'ft' in model_name:
            setting = 'finetune'
        elif 'tr' in model_name:
            setting = 'transfer'
        else:
            setting = 'scratch'
        model, net_name = get_network_model(output_type, 3, 1024, dropout, setting=setting)
        _, net_path = create_folders_logging(net_name,create=0)
        model.load_state_dict(torch.load(f"{net_path}/{net_name}.pth"))
        preprocess = transforms.Compose([
            transforms.Resize(224, antialias=True),                  # resize
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),        # normalize
        ])
        model.to(device)
        return model, preprocess, device

    #-------- AlexNet ---------------------------------------------------

    if model_name == "alexnet":
        from torchvision.models import alexnet, AlexNet_Weights
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(device)
        preprocess = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, preprocess, device

    #--------- ConvNext ---------------------------------------------------
    if "convnext" in model_name:
        if model_name == "convnext_tiny":
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
            preprocess = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        return model, preprocess, device

    # -------- DG3 ---------------------------------------------------
    if model_name == "DG3":
        import deepgaze_pytorch
        model = deepgaze_pytorch.DeepGazeIII(pretrained=True).float().to(device)
        preprocess = transforms.Compose([
            transforms.Resize(256, antialias=True),  # PIL -> PIL
            transforms.PILToTensor(),                # PIL -> torch.uint8 [C,H,W]
        ])
        return model, preprocess, device

    # -------- CLIP --------------------------------------------------------
    if model_name.startswith("clip"):
        import clip  # from - https://github.com/openai/CLIP
        model_id = {
            "clip_vitb": "ViT-B/16",
            "clip_vitl": "ViT-L/14@336px",
            "clip_RN50": "RN50",
        }[model_name]
        model, preprocess = clip.load(model_id, device=device)
        return model, preprocess, device

    # -------- DINOv2 vanilla ---------------------------------------------
    if model_name.startswith("dinov2") and "webssl" not in model_name:
        # from - https://github.com/facebookresearch/dinov2
        model = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        if hasattr(model, "config") and hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = False
        return model, preprocess, device

    # ------------ LeViTs -----------------------------------------
    levit = {
        "levit_128": "levit_128.fb_dist_in1k",
        "levit_256": "levit_256.fb_dist_in1k",
        "levit_384": "levit_384.fb_dist_in1k",
    }
    # from https://huggingface.co/timm/levit_128.fb_dist_in1k etc
    if model_name.startswith("levit"):
        import timm
        model = timm.create_model(levit[model_name], pretrained=True).to(device)
        data_config = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_config, is_training=False)
        return model, preprocess, device
    
    #--------------- HardCoreNAS -----------------------------------
    hardcorenas = {
        "hardcorenas_f": "hardcorenas_f.miil_green_in1k",
        "hardcorenas_d": "hardcorenas_d.miil_green_in1k",
        "hardcorenas_a": "hardcorenas_a.miil_green_in1k",
    }
    # from https://huggingface.co/timm/hardcorenas_f.miil_green_in1k etc
    if model_name.startswith("hardcorenas"):
        import timm
        model = timm.create_model(hardcorenas[model_name], pretrained=True).to(device)
        data_config = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_config, is_training=False)
        return model, preprocess, device
    
    #--------------- EfficientNets -----------------------------------
    if model_name == "efficientnet_b1":
        from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
        model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1).to(device)
        preprocess = EfficientNet_B1_Weights.IMAGENET1K_V1.transforms()
        return model, preprocess, device
    elif model_name == "efficientnet_b3":
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).to(device)
        preprocess = EfficientNet_B3_Weights.IMAGENET1K_V1.transforms()
        return model, preprocess, device
    elif model_name == "efficientnet_b7":
        from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
        model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1).to(device)
        preprocess = EfficientNet_B7_Weights.IMAGENET1K_V1.transforms()
        return model, preprocess, device

    # -------- ResNets (ImageNet) -----------------------------------------
    if model_name.startswith("rn"):
        # from - https://docs.pytorch.org/vision/main/models.html 
        from torchvision.models import (
            resnet50, resnet152,
            ResNet50_Weights, ResNet152_Weights,
        )
        if model_name == "rn50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "rn50init":
            model = resnet50(weights=None)
        elif model_name == "rn152":
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        elif model_name == "rn50bt":
            # from - https://github.com/facebookresearch/barlowtwins
            model = torch.hub.load(
            'facebookresearch/barlowtwins:main',  # repo@branch
            'resnet50',                           # entry-point in hubconf.py
            pretrained=True,                      # pulls the 1 000-epoch weights
            verbose=False
            )
        elif model_name == "rn50simclr":
            # from - https://huggingface.co/lightly-ai/simclrv1-imagenet1k-resnet50-1x
            model = resnet50(weights=None)
            state_dict_h = torch.load('ResNet50 1x.pth', map_location='cpu')
            model.load_state_dict(state_dict_h['state_dict'])
        else:
            raise ValueError("Unknown ResNet variant")
        model.to(device)
        if model_name == "rn50simclr":
            preprocess = transforms.Compose([
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize(224, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return model, preprocess, device

    # -------- HuggingFace ViT‑family ------------------------------------
    from transformers import (
        AutoImageProcessor, AutoProcessor, AutoModel,
        Dinov2Model, ViTMAEModel, ViTModel,
    )

    # webssl models - https://arxiv.org/pdf/2504.01017
    # from - https://huggingface.co/facebook/webssl-dino7b-full8b-518 etc
    dino_webssl = {
        "dino_webssl_1b": "facebook/webssl-dino1b-full2b-224",
        "dino_webssl_3b": "facebook/webssl-dino3b-full2b-224",
        "dino_webssl_7b": "facebook/webssl-dino7b-full8b-518",
    }
    # from - https://huggingface.co/facebook/ijepa_vitg16_22k
    # or https://huggingface.co/facebook/ijepa_vith14_1k
    ijepa = {
        "ijepa_vitg16_22k": "jmtzt/ijepa_vitg16_22k",
        "ijepa_vith14_1k": "jmtzt/ijepa_vith14_1k",
    }
    # from - https://huggingface.co/facebook/vit-mae-huge
    # or https://huggingface.co/facebook/vit-mae-base 
    # or https://huggingface.co/facebook/vit-mae-large
    # from https://huggingface.co/facebook/webssl-mae3b-full2b-224 etc
    mae = {
        "mae_vitb": "facebook/vit-mae-base",
        "mae_vitl": "facebook/vit-mae-large",
        "mae_vith": "facebook/vit-mae-huge",
        "mae_webssl_300m": "facebook/webssl-mae300m-full2b-224",
        "mae_webssl_1b": "facebook/webssl-mae1b-full2b-224",
        "mae_webssl_3b": "facebook/webssl-mae3b-full2b-224",
    }
    # from - https://huggingface.co/google/siglip2-giant-opt-patch16-384
    # or https://huggingface.co/google/siglip2-large-patch16-256
    # or https://huggingface.co/google/siglip2-base-patch16-224
    siglip2 = {
        "siglip2_g": "google/siglip2-giant-opt-patch16-384",
        "siglip2_l": "google/siglip2-large-patch16-256",
        "siglip2_b": "google/siglip2-base-patch16-224",
    }
    # dinov3 from https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009
    dinov3 = {
        "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3_convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
        "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    }

    if model_name in dino_webssl:
        proc = AutoImageProcessor.from_pretrained(dino_webssl[model_name])
        model = Dinov2Model.from_pretrained(dino_webssl[model_name]).to(device)
        preprocess = lambda img: proc(img, return_tensors="pt")["pixel_values"].squeeze(0)
    elif model_name in ijepa:
        proc = AutoProcessor.from_pretrained(ijepa[model_name])
        model = AutoModel.from_pretrained(ijepa[model_name]).to(device)
        preprocess = lambda img: proc(img, return_tensors="pt")["pixel_values"].squeeze(0)
    elif model_name in mae:
        proc = AutoImageProcessor.from_pretrained(mae[model_name])
        if model_name.startswith("mae_webssl"):
            model = ViTModel.from_pretrained(mae[model_name], use_safetensors=True).to(device)
        else:
            model = ViTMAEModel.from_pretrained(mae[model_name], use_safetensors=True).to(device)
        preprocess = lambda img: proc(img, return_tensors="pt")["pixel_values"].squeeze(0)
    elif model_name in siglip2:
        proc = AutoProcessor.from_pretrained(siglip2[model_name])
        model = AutoModel.from_pretrained(siglip2[model_name]).to(device)
        preprocess = lambda img: proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
    elif model_name in dinov3:
        # proc = AutoImageProcessor.from_pretrained(dinov3[model_name])
        model = AutoModel.from_pretrained(dinov3[model_name]).to(device)
        def _fallback_preprocess(image_size=224):
            # DINOv3 (LVD-1689M) eval transform from the official README:
            # Resize to a square -> float32 [0,1] -> normalize (ImageNet stats).
            return v2.Compose([
                v2.ToImage(),                                # accepts PIL or ndarray
                v2.Resize((image_size, image_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),       # [0,1] float32
                v2.Normalize(mean=(0.485, 0.456, 0.406),
                            std =(0.229, 0.224, 0.225)),
            ])
        # preprocess = lambda img: proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        preprocess = _fallback_preprocess()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if hasattr(model, "config") and hasattr(model.config, "output_hidden_states"):
        model.config.output_hidden_states = False
    return model, preprocess, device

#############################################################################
# 4.  Layer selection
#############################################################################

def select_layers(model_name: str, model) -> List[str]:
    names = [n for n, _ in model.named_modules() if len(n) > 0]
    print('\nAll layer names:', names, '\n')

    # if model_name.startswith("DG"):
    #     # relu, softplus, and gauss used as activation functions in DG3
    #     relus = [n for n in names if "relu" in n]
    #     softplus = [n for n in names if "softplus1" in n]
    #     gauss = [n for n in names if "gauss" in n]
    #     layers = relus + softplus + gauss
    #     print(f"▶ Hooking {len(layers)} DG3 layers")
    #     return layers

    # if model_name.startswith("rn") or "resnet" in model.__class__.__name__.lower():
    #     # relu used as activation function in ResNets (fc exists for the coco models), avgpool is final layer
    #     relus  = [n for n in names if n.endswith("relu")]
    #     finals = [n for n in names if n.endswith("avgpool") or n == "fc"]
    #     all_fcs = [n for n in names if "fc." in n] if 'coco' in model_name else []
    #     layers = relus + finals + all_fcs
    #     print(f"▶ Hooking {len(layers)} RN layers")
    #     return layers

    # if 'dino' in model_name and 'webssl' not in model_name and 'v3' not in model_name:
    #     blocks = [n for n in names if re.search(r'(?:^|\.)blocks?\.\d+$', n)] # output of each transformer block
    #     finals = names[-2:]  # 'norm' and 'head'
    #     layers = ['patch_embed'] + blocks + finals
    #     print(f"▶ Hooking {len(layers)} DINOv2 layers")
    #     return layers
    
    # if ('dino' in model_name and 'webssl' in model_name) or 'mae' in model_name or 'ijepa' in model_name:
    #     # output of each encoder block + final norm
    #     encoders = [n for n in names if re.search(r'(?:^|\.)encoder\.layer\.\d+$', n)]
    #     finals = [n for n in names if re.search(r'(?:^|\.)layernorm$', n)]
    #     layers = encoders + finals
    #     if 'mae' in model_name and 'webssl' in model_name:
    #         layers += names[-1] # 'pooler.activation'
    #     layers = ['embeddings.patch_embeddings'] + [n for n in names if n in layers]
    #     print(f"▶ Hooking {len(layers)} DINO webssl layers")
    #     return layers

    # if 'clip' in model_name and 'vit' not in model_name:
    #     # output of each relu + final pooling, etc
    #     relus  = [n for n in names if n.endswith("relu")]
    #     finals  = [n for n in names if n=='visual.attnpool']
    #     layers = relus + finals
    #     print(f"▶ Hooking {len(layers)} CLIP RN layers")
    #     return layers
    
    # if 'clip' in model_name and 'vit' in model_name:
    #     # output of each transformer block MLP + final norm/proj
    #     blocks = [n for n in names if re.search(r'(?:^|\.)visual\.transformer\.resblocks\.\d+$', n)]
    #     finals = [n for n in names if n=='visual.ln_post']
    #     layers = ['visual.ln_pre'] + blocks + finals
    #     print(f"▶ Hooking {len(layers)} CLIP ViT layers")
    #     return layers
    
    # if 'siglip' in model_name:
    #     # output of each transformer block MLP + final head
    #     encoders = [n for n, _ in model.named_modules() if re.search(r'(?:^|\.)vision_model\.encoder\.layers\.\d+$', n)]
    #     finals = [n for n in names if n=='vision_model.head']
    #     layers = ['vision_model.embeddings.patch_embedding'] + encoders + finals
    #     print(f"▶ Hooking {len(layers)} Siglip2 layers")
    #     return layers

    # print("▶ Falling back to last module only")
    # return [names[-1]]

    print(f"▶ Hooking {len(names)} layers")
    return names

#############################################################################
# 5.  Forward‑hook extraction
#############################################################################

def extract_intermediates(model: torch.nn.Module, x: torch.Tensor, module_filter: List[str], device: str) -> Dict[str, torch.Tensor]:
    acts: Dict[str, torch.Tensor] = {}
    hooks = []
    bs = x.shape[0]

    def mk(name: str):
        if len(name) == 0:
            name = f"unnamed_module_{np.random.randint(0,100)}"
            print(f"Warning: empty module name, renaming to {name}")
        def _hook(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(t, torch.Tensor):
                print(f"Warning: {name} output is not a tensor, skipping")
                return
            if t.shape[0] != bs: # in case batch dim is not first
                print(f"Warning: {name} output batch size {t.shape[0]} != expected {bs}, trying second dim")
                if t.shape[1] == bs:
                    t = t.permute(1, 0, *range(2, t.ndim))
                    print(f"  • {name} permuted to put batch dim first")
                    acts[name] = t.detach().cpu().reshape(bs, -1)
                else:
                    print(f"Error: {name} output batch size not found in first two dims, skipping")
            #     bdim = list(t.shape).index(bs)
            #     t = t.permute([bdim] + [d for d in range(t.ndim) if d != bdim])
            # acts[name] = t.detach().cpu().reshape(bs, -1).to(torch.float16)
            else:
                acts[name] = t.detach().cpu().reshape(bs, -1)
        return _hook

    for n, m in model.named_modules():
        if n in module_filter:
            hooks.append(m.register_forward_hook(mk(n)))
    with torch.inference_mode():
        if args.model.startswith("clip"):
            _ = model.encode_image(x)
        elif 'ijepa' in args.model or 'mae' in args.model:
            _ = model(pixel_values=x)
        elif 'siglip' in args.model:
            _ = model.get_image_features(pixel_values=x)
        elif 'DG' in args.model:
            img_batch, cb, x_hist, y_hist = preprocess_batch(x, model, device)
            _ = model(img_batch, cb, x_hist, y_hist)
        else:
            _ = model(x)

    for h in hooks:
        h.remove()

    return acts

#############################################################################
# 6.  Helper functions
#############################################################################

def row_center_norm(a: np.ndarray) -> np.ndarray:
    """Row-centre and ℓ2-normalise each sample (row)."""
    a = a - a.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1, keepdims=True) + 1e-6
    return a / denom

# ---------------------------------------------------------------------
# 7.  Block-wise condensed RDM ----------------------------------------
# ---------------------------------------------------------------------

def rdm_condensed_from_h5_chunked(h5, key, device="cuda", feat_block=32768, row_block=512):
    """
    Float32, all features. CPU passes for mean/norm; GPU-tiled GEMMs for speed.
    Memory-safe: never materializes [N x F] on GPU; only row-tiles per feature-chunk.
    """
    import numpy as np, torch
    from scipy.spatial.distance import squareform

    ds = h5[key]                          # [N, F] on disk
    N = ds.shape[0]
    ds = ds.reshape(N,-1) # in case ds was not squeezed properly
    eps = 1e-6

    # Pass 0: remove features that have near-zero variance
    feat_std = ds[:].std(axis=0)
    ds = ds[:, feat_std > eps]
    N, F = int(ds.shape[0]), int(ds.shape[1])
    print(f"▶ Using {F} features in {key} after filtering zero-variance ones")

    # ---- Pass 1: row means (CPU, feature-chunked) ----
    row_sum = np.zeros(N, dtype=np.float64)
    total_F = 0
    for f0 in range(0, F, feat_block):
        f1 = min(f0 + feat_block, F)
        X = ds[:, f0:f1].astype(np.float32)          # [N, fb]
        row_sum += X.sum(axis=1, dtype=np.float64)
        total_F += X.shape[1]
    mu = (row_sum / max(total_F, 1)).astype(np.float32)  # [N]

    # ---- Pass 2: row norms after centering (CPU, feature-chunked) ----
    row_norm2 = np.zeros(N, dtype=np.float64)
    for f0 in range(0, F, feat_block):
        f1 = min(f0 + feat_block, F)
        X = ds[:, f0:f1].astype(np.float32)          # [N, fb]
        X -= mu[:, None]
        row_norm2 += (X * X).sum(axis=1, dtype=np.float64)
    row_norm = np.sqrt(row_norm2).astype(np.float32)  # [N]

    # ---- Pass 3: accumulate Gram in GPU row-tiles across feature-chunks ----
    G = np.zeros((N, N), dtype=np.float32)           # stays on CPU
    use_gpu = (device.startswith("cuda") and torch.cuda.is_available())
    dev = torch.device(device if use_gpu else "cpu")

    for f0 in range(0, F, feat_block):
        f1 = min(f0 + feat_block, F)
        Xfull = ds[:, f0:f1].astype(np.float32)      # CPU [N, fb]
        # center & normalize per row (CPU)
        Xfull -= mu[:, None]
        Xfull /= (row_norm[:, None] + eps)

        # reuse Ai for each j-tile
        for i0 in range(0, N, row_block):
            i1 = min(i0 + row_block, N)
            Ai = torch.from_numpy(Xfull[i0:i1, :]).to(dev, non_blocking=False)
            for j0 in range(i0, N, row_block):
                j1 = min(j0 + row_block, N)
                Bj = torch.from_numpy(Xfull[j0:j1, :]).to(dev, non_blocking=False)
                Gb = (Ai @ Bj.T).float().cpu().numpy()   # [ri, rj]
                G[i0:i1, j0:j1] += Gb
                if j0 != i0:
                    G[j0:j1, i0:i1] += Gb.T
            del Ai
            if use_gpu:
                torch.cuda.synchronize()
        del Xfull
        if use_gpu:
            torch.cuda.empty_cache()

    # corr -> distance, condensed
    np.fill_diagonal(G, 1.0)
    return squareform(1.0 - G, checks=False).astype(np.float32)

def rdm_condensed_from_h5(h5: h5py.File, key: str, device: str = 'cuda', force_avgpool: bool = False) -> np.ndarray:
    """Return the SciPy-style condensed correlation-distance vector for layer *key*."""
    ds = h5[key]                         # shape: [N, F]
    ds = ds[()]
    N  = ds.shape[0]
    if force_avgpool:
        print(ds.shape) # avgpool hack
        ds = ds.mean(axis=1)
        ds = ds.mean(axis=1)
        print(ds.shape)
    ds = ds.reshape(N,-1) # in case ds was not squeezed properly
    out = np.empty(N * (N - 1) // 2, dtype=np.float32)

    use_features = ds.std(axis=0) > 1e-6
    if not use_features.any():
        print(f"Warning: No features in {key} with std > 1e-6, returning empty RDM")
        return
    print(f"▶ Using {use_features.sum()} features in {key}")

    ds = torch.Tensor(ds[:, use_features]).to(device)

    # row‐center & normalize so corr(u,v) = u_norm·v_norm
    mean = ds.mean(dim=1, keepdim=True)
    ds = ds - mean
    norm = ds.norm(dim=1, keepdim=True)
    zero_rows = norm.squeeze(1) < 1e-12
    ds = ds / norm.clamp_min(1e-12)

    # full correlation matrix = X @ X^T
    #  then correlation‐distance = 1 – corr
    corr = ds @ ds.t()
    corr.clamp_(-1.0, 1.0)
    dist = 1.0 - corr

    if zero_rows.any():
        zr = zero_rows.nonzero(as_tuple=True)[0]
        dist[zr, :] = 1.0
        dist[:, zr] = 1.0
        dist[zr, zr] = 0.0

    dist = dist.cpu().numpy()  # move to CPU for squareform
    dist = dist + dist.T  # make symmetric
    np.fill_diagonal(dist,0)  # set diagonal to 0

    # dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    out = squareform(dist)

    print(f"▶ RDM for {key} has shape {out.shape} and dtype {out.dtype}")

    return out

# ---------------------------------------------------------------------
# 8.  Main -------------------------------------------------------------
# ---------------------------------------------------------------------

def main():

    print(f"▶ Extracting activations for model {args.model} on {args.split} split, extent={args.extent}")

    if ('mpnet_blt' not in args.model) and ('multihot_blt' not in args.model):
        
        model, preprocess, device = load_model_and_preprocess(args.model)
        model.eval()

        h5_path  = Path(args.out_dir) / f"{args.model}_{args.extent}_{args.split}_acts.h5" if args.extent != "full" else Path(args.out_dir) / f"{args.model}_{args.split}_acts.h5"
        rdm_path = Path(args.out_dir) / f"{args.model}_{args.extent}_{args.split}_RDMs.pkl" if args.extent != "full" else Path(args.out_dir) / f"{args.model}_{args.split}_RDMs.pkl"

        dataset, loader = get_dataloader(args.split, args.in_memory, preprocess, extent=args.extent)
        layers          = select_layers(args.model, model)
        N_total         = len(dataset)

        # ---------- prime HDF5 by running first batch ---------------------
        first_imgs = next(iter(loader))

        with torch.no_grad(): 
            first_acts = extract_intermediates(model, first_imgs.to(device), layers, device)

        h5 = h5py.File(h5_path, "w")
        for k, v in first_acts.items():
            # check if v is empty
            if v.numel() == 0:
                print(f"Warning: Layer {k} produced empty activations, skipping")
                continue
            F = v.shape[1]
            h5.create_dataset(k, shape=(N_total, F), dtype="float32", chunks=(args.batch_size, F))
            h5[k][0:v.shape[0]] = v
        written = v.shape[0]

        del first_imgs, first_acts, v
        torch.cuda.empty_cache(); gc.collect()

        # ---------- iterate over remaining batches ------------------------
        loader_iter = iter(loader)
        next(loader_iter)  # we consumed batch 0
        for b_idx, imgs in enumerate(loader_iter, start=1):
            imgs = imgs.to(device)
            with torch.no_grad():
                acts = extract_intermediates(model, imgs, layers, device)
            for k, v in acts.items():
                h5[k][written: written + v.shape[0]] = v
            written += v.shape[0]
            print(f"▶ {written}/{N_total} images processed")
            del imgs, acts, v
            torch.cuda.empty_cache(); gc.collect()

        assert written == N_total, "Dataset size mismatch during write"

        force_avgpool = False

    else:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if args.model == "mpnet_blt":
            if args.extent == "full":
                h5_path_existing = "/share/klab/adoerig/adoerig/nsd_visuo_semantics_cleanupJul24/nsd_visuo_semantics/examples/dnn_extracted_activities/mpnet_rec_seed1_nsd_SPECIAL515_NoSpaceAvg_activations_epoch200.h5"
            elif args.extent == "NSD_crop":
                h5_path_existing = "/share/klab/adoerig/adoerig/nsd_visuo_semantics_cleanupJul24/nsd_visuo_semantics/examples/dnn_extracted_activities/mpnet_rec_seed1_nsd_SPECIAL515_NoSpaceAvg_activations_epoch200_centralCrop91.h5"
            print(f"Loading precomputed activations from {h5_path_existing}")

            h5 = h5py.File(h5_path_existing, "r")
            layers = list(h5.keys())
            print(f"Layers found: {layers}")
            rdm_path = Path(args.out_dir) / f"{args.model}_{args.extent}_{args.split}_RDMs.pkl"

            force_avgpool = True # apparently otherwise it doesn't read out the last layer's avgpool properly...

    # import pdb; pdb.set_trace()  # Debugging breakpoint

    # ---------- compute RDMs -----------------------------------------
    RDMs = {}
    for k in layers:
        if k not in h5:
            print(f"⚠️  Skipping RDM for {k} as no activations were recorded")
            continue
        print(f"▶ Computing RDM for {k}")
        if h5[k].shape[1] < 6000000:
            RDMs[k] = rdm_condensed_from_h5(h5, k, device, force_avgpool=force_avgpool)
        else: # use chunked version for very wide layers
            RDMs[k] = rdm_condensed_from_h5_chunked(h5, k, device)
        # print(f"   • condensed size = {RDMs[k].shape[0]}")

    with open(rdm_path, "wb") as f:
        pickle.dump(RDMs, f)
    print(f"✔ RDMs written → {rdm_path}")

    h5.close()
    if h5_path.exists():
        os.remove(h5_path)
        print(f"🗑 Deleted activations file {h5_path}")

if __name__ == "__main__":
    main()

# helpers/otf_extractor.py

import math
import io
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from scipy.ndimage import zoom

import deepgaze_pytorch
from pysaliency.models import sample_from_logdensity


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
            history.append(-1) # to align with Varun's code
    return history

class OnTheFlyExtractor(nn.Module):
    """
    Given a batch of raw scene images (uint8 array, 256×256x3), produces:
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
        r50v: int = 1,                  # ResNet50 version 0/1/2
        n_fixs: int = 7,                # number of fixations to sample
        device: str = "cuda",           # where GPN will live
        dg3_device: str = "cuda",       # where DG3 will run
        centerbias_path: str = "centerbias_mit1003.npy",
        seed: int = 42,
        rn50_mb: int = 512,              # micro-batch size for ResNet50
        dg3_mb: int = 64,                # micro-batch size for DeepGazeIII - 90 max on 1 L40S
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
                      .to(torch.float32).to(self.dg3_dev).eval()
        cb = np.load(centerbias_path)                        # [1024×1024]
        cb -=  logsumexp(cb)                              # normalize log‐density
        # rescale to dg3_res
        cb = zoom(cb, (dg3_res/cb.shape[0], dg3_res/cb.shape[0]), order=0, mode='nearest') # [dg3_res×dg3_res] - as done in DG3 example
        # will broadcast inside DG3
        self.cb = torch.tensor([cb]).to(torch.float32).to(self.dg3_dev) # [1,dg3_res,dg3_res]
        self._rng = np.random.RandomState(seed)
        self.dg3_prep = transforms.Resize(dg3_res, antialias=True)

        # ─── ResNet-50 setup (FP16) ────────────────────────────────────────
        weights = {0: None,
                   1: ResNet50_Weights.IMAGENET1K_V1,
                   2: ResNet50_Weights.IMAGENET1K_V2}[r50v]
        self.net = resnet50(weights=weights)
        self._acts = {} # remember to empty the list after each forward pass
        def get_activation(name):
            def hook(model, input, output):
                self._acts[name] = output.detach()
            return hook
        self.net.avgpool.register_forward_hook(get_activation('avgpool'))
        self.net = self.net.to(device).eval()
        self.prep = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])

        print(f"OnTheFlyExtractor: ResNet50v{r50v} on {device}, DeepGazeIII on {dg3_device}, {n_fixs} fixations")

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
            inp = self.prep(chunk).to(self.device)   
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
            up = self.dg3_prep(sub).to(self.dg3_dev)  # [b,3,dg3_res,dg3_res]
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

            print(f"Processed {min(i+self.dg3_mb,B)}/{B} images for DG3 fixation sampling")

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
            inp = self.prep(c).to(self.device)
            with torch.no_grad():
                _ = self.net(inp)
                h = self._acts.pop("avgpool")  # [b,2048,1,1]
            all_actvs.append(h.squeeze(-1).squeeze(-1))

            print(f"Processed {min(i+self.rn50_mb,crops.size(0))}/{crops.size(0)} glimpses for ResNet50 activations")

        all_actvs = torch.cat(all_actvs, 0)     # [B*T,2048]
        fix_actvs = all_actvs.view(B, self.n_fixs, 2048).cpu().numpy() 
        # print(fix_actvs.shape)

        return fix_actvs, next_abs, next_rel, scene_embed
# GPN/GSN training
Codebase for training Glimpse Prediction/Stitching Networks.

## Codebase map
1. Run [train_net.py](train_net.py) with the appropriate hyperparameters to train GPNs/GSNs — the default setting will train a GPN-R-SimCLR.
2. The trained models will be saved under logs/net_params

## Requirements
1. In [train_net.py](train_net.py), on line 63, include the path to the folder which contains the glimpse sequences dataset.
# GPN/GSN training
Codebase for training Glimpse Prediction/Stitching Networks.

## Codebase map
1. Run [train.py](train.py) with the appropriate hyperparameters to train GPNs/GSNs — the default setting will train a GPN-R-SimCLR.
2. The trained models will be saved under logs/net_params

## Requirements
1. In [train.py](train.py), on line 63, include the path to the folder which contains the glimpse sequences dataset.
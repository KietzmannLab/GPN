# GPN/GSN training
Codebase for extracting the RDMs for the alternative models.

## Codebase map
1. Run [extract_coco_rdms_actvs.py](extract_coco_rdms_actvs.py) with the appropriate hyperparameters to extract RDMs from pretrained vision/vision-language models — the default setting will extract RDMs from an EfficientNet-B3 given central crops. The RDMs will be saved under ../rdms 
3. Run [extract_best_network_corr.py](extract_best_network_corr.py) to find the layer per network whose RDM best aligns with a given ROI RDM, across the streams ROIs. The result will be saved in 'best_layers_streams_allrois.pkl'

## Requirements
1. In [extract_coco_rdms_actvs.py](extract_coco_rdms_actvs.py), on line 90, include the path to the folder which contains the test set indices of special-515 images (provided in this folder).
2. On line 123, include the path to h5 file containing the h5 dataset containing DeepGaze3 fixations.
3. On line 148, adjust the 'centerbias_mit1003.npy' path (file provided in this folder).
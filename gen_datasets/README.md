# GPN/GSN training
Codebase for generating the glimpse sequences datasets.

## Codebase map
1. Run [gen_actvs_fix_data.py](gen_actvs_fix_data.py) with the appropriate hyperparameters to generate glimpse sequences, given the chosen backbone — the default setting will extract the glimpses and their embeddings on the 'test' split with a RN50-SimCLR backbone.
2. Run [gen_actvs_fix_data_test_515.py](gen_actvs_fix_data_test_515.py) with the appropriate hyperparameters to generate the 'test-515' split datasets for analyses.

## Requirements
1. Assuming access to the COCO scenes and MPNet embeddings of scene captions in h5 format, and access to DeepGaze3 extracted fixations on these scenes. Please contact Prof. Tim Kietzmann for access to these large datasets in h5 format.
2. In [gen_actvs_fix_data.py](gen_actvs_fix_data.py), on line 33, include the path to the folder which should contain the glimpse sequences dataset.
3. In [gen_actvs_fix_data_test_515.py](gen_actvs_fix_data_test_515.py), change all the paths to point to the folder which contains the glimpse sequences dataset.
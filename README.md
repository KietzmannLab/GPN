# Glimpse Prediction Networks
Official codebase of the GPN project.

## Codebase map
1. The figures included in the paper can be plotted using [paper_plots.ipynb](paper_plots.ipynb) and [gpn_analysis.ipynb](gpn_analysis.ipynb)
2. GPN/GSN training scripts can be found under [train/](train/)
3. Scripts to generate the datasets of glimpse sequences can be found under [gen_datasets](gen_datasets/)
4. Scripts to extract the RDMs for alternative networks, and to find the best-ROI-aligned layers, can be found under [alternative_models](alternative_models/)

## Environment setup
The code has been tested with:
- Python 3.9.19 + PyTorch 2.3.1 (CPU) on macOS (Apple M1)
- Python 3.10.12 + PyTorch 2.5.1+cu124 on a Linux HPC cluster

You can setup the enviroment as:

```bash
conda env create -f environment.yml
conda activate gpn
```

## Pretrained networks and analysis data
1. Download the 'logs' folder from [OSF]() and place it under [train/](train/) — these contain the pretrained GPN/GSN weights and performance metrics.
2. Download 'saved_actvs', 'rdms', and 'datasets' from [OSF]() and place them at the same level as the figure plotting scripts before executing them — these contain analysis-relevant measures for the networks, their rdms, and the 'test-515' glimpse sequences and COCO images datasets, and the SCEGRAM dataset, required in [gpn_analysis.ipynb](gpn_analysis.ipynb)
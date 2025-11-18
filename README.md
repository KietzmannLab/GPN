# Glimpse Prediction Networks
Official codebase of the GPN project. Release v1. (archived at [OSF](https://doi.org/10.17605/OSF.IO/G29PD))

This repository is geared towards making all the code and analyses from the paper ([https://arxiv.org/abs/2511.12715](https://arxiv.org/abs/2511.12715)) possible. 

If instead you are interested in quickly accessing the best NSD-aligned GPN models (SimCLR backbone), check out the [HuggingFace](https://doi.org/10.57967/hf/6992) repository instead.

## Codebase map
1. The figures included in the paper can be plotted using [paper_plots.ipynb](paper_plots.ipynb) and [gpn_analysis.ipynb](gpn_analysis.ipynb)
2. The mapping between GPN/GSN variant/backbone names and their identifiers can be found in [paper_plots.ipynb](paper_plots.ipynb)
3. GPN/GSN training scripts can be found under [train](train/)
4. Scripts to generate the datasets of glimpse sequences can be found under [gen_datasets](gen_datasets/)
5. Scripts to extract the RDMs for alternative networks, and to find the best-ROI-aligned layers, can be found under [alternative_models](alternative_models/)

## Requirements

### Environment setup
The code has been tested with:
- Python 3.9.19 + PyTorch 2.3.1 (CPU) on macOS (Apple M1)
- Python 3.10.12 + PyTorch 2.5.1+cu118 on a Linux HPC cluster

Clone this repository (or download it directly):

```bash
git clone https://github.com/KietzmannLab/GPN.git
cd GPN
```

Setup your enviroment as:

```bash
conda env create -f environment.yml
conda activate gpn
```
In 'environment.yml', uncomment lines 5 and 30 when using a GPU and set the required cuda version.
CLIP and DeepGaze are not installed in this setup. If desired:
1. pip install "git+https://github.com/openai/CLIP.git"
2. Clone and install https://github.com/matthias-k/DeepGaze

### Pretrained networks and analysis data
1. Download the 'logs' folder from [OSF](https://doi.org/10.17605/OSF.IO/G29PD) and place it under [train/](train/) — these contain the pretrained GPN/GSN weights and performance metrics. (~3GB)
2. Download 'saved_actvs', 'rdms', and 'datasets' from [OSF](https://doi.org/10.17605/OSF.IO/G29PD) and place them at the same level as the figure plotting scripts before executing them — these contain analysis-relevant data from the networks, their rdms, and the 'test-515' glimpse sequences and COCO images datasets, and the SCEGRAM dataset, required in [gpn_analysis.ipynb](gpn_analysis.ipynb) (~40GB)
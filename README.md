# GPN
Official codebase of the Glimpse Prediction Networks project.

## Codebase map
1. The figures included in the paper can be plotted using [paper_plots.ipynb](paper_plots.ipynb) and [gpn_analysis.ipynb](gpn_analysis.ipynb).
2. GPN/GSN training scripts can be found under [train/](train/).


## Requirements

### Importing pretrained networks and analysis data
1. Download the 'logs' folder from [OSF]() and place it under [train/](train/) — these contain the pretrained GPN/GSN weights and performance metrics.
2. Download 'saved_actvs', 'rdms', and 'datasets' from [OSF]() and place them at the same level as the figure plotting scripts before executing them — these contain analysis-relevant measures for the networks, their rdms, and the 'test-515' glimpse sequences and COCO images datasets, and the SCEGRAM dataset, required in [gpn_analysis.ipynb](gpn_analysis.ipynb).
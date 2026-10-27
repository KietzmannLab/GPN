# Alternative-model RDMs

`extract_coco_rdms_actvs.py` extracts activations and representational dissimilarity matrices (RDMs) from comparison networks. Its default model is EfficientNet-B3, and its default output directory is `../rdms/` when run from this directory. `extract_best_network_corr.py` selects the best-aligned layer for each network and brain region using the extracted RDMs and neural RDMs in `../rdms/`.

Extraction requires the source COCO scenes, special-515 indices, fixation data for gaze-based models, and the selected network weights or package. Configure the cluster-specific source paths in `extract_coco_rdms_actvs.py` for your environment. The RDMs and layer-selection data used by the paper notebooks are supplied on [Zenodo](https://doi.org/10.5281/zenodo.20667384); see [Science paper data and models](../SCIENCE_PAPER.md).

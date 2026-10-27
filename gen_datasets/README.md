# Glimpse-sequence datasets

`gen_actvs_fix_data.py` extracts glimpse crops and backbone embeddings from COCO scenes and DeepGaze 3 fixation traces. Its default arguments select the test split and RN50-SimCLR. `gen_actvs_fix_data_test_515.py` selects the special-515 scenes from those outputs for the analysis notebooks.

Generating these files requires the full source COCO scenes, caption embeddings, fixation traces, and special-515 indices. Configure the source, output, and backbone-weight paths in the scripts for your environment. The published test-515 datasets are supplied on [Zenodo](https://doi.org/10.5281/zenodo.20667384); see [Science paper data and models](../SCIENCE_PAPER.md) for the download map and backbone IDs.

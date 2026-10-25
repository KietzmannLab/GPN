# Glimpse Prediction Networks

Code and analyses for [Thorat et al., *Science Advances* (2026)](https://doi.org/10.1126/sciadv.aee1955). Pretrained GPNs are also available on [Hugging Face](https://huggingface.co/novelmartis/GPN).

## Quick start

```bash
git clone https://github.com/KietzmannLab/GPN.git
cd GPN
conda env create -f environment.yml
conda activate gpn-science
```

Before running the notebook, download the base data and all `science-paper-rdms-*.zip` files from the [GPN data record (all versions)](https://doi.org/10.5281/zenodo.20667384). Extract them **into the repository root**. The base record provides `datasets.zip`, `logs.zip`, and split `rdms.part_*` and `saved_actvs.part_*` archives. The Science Advances ZIPs extract individually. After downloading the files into this directory:

```bash
cat rdms.part_* > rdms.zip
cat saved_actvs.part_* > saved_actvs.zip
unzip -q datasets.zip
unzip -q logs.zip -d train
unzip -q rdms.zip
unzip -q saved_actvs.zip
for archive in science-paper-rdms-*.zip; do unzip -oq "$archive"; done
```

The resulting folders include `rdms/`, `saved_actvs/`, `datasets/`, and `train/logs/`. Open the summary notebook:

```bash
jupyter notebook paper_plots.ipynb
```

Both notebooks include saved default outputs that can be viewed on GitHub without setup. Use **Kernel → Restart & Run All** to reproduce the paper-wide figures and summary statistics in `figures/`. The notebook uses saved analysis data; plotting does not train models or download backbone weights.

To inspect one network, open [gpn_analysis.ipynb](gpn_analysis.ipynb), choose its name in the first code cell, and Run All. The default preview uses `GPN-RS-SimCLR`. To run a model absent from the original archive, also extract all `science-paper-models-*.zip` and `science-paper-test-embeddings-*.zip` files:

```bash
for archive in science-paper-models-*.zip science-paper-test-embeddings-*.zip; do unzip -oq "$archive"; done
```

The notebook loads cached results when available and identifies optional analyses that need recomputation.

The model notebook selects CUDA, then MPS, then CPU automatically, so Apple Silicon runs the network on its GPU. Set `model_device = 'cpu'` in the first code cell when comparing newly computed values with the saved CPU reference; GPU floating-point results can differ slightly. The training script retains its CUDA default.

The data bundles, model names, and optional training commands are described in [Science paper data and models](SCIENCE_PAPER.md). Differences between the published text and the available analysis code/data are recorded in [Paper audit](PAPER_AUDIT.md).

## Repository map

- [paper_plots.ipynb](paper_plots.ipynb): results across networks and brain regions.
- [gpn_analysis.ipynb](gpn_analysis.ipynb): behavior and representations of one selected model.
- [train/](train/): model definition and training.
- [gen_datasets/](gen_datasets/): glimpse-sequence embedding generation.
- [analysis_helpers/](analysis_helpers/): RSA, model identifiers, and plotting helpers.
- [alternative_models/](alternative_models/): alternative-network extraction and layer selection.

The pinned environment targets Python 3.10 and PyTorch 2.7.1 on macOS. The published models were trained with PyTorch 2.5.1 and CUDA on the cluster; for new GPU training, adjust the PyTorch and CUDA lines in [environment.yml](environment.yml). Optional CLIP and DeepGaze installations are described in the relevant helper scripts.

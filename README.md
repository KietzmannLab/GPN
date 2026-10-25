# Glimpse Prediction Networks

Code and analyses for [Thorat et al., *Science Advances* (2026)](https://doi.org/10.1126/sciadv.aee1955). The four SimCLR GPNs and two GSN controls are also available on [Hugging Face](https://huggingface.co/novelmartis/GPN) for quick inference. S/RS use saccade inputs; B/R and both GSNs use zero saccade tensors, as in training.

## Quick start

```bash
git clone https://github.com/KietzmannLab/GPN.git
cd GPN
conda env create -f environment.yml
conda activate gpn-science
```

For the summary notebook, download all `rdms.part_*` files, `science-paper-analysis-cache.zip`, and all `science-paper-rdms-*.zip` files from the [GPN data record (all versions)](https://doi.org/10.5281/zenodo.20667384). Extract them **into the repository root**. The Science Advances ZIPs extract individually:

```bash
cat rdms.part_* > rdms.zip
unzip -q rdms.zip
unzip -oq science-paper-analysis-cache.zip
for archive in science-paper-rdms-*.zip; do unzip -oq "$archive"; done
```

Open the summary notebook:

```bash
jupyter notebook paper_plots.ipynb
```

Both notebooks include saved default outputs that can be viewed on GitHub without setup. Use **Kernel → Restart & Run All** to reproduce the paper-wide figures and summary statistics in `figures/`. The notebook uses saved analysis data; plotting does not train models or download backbone weights.

To inspect the default `GPN-R-SimCLR` with its exact saved version11 activations, also download `datasets.zip`, `logs.zip`, and `science-paper-core-activations-GPN-R-SimCLR.zip` and extract them into the repository root:

```bash
unzip -q datasets.zip
unzip -q logs.zip -d train
unzip -oq science-paper-core-activations-GPN-R-SimCLR.zip
```

Then open [gpn_analysis.ipynb](gpn_analysis.ipynb) and Run All. To select any of the 46 published checkpoints, also extract these archives:

```bash
for archive in science-paper-core-models-*.zip science-paper-models-*.zip science-paper-test-embeddings-*.zip; do unzip -oq "$archive"; done
```

For the exact saved activations of another SimCLR or ILSVRC B/S/R/RS model, extract its `science-paper-core-activations-GPN-<variant>-<backbone>.zip` before running the notebook. The notebook generates activations from the checkpoint and test embeddings when that cache is absent. Optional analyses that need additional source images or backbone packages are labeled in the notebook.

The model notebook selects CUDA, then MPS, then CPU automatically, so Apple Silicon runs the GPN on its GPU. The DINOv2 image-embedding helper uses CPU, as in version11. Set `model_device = 'cpu'` in the first code cell when comparing newly computed values with the saved CPU reference; GPU floating-point results can differ slightly. The training script retains its CUDA default.

The data bundles, model names, and optional training commands are described in [Science paper data and models](SCIENCE_PAPER.md). Differences between the published text and the available analysis code/data are recorded in [Paper audit](PAPER_AUDIT.md).

## Repository map

- [paper_plots.ipynb](paper_plots.ipynb): results across networks and brain regions.
- [gpn_analysis.ipynb](gpn_analysis.ipynb): behavior and representations of one selected model.
- [train/](train/): model definition and training.
- [gen_datasets/](gen_datasets/): glimpse-sequence embedding generation.
- [analysis_helpers/](analysis_helpers/): RSA, model identifiers, and plotting helpers.
- [alternative_models/](alternative_models/): alternative-network extraction and layer selection.

The pinned environment targets Python 3.10 and PyTorch 2.7.1 on macOS. The published models were trained with PyTorch 2.5.1 and CUDA on the cluster; for new GPU training, adjust the PyTorch and CUDA lines in [environment.yml](environment.yml). Optional CLIP and DeepGaze installations are described in the relevant helper scripts.

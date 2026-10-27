# Glimpse Prediction Networks

Code and analyses for [Thorat et al., *Science Advances* (2026)](https://doi.org/10.1126/sciadv.aee1955). The four SimCLR GPNs and two GSN controls are also available on [Hugging Face](https://huggingface.co/novelmartis/GPN) for quick inference. S/RS use saccade inputs; B/R and both GSNs use zero saccade tensors, as in training.

## Quick start

```bash
git clone https://github.com/KietzmannLab/GPN.git
cd GPN
conda env create -f environment.yml
conda activate gpn-science
```

For the default setup, download `datasets.zip`, `logs.zip`, `rdms.part_aa` through `rdms.part_ad`, `science-paper-analysis-cache.zip`, all 12 `science-paper-rdms-*.zip` files, `science-paper-core-activations-GPN-R-SimCLR.zip`, and `science-paper-core-activations-GPN-RS-ILSVRC.zip` from the [GPN data record](https://doi.org/10.5281/zenodo.20667384). This covers the summary notebook and exact saved activations for those two models in the single-model notebook.

Save these files directly in the cloned `GPN/` folder, open a macOS/Linux terminal there, and run the commands below. The download is about 24.2 GB; joining the RDM parts creates another 14.4 GB file, and extraction needs further free space. Join only the `rdms.part_*` files; each Science Advances ZIP extracts independently:

```bash
cat rdms.part_aa rdms.part_ab rdms.part_ac rdms.part_ad > rdms.zip
unzip -oq datasets.zip
unzip -oq logs.zip -d train
unzip -oq rdms.zip
unzip -oq science-paper-analysis-cache.zip
for archive in science-paper-rdms-*.zip; do unzip -oq "$archive"; done
unzip -oq science-paper-core-activations-GPN-R-SimCLR.zip
unzip -oq science-paper-core-activations-GPN-RS-ILSVRC.zip
```

Open the summary notebook:

```bash
jupyter notebook paper_plots.ipynb
```

Both notebooks include saved default outputs that can be viewed on GitHub without setup. Use **Kernel → Restart & Run All** to reproduce the paper-wide figures and summary statistics in `figures/`. The notebook uses saved analysis data; plotting does not train models or download backbone weights.

Then open [gpn_analysis.ipynb](gpn_analysis.ipynb) and Run All for its default `GPN-R-SimCLR`. Change its model-selection cell to inspect another network. The B/S/R/RS SimCLR and ILSVRC checkpoints and test embeddings are in `logs.zip` and `datasets.zip`; the notebook computes and saves activations if that model's exact cache is absent. For the 32 GPNs with other backbones, download only the checkpoint and test-embedding ZIPs listed for that model in [DATA_ASSETS.json](DATA_ASSETS.json); extract each into the repository root. Download `science-paper-additional-logs.zip` if you want those models' training histories. Recomputed activations can differ slightly from the supplied saved activations because of device arithmetic and test-loader shuffling; the summary notebook uses the supplied published analysis cache and RDMs. See [Science paper data and models](SCIENCE_PAPER.md) for the full download map and optional backbone requirements.

The model notebook selects CUDA, then MPS, then CPU automatically, so Apple Silicon runs the GPN on its GPU. The DINOv2 image-embedding helper uses CPU. Set `model_device = 'cpu'` in the first code cell when comparing newly computed values with the saved CPU reference; GPU floating-point results can differ slightly. The training script uses CUDA by default.

The data bundles, model names, and optional training commands are described in [Science paper data and models](SCIENCE_PAPER.md). [Paper audit](PAPER_AUDIT.md) records exact numerical and method-description differences between the published text and available analysis code/data. These do not change the paper's main directional conclusions.

## Repository map

- [paper_plots.ipynb](paper_plots.ipynb): results across networks and brain regions.
- [gpn_analysis.ipynb](gpn_analysis.ipynb): behavior and representations of one selected model.
- [train/](train/): model definition and training.
- [gen_datasets/](gen_datasets/): glimpse-sequence embedding generation.
- [analysis_helpers/](analysis_helpers/): RSA, model identifiers, and plotting helpers.
- [alternative_models/](alternative_models/): alternative-network extraction and layer selection.

The pinned environment targets Python 3.10 and PyTorch 2.7.1 on macOS. The published models were trained with PyTorch 2.5.1 and CUDA on the cluster; for new GPU training, adjust the PyTorch and CUDA lines in [environment.yml](environment.yml). Optional CLIP and DeepGaze installations are described in the relevant helper scripts.

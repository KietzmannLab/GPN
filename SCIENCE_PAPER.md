# Science Advances data and model reference

Run the across-model analyses in [paper_plots.ipynb](paper_plots.ipynb). Choose one network in [gpn_analysis.ipynb](gpn_analysis.ipynb). Both notebooks expect to be run from the repository root.

## Data layout

All paths below are relative to the repository root. Get the archives from the [GPN Zenodo record (all versions)](https://doi.org/10.5281/zenodo.20667384). The original `rdms.part_*` files must be joined into `rdms.zip`; each `science-paper-*.zip` extracts independently. Extract the Science Advances RDM ZIPs after the original `rdms.zip` so the published analysis files take precedence.

| Archive | Files after extraction | Purpose |
| --- | --- | --- |
| `datasets.zip` | `datasets/coco-test-515.h5`, `datasets/coco_dg3_fixations_test_515.h5`, `datasets/coco_NSD_dg3fix91_r50v{1,5,6}ap_7fix_test_515.h5`, `datasets/SCEGRAM/` | Test scenes, fixations, and ILSVRC/DINOv2-B/SimCLR glimpse embeddings; SCEGRAM images and coordinates. |
| `logs.zip` (extract under `train/`) | `train/logs/net_params/<id>/<id>.pth`, `train/logs/perf_logs/` | Best checkpoints for the 12 core GPNs and two GSNs, plus original controls and training histories. |
| `rdms.part_*` (join as `rdms.zip`) | `rdms/` | Original comparison-network and other RDMs used by the notebooks. |
| `science-paper-rdms-*.zip` (12) | `rdms/gpn_rdms_<id>.pkl` for all 46 published models, `rdms/streams_all_neural_rdms_correlation_special515.pkl`, `analysis_helpers/best_layers_streams_allrois.pkl` | Model-to-brain comparisons and layer selection. |
| `science-paper-analysis-cache.zip` (1) | `saved_actvs/science_paper_behavior_cache.pkl`, `saved_actvs/best_network_layers.pkl`, small probe caches | Default behavior plots, position-decoding results, and GPN-R-SimCLR probes without the old large activation archive. |
| `science-paper-core-models-*.zip` (5) | 12 core GPN best checkpoints under `train/logs/net_params/`, and 46 loss histories under `train/logs/perf_logs/` | Standalone core checkpoint copies and complete training-history plots. The 12 best checkpoints are also in `logs.zip`. |
| `science-paper-models-*.zip` (13) | 32 additional GPN checkpoints under `train/logs/net_params/`, plus `dvd-b-565.pth` | Published GPNs using the eight additional backbones. |
| `science-paper-core-test-embeddings-*.zip` (2) | `datasets/coco_NSD_dg3fix91_r50v{1,5}ap_7fix_test_515.h5` | Separately packaged ILSVRC and DINOv2-B test embeddings; optional when `datasets.zip` is extracted. |
| `science-paper-test-embeddings-*.zip` (7) | `datasets/coco_NSD_dg3fix91_r50v<ID>ap_7fix_test_515.h5` for IDs 4 and 7–13 | Test embeddings for the eight additional backbones. |
| `science-paper-core-activations-GPN-<variant>-<backbone>.zip` (8) | `saved_actvs/<id>_repr.pkl` | Exact saved activations for B/S/R/RS with SimCLR or ILSVRC; optional when recomputing from a checkpoint and test embeddings. |

The `{1,5,6}` dataset filename notation in this table abbreviates three separate files. The full backbone IDs are below. Every published model name maps to its checkpoint `<id>` in [science_models.json](analysis_helpers/science_models.json). The same ID names its RDM and optional activation cache. `DATA_SHA256SUMS` checks the 48 Science Advances ZIPs; each also contains a `MANIFEST-*.json`. The original `saved_actvs.part_*` archive and `GPN_v1.zip` are not required by either notebook.

### Minimum files by task

| Task | Download |
| --- | --- |
| View the saved notebook outputs on GitHub | Nothing. |
| Rerun `paper_plots.ipynb` | Original `rdms.part_*`, `science-paper-rdms-*.zip`, and `science-paper-analysis-cache.zip`. No model checkpoint, test dataset, or backbone download is needed. |
| Run default `gpn_analysis.ipynb` with exact GPN-R-SimCLR activations | The summary files above, `datasets.zip`, `logs.zip`, and `science-paper-core-activations-GPN-R-SimCLR.zip`. |
| Run the single-model notebook across all 46 published models | The summary files, `datasets.zip`, `logs.zip`, all `science-paper-core-models-*.zip`, `science-paper-models-*.zip`, and `science-paper-test-embeddings-*.zip`. The eight activation ZIPs are optional shortcuts. |
| Train all 46 models from initialization | Requires the full training and validation embedding datasets, which these analysis archives do not supply. |

For your own glimpse images, [Hugging Face](https://huggingface.co/novelmartis/GPN) offers the four SimCLR GPNs and two matched GSNs without the Zenodo analysis archives. The GitHub notebooks use the Zenodo checkpoints and data. The selected GPN/GSN model needs its backbone's test-515 embedding file only when the notebook computes activations rather than loading a saved activation cache. Optional probes that extract embeddings from new images may also need the selected backbone's weights or package; the SCEGRAM source images are in `datasets.zip`.

## Additional backbone IDs

The existing training and dataset-generation interfaces accept these additional `--bbv` (training) / `--r50v` (generation) values. The `r50v<ID>` dataset filename convention is retained for all backbones.

| Backbone | ID | Features | Input dropout, B / S / R / RS |
| --- | ---: | ---: | --- |
| RN50-DVD-B | 4 | 2048 | .1 / .1 / .1 / .1 |
| EfficientNet-B1 | 7 | 1280 | .1 / .1 / .25 / .1 |
| ViT-B/32 | 8 | 768 | .1 / .1 / .25 / .25 |
| RN101-ILSVRC | 9 | 2048 | .1 / .1 / .25 / .1 |
| ConvNeXt-Tiny | 10 | 768 | .1 / .1 / .1 / .1 |
| Swin-T | 11 | 768 | .1 / .1 / .1 / .1 |
| MaxViT-T | 12 | 512 | .1 / .1 / .1 / .1 |
| CORnet-S | 13 | 512 | .1 / .1 / .1 / .1 |

LSTM dropout is .1 for these backbones. B/S use `--recurrence 0`; R/RS use `--recurrence 1`. S/RS use `--provide_loc 1`; B/R use `--provide_loc 0`. CORnet-S needs the optional [CORnet package](https://github.com/dicarlolab/CORnet) only when selected. The DVD-B backbone checkpoint in the models archive extracts to `dvd-b-565.pth`.

## cGSN training

The published cGSN used binary cross-entropy with logits and multihot scene labels. To train a new cGSN with that objective, explicitly add `--paper_cgsn_bce` with `--scene_loss 1`:

```bash
python train/train_net.py --bbv 6 --recurrence 1 --provide_loc 0 \
  --glimpse_loss 0 --semantic_loss 0 --scene_loss 1 --paper_cgsn_bce \
  --input_dropout 0.5 --rnn_dropout 0.25 --trainer train_515 --network_id 1
```

The flag creates separately named checkpoints and loss logs. The existing default training behavior and checkpoint identifiers remain available. Training requires the full train/validation datasets and a suitable GPU environment; the notebooks use published checkpoints and do not retrain.

## Figure entry points

| Result | Notebook section |
| --- | --- |
| Figs. 2–3; S1–S3; S7–S10 | `paper_plots.ipynb`: behavior, neural alignment, fixation controls |
| Fig. 4B; Figs. S6/S11 | `paper_plots.ipynb`: layers and backbone sweep |
| Single-model behavior, binding, and representations | `gpn_analysis.ipynb` |
| Fig. S5 searchlight | External Doerig et al. analysis code was used; not recomputed by these notebooks |

Fig. 1 is a schematic. Some qualitative panels were manually composed from model outputs. See [PAPER_AUDIT.md](PAPER_AUDIT.md) for checks and reporting differences.

## Verification

```bash
python -m unittest discover -s tests -v
```

The unit tests run offline. `GPN_PAPER_DATA_ROOT` enables checks of all 46 checkpoints when the archives are extracted. `GPN_SCIENCE_BASELINE` enables byte-for-byte comparison to the original-release regression fixtures. `GPN_CLUSTER_SOURCE` enables a direct cGSN loss/gradient comparison to the surviving cluster helper. The shared repository uses the paper notebooks for scientific figures; tests check code contracts and numerical compatibility.

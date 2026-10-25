# Science Advances data and model reference

Run the across-model analyses in [paper_plots.ipynb](paper_plots.ipynb). Choose one network in [gpn_analysis.ipynb](gpn_analysis.ipynb). Both notebooks expect to be run from the repository root.

## Data layout

Download the base data from the [GPN data record (all versions)](https://doi.org/10.5281/zenodo.20667384) and extract its `rdms/`, `saved_actvs/`, `datasets/`, and `train/logs/` directories into the repository root. The Science Advances version of the record supplies these additions:

| Archive | Contents | Needed for |
| --- | --- | --- |
| `science-paper-rdms-*.zip` (12 files) | RDMs for all 46 published GPN/GSN checkpoints, neural RDMs, layer-selection metadata | `paper_plots.ipynb` and cached neural analyses in `gpn_analysis.ipynb` |
| `science-paper-models-*.zip` (13 files) | Best checkpoints for 32 additional GPNs; DVD-B backbone weights | Opening and evaluating the additional models |
| `science-paper-test-embeddings-*.zip` (7 files) | Test-515 embeddings for eight additional backbones | Recomputing activations for the additional models |

The base archive supplies the remaining checkpoints and embeddings. Each Science Advances ZIP extracts directly into the repository root; no joining is needed. Extract the Science Advances RDM ZIPs after the base RDM archive so their RDMs take precedence. Download every file in a group before using that group. Confirm their SHA-256 values against the repository [DATA_SHA256SUMS](DATA_SHA256SUMS) file. Published model names map to checkpoint identifiers in [analysis_helpers/science_models.json](analysis_helpers/science_models.json).

Summary plotting reads stored RDMs and does not need pretrained backbone packages or network access. The single-model notebook loads cached activations when available and uses the chosen checkpoint/test embeddings when they are absent. Some optional model probes, including SCEGRAM image extraction, require their source images and backbone weights. They are labeled in the notebook.

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

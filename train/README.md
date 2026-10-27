# GPN and GSN training

Run `train_net.py` from this directory to train a model. Its default arguments specify GPN-R-SimCLR. Checkpoints and training histories are written under `logs/net_params/` and `logs/perf_logs/`.

Training requires the full COCO train/validation glimpse-embedding datasets and special-515 index files. The Zenodo analysis bundles supply test-515 data and published checkpoints, not the full training inputs. Configure `dataset_path` in `train_net.py` and the dataset and index paths in `helpers/helper_funcs.py` for your environment before training. The command for the published cGSN objective and the backbone IDs are documented in [Science paper data and models](../SCIENCE_PAPER.md).

# Reproducibility audit for the Science Advances paper

This file connects the released code and data to the paper's results and records claims that do not reproduce exactly. The notebooks preserve `version11` analysis values. Checkpoint provenance was checked against the Osnabrück training cluster; numerical analyses were checked against `version11`.

## Paper result map

| Paper result | Released entry point and data | Verification |
| --- | --- | --- |
| Fig. 2; Figs. S2–S3, behavior | `paper_plots.ipynb`; exact behavior arrays in `science-paper-analysis-cache.zip` | Default plots and statistics rerun; reporting exceptions are listed below. |
| Fig. 3; Figs. S7–S10; Tables S2–S7, neural alignment | `paper_plots.ipynb`; published-model and neural RDMs in `science-paper-rdms-*.zip`, comparison RDMs in the base archive | All 50 detailed table entries match printed precision; 92 VVC/parietal score vectors match `version11`. |
| Fig. 4B; Figs. S6 and S11, backbone sweeps | `paper_plots.ipynb`; same RDMs and saved layer-selection data | All 92 score vectors and selected layer/glimpse pairs reproduce; caption/statistic exceptions are listed below. |
| Fig. S4 and individual-model probes | `gpn_analysis.ipynb`; checkpoint, test embeddings, RDMs, and optional saved activation ZIP for the selected model | Default GPN-R-SimCLR outputs match its saved preview; the position-decoding and embedding-control qualifications are listed below. |
| Fig. S5, searchlight | External Doerig et al. analysis code | Not independently rerun in this release. |

## What reproduces

- All 46 published checkpoints have the same SHA-256 hashes on the training cluster and in the analysis copy. All 46 load strictly with the release model definitions.
- The published cGSN checkpoint was trained with binary cross-entropy with logits and multihot scene labels. The release's paper-training command selects that objective with `--paper_cgsn_bce`; its loss and gradient match the cluster helper. The notebooks load the published checkpoint directly.
- The 50 detailed entries in Tables S2–S7 match their printed score precision and selected input/layer/glimpse.
- All 92 VVC/parietal model score vectors in the backbone sweeps match the analysis copy exactly. A separate direct run using its own notebook, helper, and data yielded identical analysis logs and all 92 vectors.
- A fresh checkpoint-to-RDM replay retained all 92 selected layer/glimpse pairs. 36 of 46 model RDM collections were byte-identical to the saved RDMs; the largest RDM element difference in the others was `1.27e-6`, and the largest subject alignment-score difference was `9.5e-9`.
- A fresh macOS `gpn-science` environment ran both default notebooks to completion using the released local data layout. The summary notebook produced 15 figures and its model-alignment CSV and statistics JSON match the verified analysis outputs byte for byte. The GPN-R-SimCLR notebook produced 32 figures; with its supplied activation cache, every printed result matches the saved preview. GPN-B-ILSVRC, cGSN-R-SimCLR, and GPN-R-DINOv2-B were also run end to end.

## Differences and qualifications

| Result | Published wording/value | Released analysis / `version11` | Effect on interpretation |
| --- | --- | --- | --- |
| GPN-R versus sGSN, VVC | Gap `0.075` | `0.0703712537`, with superiority `p=0.0078125` | The direction/significance holds; the printed gap does not reproduce. |
| Fig. S6, ViT-B/32 recurrence | R–B difference `>0.01`, `W=0`, FDR `p=0.0086` in the blanket caption | Difference `0.00402824`, `W=3`, FDR `p=0.0390625` | Positive and still below 0.05; the blanket magnitude/statistic claim needs a ViT exception. |
| Fig. S4A position decoding | Trained RS first-layer scores near ceiling | Fresh five-fold replay differs from its cache by at most `0.00155663` R²; other models differ by at most `6.56e-7` | Near-ceiling/path-integration pattern holds; exact cache history is unavailable. |
| Fig. S4B embedding control | Dimensions independently shuffled per glimpse | Both analysis branches resample scalar values, with replacement, from pooled embedding elements | Binding is supported for the control actually run. It does not separately establish the result for a within-glimpse permutation. The notebook can rerun the control, but its random draws need not match the saved result. |
| Fig. S4B sample count | 1,000 fixation sequences | 250 base traces × four return-query simulations | The author's 1,000 trace-query count is valid; these are not 1,000 distinct base traces. |
| Nonrecurrent glimpse test | Friedman `df=24` | Six glimpse levels across 25 blocks give `df=5`; the printed p-values match `df=5` | The strong significance holds; the degrees-of-freedom label differs. |
| Fig. 2 behavior contrasts | Interaction `W=152`, Holm `p=.79`; recurrence `W=97`, Holm `p=.16` | `W=149`, Holm `p=.7310`; `W=107`, Holm `p=.2818` | Both remain nonsignificant. |
| Identity-corrected spatial contrast | Mean `+4.6e-4` | Mean `-0.0004597294`, corrected `p=.9789` | The null conclusion holds; the printed sign differs. |
| Fig. 3 superiority tests | General description says two-sided | The relevant analysis calls one-sided `greater` tests. A two-sided calculation for the strongest contrasts yields FDR `p=.03125`. | Superiority remains below 0.05, but test-tail wording should be read with this qualification. |
| Three-backbone saccade main effect | Holm-adjusted `p=.0078` | Raw `p=.0078125`; Holm over the three collapsed contrasts `p=.0234375` | Still significant; the correction label/value does not reproduce. |
| GSN unique variance | FDR-adjusted `p=.33` | cGSN `p=.5`, sGSN `p=.125` | Both remain nonsignificant. |
| Fig. S6 DVD-B S/RS | Equal maximum VVC alignment | Both select the same nonlearned cumulative-glimpse baseline at glimpse 5 (`R²=.1216742385`). Their best learned `lstm_out` scores are `0.0732514355` (S) and `0.1050566572` (RS). | Equal plotted maxima do not imply equal learned representations. |
| Approximate summaries | Several values printed at limited precision | Most differ only slightly. A strict `p < .015` is not satisfied by the reproduced `p=.015625`. | Approximate decimal reporting does not alter main patterns; the strict inequality is an exact wording exception. |
| Fresh inference versus supplied caches | Saved model-derived results | Largest prediction-element drift is `3.15e-5`; fresh RDM and score drift are as above. | No published layer/glimpse winner changes. The released caches retain the analysis values. |

## Scope limits

The notebooks start from supplied neural RDMs, test-515 embeddings, and published checkpoints. This release does not reconstruct neural RDMs from raw fMRI, retrain all 46 networks from initialization, regenerate every alternative-encoder RDM from raw images, or rerun the external Fig. S5 searchlight. Every checkpoint was loaded and exercised, but the single-model notebook was run end to end for four representative models rather than all 46. Some qualitative panels were manually composed from model outputs.

The available cluster cGSN helper implements BCE with logits and matches the release training loss and gradient. Its exact training-day source revision was not archived, so the checkpoint hashes establish which weights were analyzed, but cannot by themselves certify every training step.

Additional RN50-init/RN152 controls outside Tables S2–S7 have 28 cached layer selections that cite layers absent from four supplied RDM files; the published 50 table rows are unaffected. Pixel-level identity of fresh PNG renders is not claimed; the numerical CSV and statistics JSON match the verified analysis outputs byte for byte.

The paper's main directional conclusions remain supported by the released analyses. The table above separates those conclusions from exact numerical and method-description statements that differ. The notebooks use the `version11` analysis values; they do not substitute printed paper numbers.

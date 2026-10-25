# Reproducibility audit for the Science Advances paper

This file records differences between the published text and the surviving code/data. It does **not** change published values or fit code to them. For training provenance we used the Osnabrück cluster source and checkpoint hashes; for analyses and numerical comparisons we used `version11`; for public use we preserved the GPN repository's model and notebook interfaces.

## What reproduces

- All 46 published checkpoints have the same SHA-256 hashes on the training cluster and in the analysis copy. All 46 load strictly with the release model definitions.
- The 50 detailed entries in Tables S2–S7 match their printed score precision and selected input/layer/glimpse.
- All 92 VVC/parietal model score vectors in the backbone sweeps match the analysis copy exactly. A separate direct run using its own notebook, helper, and data yielded identical analysis logs and all 92 vectors.
- A fresh checkpoint-to-RDM replay retained all 92 selected layer/glimpse pairs. 36 of 46 model RDM collections were byte-identical to the saved RDMs; the largest RDM element difference in the others was `1.27e-6`, and the largest subject alignment-score difference was `9.5e-9`.
- The original released defaults remain byte-identical for fixed-input model outputs, RSA/CI arrays, and a seeded optimizer step across 14 original models in the tested runtime. The cGSN BCE option matches the surviving cluster helper's loss and gradient.
- A fresh macOS `gpn-science` environment ran both notebooks to completion in a Jupyter kernel using the supplied local data. The resulting model-alignment CSV and statistics JSON are byte-identical to the verified analysis outputs.

## Differences and qualifications

| Result | Published wording/value | Surviving analysis | Effect on interpretation |
| --- | --- | --- | --- |
| cGSN training | BCE with multihot scene labels | The cluster helper uses BCE with logits. The earlier public default used cosine contrast for `scene_loss=1`; `--paper_cgsn_bce` now selects the cluster objective explicitly. | Published checkpoint hashes match the cluster. Use the explicit flag for new BCE training. The surviving helper postdates the checkpoint, so its training-day bytes cannot be certified. |
| GPN-R versus sGSN, VVC | Gap `0.075` | `0.0703712537`, with superiority `p=0.0078125` | The direction/significance holds; the printed gap does not reproduce. |
| Fig. S6, ViT-B/32 recurrence | R–B difference `>0.01`, `W=0`, FDR `p=0.0086` in the blanket caption | Difference `0.00402824`, `W=3`, FDR `p=0.0390625` | Positive and still below 0.05; the blanket magnitude/statistic claim needs a ViT exception. |
| Fig. S4A position decoding | Trained RS first-layer scores near ceiling | Fresh five-fold replay differs from its cache by at most `0.00155663` R²; other models differ by at most `6.56e-7` | Near-ceiling/path-integration pattern holds; exact cache history is unavailable. |
| Fig. S4B embedding control | Dimensions independently shuffled per glimpse | Both analysis branches resample scalar values, with replacement, from pooled embedding elements | Binding is supported for the control actually run. It does not separately establish the result for a within-glimpse permutation. No ICL analysis script was found in the inspected cluster training tree. |
| Fig. S4B sample count | 1,000 fixation sequences | 250 base traces × four return-query simulations | The author's 1,000 trace-query count is valid; these are not 1,000 distinct base traces. |
| Nonrecurrent glimpse test | Friedman `df=24` | Six glimpse levels across 25 blocks give `df=5`; the printed p-values match `df=5` | The strong significance holds; the degrees-of-freedom label differs. |
| Fig. 2 behavior contrasts | Interaction `W=152`, Holm `p=.79`; recurrence `W=97`, Holm `p=.16` | `W=149`, Holm `p=.7310`; `W=107`, Holm `p=.2818` | Both remain nonsignificant. |
| Identity-corrected spatial contrast | Mean `+4.6e-4` | Mean `-0.0004597294`, corrected `p=.9789` | The null conclusion holds; the printed sign differs. |
| Fig. 3 superiority tests | General description says two-sided | The relevant analysis calls one-sided `greater` tests. A two-sided calculation for the strongest contrasts yields FDR `p=.03125`. | Superiority remains below 0.05, but test-tail wording should be read with this qualification. |
| Three-backbone saccade main effect | Holm-adjusted `p=.0078` | Raw `p=.0078125`; Holm over the three collapsed contrasts `p=.0234375` | Still significant; the correction label/value does not reproduce. |
| GSN unique variance | FDR-adjusted `p=.33` | cGSN `p=.5`, sGSN `p=.125` | Both remain nonsignificant. |
| Fig. S6 DVD-B S/RS | Equal maximum VVC alignment | Both select the same nonlearned cumulative-glimpse baseline at glimpse 5 (`R²=.1216742385`). Their best learned `lstm_out` scores are `0.0732514355` (S) and `0.1050566572` (RS). | Equal plotted maxima do not imply equal learned representations. |
| Approximate summaries | Several values printed at limited precision | Most differ only slightly. A strict `p < .015` is not satisfied by the reproduced `p=.015625`. | Approximate decimal reporting does not alter main patterns; the strict inequality is an exact wording exception. |
| Historical saved activations/RDMs | Archived model-derived results | Largest prediction-element drift is `3.15e-5`; fresh RDM and score drift are as above. | No layer/glimpse winner changes; current-run patch compatibility is separately byte-identical. |
| Extra RN50-init/RN152 assets | Outside Tables S2–S7 | 28 cached selections cite layers absent from four extra RDM files; 560 other checks match. | The published 50 table rows are unaffected. |

## Scope limits

The audits start from supplied neural RDMs and embedding datasets; raw fMRI preprocessing, full end-to-end retraining, and fresh extraction of all alternative pretrained encoders were not repeated. The author reports that the Fig. S5 searchlight used code from Doerig et al. (*Nature Machine Intelligence*); that external computation was not independently rerun here. Some qualitative panels are manually composed from model outputs.

Fresh PNG renders of Figs. 4B, S6, and S11 differ in canvas dimensions from an earlier verification render. The numerical CSV and statistics JSON match byte for byte, and the new plots were visually checked; pixel-level identity is not claimed.

The paper's main directional conclusions remain supported by the available analyses. The table above distinguishes those conclusions from precise numerical and method-description statements that do not reproduce exactly. The notebooks use the surviving analysis values; they do not silently substitute printed paper numbers.

"""Paper-wide layer and backbone plots using the repository RSA functions."""

import argparse
import csv
import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

from .analysis_helper_funcs import compute_gpn_correlations, mean_and_bootstrap_ci


MODEL_IDS = json.loads(Path(__file__).with_name('science_models.json').read_text())
BACKBONES = ('ILSVRC', 'SimCLR', 'DVD', 'RN101', 'cornet', 'EffNet',
             'ConvNext', 'ViT', 'MaxViT', 'Swin', 'Dinov2_B')
BACKBONE_LABELS = ('RN50-ILSVRC', 'RN50-SimCLR', 'RN50-DVD-B', 'RN101-ILSVRC',
                   'CORnet-S', 'EfficientNet-B1', 'ConvNeXt-Tiny', 'ViT-B/32',
                   'MaxViT-T', 'Swin-T', 'DINOv2-B')
VARIANTS = ('B', 'S', 'R', 'RS')
INTERNAL_LAYERS = ('rn50_glimpse', 'lstm_h_0', 'lstm_h_1', 'lstm_out',
                   'glimpse_output', 'RN50_gl_seqadd')


def read_pickle(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def model_layers(rdms, n_pairs):
    """Exclude target RDMs while retaining the dataset's model-layer order."""
    layers = [key for key in rdms if key not in ('semantic_embed', 'scene_embed')]
    if not layers or any(np.shape(rdms[key]) != (6, n_pairs) for key in layers):
        raise ValueError('Expected each model layer RDM to have shape (6, n_scene_pairs)')
    return layers


def best_alignment(neural_rdms, rdms):
    """Use the same subject-mean layer/glimpse selection as paper_plots.ipynb."""
    layers = model_layers(rdms, np.shape(neural_rdms)[1])
    scores = compute_gpn_correlations(neural_rdms, rdms, layers, n_glimpse=6,
                                      measure='regression')
    glimpse, layer = np.unravel_index(np.nanargmax(scores.mean(axis=0)), scores.shape[1:])
    input_layer = layers.index('rn50_glimpse')
    input_glimpse = scores[:, :, input_layer].mean(axis=0).argmax()
    return {'layer': layers[layer], 'glimpse': int(glimpse),
            'subjects': scores[:, glimpse, layer],
            'input_subjects': scores[:, input_glimpse, input_layer]}


def sweep_statistics(results):
    """The published 2 x 2 x 11 interaction and four families of paired contrasts."""
    interactions = [((results[f'GPN-RS-{b}']['subjects'] - results[f'GPN-R-{b}']['subjects'])
                     - (results[f'GPN-S-{b}']['subjects'] - results[f'GPN-B-{b}']['subjects'])) / 2
                    for b in BACKBONES]
    statistic, p_value = friedmanchisquare(*interactions)
    summary = {'interaction': {'chi2': float(statistic), 'df': len(BACKBONES) - 1,
                               'p': float(p_value)}, 'contrasts': {}}
    for first, second in [('R', 'B'), ('RS', 'S'), ('S', 'B'), ('RS', 'R')]:
        rows = []
        for backbone in BACKBONES:
            diff = (results[f'GPN-{first}-{backbone}']['subjects']
                    - results[f'GPN-{second}-{backbone}']['subjects'])
            # scipy's signed-rank test is undefined for an entirely zero contrast.
            w, p = (0.0, 1.0) if np.all(diff == 0) else wilcoxon(diff)
            rows.append({'backbone': backbone, 'difference': float(diff.mean()),
                         'W': float(w), 'p': float(p)})
        adjusted = multipletests([row['p'] for row in rows], method='fdr_bh')[1]
        for row, p in zip(rows, adjusted):
            row['p_fdr'] = float(p)
        summary['contrasts'][f'{first}-{second}'] = rows
    return summary


def plot_sweep(results, roi, output, alternative=None):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    colors = ('#80cdc1', '#018571', '#dfc27d', '#a6611a')
    ticks, labels = [], []
    for group, backbone in enumerate(BACKBONES):
        for offset, variant in enumerate(VARIANTS):
            x = group * 5 + offset
            scores = results[f'GPN-{variant}-{backbone}']['subjects'][:, None]
            mean, _, _, low, high = mean_and_bootstrap_ci(scores)
            ax.errorbar(x, mean[0], yerr=[[low[0]], [high[0]]], fmt='o',
                        color=colors[offset], markersize=3, linewidth=0.8)
            ticks.append(x)
            labels.append(variant)
        # Plot the GPN-B input-embedding baseline for each backbone.
        baseline = results[f'GPN-B-{backbone}']['input_subjects'].mean()
        ax.plot([group * 5 - 0.3, group * 5 + 3.3], [baseline, baseline], 'k:', linewidth=1)
        ax.text(group * 5 + 1.5, -0.19, BACKBONE_LABELS[group], ha='right', va='top',
                rotation=25, transform=ax.get_xaxis_transform(), fontsize=7)
        if group:
            ax.axvline(group * 5 - 1, color='0.85', linewidth=0.5)
    if alternative is not None:
        ax.axhline(alternative, color='salmon', linestyle='-.', linewidth=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r'Max. alignment ($R^2$)')
    ax.set_title('Ventral visual cortex (VVC)' if roi == 'ventral' else 'Parietal cortex')
    ax.set_ylim(bottom=0)
    ax.margins(x=0.02)
    fig.subplots_adjust(bottom=0.34)
    for suffix in ('pdf', 'png'):
        fig.savefig(Path(output).with_suffix('.' + suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_fig4(neural_rdms, rdms, output):
    scores = compute_gpn_correlations(neural_rdms, rdms, INTERNAL_LAYERS,
                                      n_glimpse=6, measure='regression')
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    labels = ('Glimpse embed', 'LSTM h0', 'LSTM h1', 'LSTM out', 'Prediction', 'Glimpse seqadd')
    for index, label in enumerate(labels):
        mean, _, _, low, high = mean_and_bootstrap_ci(scores[:, :, index])
        ax.errorbar(np.arange(6), mean, yerr=np.stack([low, high]),
                    fmt='o--' if index == 5 else 'o-', markersize=3,
                    color='black' if index == 3 else None, label=label)
    ax.set(xlabel='Glimpse number', ylabel=r'VVC alignment ($R^2$)', ylim=(0, 0.27))
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
    for suffix in ('pdf', 'png'):
        fig.savefig(Path(output).with_suffix('.' + suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)
    return {'first_glimpse_gain': float(np.mean(scores[:, 0, 3] - scores[:, 0, 0])),
            'lstm_alignment_glimpse_spearman': float(np.mean([spearmanr(np.arange(6), row).statistic
                                                   for row in scores[:, :, 3]])),
            'max_lstm_minus_max_seqadd': float(np.mean(scores[:, :, 3].max(axis=1)
                                                       - scores[:, :, 5].max(axis=1))),
            'last_lstm_minus_prediction': float(np.mean(scores[:, 5, 3] - scores[:, 5, 4]))}


def run(data_root, output_dir, figures=('4', 'S6', 'S11')):
    root, output = Path(data_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested = set(figures)
    rois = [roi for roi, figure in [('ventral', 'S6'), ('parietal', 'S11')] if figure in requested]
    neural = read_pickle(root / 'rdms/streams_all_neural_rdms_correlation_special515.pkl')
    neural_rdms = {roi: np.stack([subject[roi] for subject in neural.values()])
                   for roi in set(rois) | ({'ventral'} if '4' in requested else set())}
    results = {roi: {} for roi in rois}
    summary = {}
    rows = []
    needed = MODEL_IDS if rois else {'GPN-R-SimCLR': MODEL_IDS['GPN-R-SimCLR']}
    for label, identifier in needed.items():
        rdms = read_pickle(root / 'rdms' / f'gpn_rdms_{identifier}.pkl')
        for roi in rois:
            best = best_alignment(neural_rdms[roi], rdms)
            results[roi][label] = best
            rows.append([roi, label, best['layer'], best['glimpse'],
                         float(best['subjects'].mean()), *best['subjects'].tolist()])
        if label == 'GPN-R-SimCLR' and '4' in requested:
            summary['figure4'] = plot_fig4(neural_rdms['ventral'], rdms, output / 'Figure_4B')
        print(f'Analysed {label}', flush=True)
    if rows:
        with (output / 'model_alignment.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['roi', 'model', 'layer', 'glimpse', 'mean_r2']
                            + [f'subject_{i+1}' for i in range(len(neural))])
            writer.writerows(rows)
    # Load saved alternative-model layer selections when available.
    metadata_path = root / 'analysis_helpers/best_layers_streams_allrois.pkl'
    metadata = read_pickle(metadata_path) if metadata_path.exists() else None
    for roi in rois:
        alternative = None
        if metadata is not None:
            alternative = float(np.asarray(metadata['net_corr_best_layers_allrois'][roi]).mean(axis=0).max())
        figure = 'S6' if roi == 'ventral' else 'S11'
        plot_sweep(results[roi], roi, output / f'Figure_{figure}', alternative)
        summary[figure] = sweep_statistics(results[roi])
    (output / 'statistics.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('figures/science_paper'))
    parser.add_argument('--figures', nargs='+', choices=('4', 'S6', 'S11'), default=['4', 'S6', 'S11'])
    args = parser.parse_args()
    run(args.data_root, args.output_dir, args.figures)


if __name__ == '__main__':
    main()

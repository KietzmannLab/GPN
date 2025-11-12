import os
import glob
import pickle
import numpy as np
from scipy.stats import zscore, spearmanr, pearsonr
from sklearn.linear_model import LinearRegression

def load_pickle(filepath):
    """Load a pickled file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def compute_network_correlations_all(neural_rdms, net_names, metric='regression', standardize=True, positive=True, verbose=False):

    net_count = 0
    for rdm_file in net_names:
        with open(rdm_file, 'rb') as f:
            rdms = pickle.load(f)
        net_count += len(rdms)

    n_subjects = len(neural_rdms)
    corr_array = np.zeros((n_subjects, net_count))
    network_names = []
    network_layer_names = []
    
    net_count = 0
    for rdm_file in net_names:
        with open(rdm_file, 'rb') as f:
            rdms = pickle.load(f)
        network_name = rdm_file.split('/')[-1].split('_test')[0]
        for name in rdms:
            if rdms[name] is None:
                if verbose:
                    print(f"Warning: RDM for {network_name} - {name} is None. Skipping.")
                continue
            if rdms[name].std() < 1e-8:
                if verbose:
                    print(f"Warning: RDM for {network_name} - {name} has zero variance. Skipping.")
                continue
            if standardize:
                net_rdm_h = zscore(rdms[name])
            else:
                net_rdm_h = rdms[name]
            for subj_idx in range(n_subjects):
                if rdms[name].shape != neural_rdms[subj_idx].shape:
                    if verbose:
                        print(f"Warning: Shape mismatch for {network_name} - {name}: network RDM shape {rdms[name].shape}, neural RDM shape {neural_rdms[subj_idx].shape}")
                    continue
                if standardize:
                    neural_rdm_h = zscore(neural_rdms[subj_idx])
                else:
                    neural_rdm_h = neural_rdms[subj_idx]
                if metric != 'regression':
                    corr_array[subj_idx, net_count] = spearmanr(net_rdm_h, neural_rdm_h).statistic if metric == 'spearman' else pearsonr(net_rdm_h, neural_rdm_h).statistic
                else:
                    corr_array[subj_idx, net_count] = LinearRegression(positive=positive).fit(net_rdm_h.reshape(-1, 1), neural_rdm_h.reshape(-1, 1)).score(net_rdm_h.reshape(-1, 1), neural_rdm_h.reshape(-1, 1))
            network_names.append(network_name)
            network_layer_names.append(name)
            net_count += 1
    return corr_array, network_names, network_layer_names

network_rdm_files = sorted(glob.glob('../rdms/*test_515_RDMs.pkl'))

data_NSD = load_pickle('../rdms/streams_all_neural_rdms_correlation_special515.pkl')
streams_rois = list(data_NSD['subj01'].keys())
print(f'Found {len(streams_rois)} ROIs: {streams_rois}')

best_layer_names_allrois = {}
net_corr_best_layers_allrois = {}
net_names_all_allrois = {}
net_layers_compared_all_allrois = {}

for n, roi in enumerate(streams_rois):

    print(f'Processing correlations for ROI: {roi}, {n+1} out of {len(streams_rois)}')

    roi_RDMs = np.vstack([data_NSD[subj][roi] for subj in data_NSD])

    net_corr, net_names, net_layer_names = compute_network_correlations_all(roi_RDMs, network_rdm_files, verbose=False)
    
    select_layers = ~np.isnan(net_corr.mean(axis=0))
    net_names = [x for x, m in zip(net_names, select_layers) if m]
    net_layer_names = [x for x, m in zip(net_layer_names, select_layers) if m]
    net_corr = net_corr[:,select_layers]

    best_layer_names = []
    net_corr_best_layers = []
    net_names_all = []
    net_layers_compared_all = [] 
    for net_name_h in set(net_names):
        net_name_ids = [i for i in range(len(net_names)) if net_names[i] == net_name_h]
        net_layer_best_id = np.argmax(net_corr[:,net_name_ids].mean(axis=0))
        net_names_all.append(net_name_h)
        best_layer_names.append(net_layer_names[net_name_ids[net_layer_best_id]])
        net_corr_best_layers.append(net_corr[:,net_name_ids[net_layer_best_id]])
        net_layers_compared_all.append(len(net_name_ids))

    best_layer_names_allrois[roi] = best_layer_names
    net_corr_best_layers_allrois[roi] = np.array(net_corr_best_layers).T
    net_names_all_allrois[roi] = net_names_all
    net_layers_compared_all_allrois[roi] = net_layers_compared_all

# Save the results
save_data = {
    'best_layer_names_allrois': best_layer_names_allrois,
    'net_corr_best_layers_allrois': net_corr_best_layers_allrois,
    'net_names_all_allrois': net_names_all_allrois,
    'net_layers_compared_all_allrois': net_layers_compared_all_allrois,
    'streams_rois': streams_rois,
}
with open(f'best_layers_streams_allrois.pkl', 'wb') as f:
    pickle.dump(save_data, f) 

print(f'Saved best layers for all streams ROIs to best_layers_streams_allrois.pkl')
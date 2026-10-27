"""Offline unit tests plus opt-in checks against published data/checkpoints.

python -m unittest discover -s tests -v
GPN_PAPER_DATA_ROOT enables all-checkpoint/data checks.
GPN_SCIENCE_BASELINE enables bitwise comparisons to the original release.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from analysis_helpers.analysis_helper_funcs import (CocoGaze_t515, compute_gpn_correlations,
                                                   get_hyp, get_net_params)
from analysis_helpers.science_paper import (BACKBONES, MODEL_IDS, best_alignment,
                                            model_layers, plot_fig4, sweep_statistics)
from train.models.backbones import feature_dim, forward_features, load_extended_backbone
from train.models.helper_funcs import get_network_model

torch.set_num_threads(1)


def assert_bitwise_equal(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_array_equal(actual, expected)
    if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
        raise AssertionError('Array dtype, shape or bytes differ from the original')


def hypothesis(label, root='datasets/'):
    rec, loc, bbv, n, layers, di, dr, gaze, dva = get_net_params(label.replace('Dinov2_B', 'Dinov2B'))
    return get_hyp(str(root), gaze, dva, n, layers, dr, di, 1, -2, rec, loc, bbv, label)


class BackboneTests(unittest.TestCase):
    def test_feature_dimensions_and_legacy_defaults(self):
        expected = {0: 2048, 1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 768, 6: 2048,
                    7: 1280, 8: 768, 9: 2048, 10: 768, 11: 768, 12: 512, 13: 512}
        self.assertEqual({key: feature_dim(key) for key in expected}, expected)

    def test_dinov2_image_embeddings_use_cpu_when_mps_is_available(self):
        from analysis_helpers.analysis_helper_funcs import get_RN50_embeddings

        class DummyDino(torch.nn.Module):
            chosen_device = None

            def to(self, device):
                self.chosen_device = str(device)
                return self

            def forward(self, images):
                self.saw_cpu_input = images.device.type == 'cpu'
                return torch.ones(images.shape[0], 768)

        model = DummyDino()
        images = np.zeros((2, 91, 91, 3), dtype=np.uint8)
        with patch('torch.hub.load', return_value=model), \
             patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            features = get_RN50_embeddings(images, bbv=5)
        self.assertEqual(model.chosen_device, 'cpu')
        self.assertTrue(model.saw_cpu_input)
        self.assertEqual(features.shape, (2, 768))

    def test_all_paper_hyperparameters_match_checkpoint_identifiers(self):
        for label, identifier in MODEL_IDS.items():
            with self.subTest(model=label), contextlib.redirect_stdout(io.StringIO()):
                hyp = hypothesis(label)
                # Only the constructor size is reduced for this offline shape test.
                hyp['network']['n_rnn'] = 12
                model, name = get_network_model(hyp)
                self.assertEqual(name.replace('_n_12_', '_n_1024_'), identifier)
                dim = feature_dim(hyp['dataset']['bbv'])
                self.assertEqual(model.actv_proj.in_features, dim)
                model.eval()
                with torch.no_grad():
                    _, output = model(torch.zeros(2, 2, dim), torch.zeros(2, 2, 2))
                slot = 1 if label.startswith('sGSN') else 2 if label.startswith('cGSN') else 0
                self.assertEqual(output[slot].shape, (2, 2, 768 if slot == 1 else 91 if slot == 2 else dim))

    def test_extended_encoder_readouts_without_weight_downloads(self):
        for bbv in range(7, 14):
            with self.subTest(backbone=bbv):
                try:
                    model, preprocess, layer, cls_token = load_extended_backbone(bbv, pretrained=False)
                except ImportError:
                    if bbv == 13:
                        continue  # CORnet is optional for users of other backbones.
                    raise
                model.eval()
                images = preprocess(torch.zeros(1, 3, 91, 91, dtype=torch.uint8))
                with torch.no_grad():
                    features = forward_features(model, layer, images, cls_token)
                self.assertEqual(features.shape, (1, feature_dim(bbv)))
                self.assertFalse(layer._forward_hooks)
                del model, features

    def test_readout_hook_removed_if_forward_fails(self):
        class Failing(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.readout = torch.nn.Identity()

            def forward(self, x):
                self.readout(x)
                raise RuntimeError('test failure')

        model = Failing()
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            forward_features(model, model.readout, torch.zeros(1, 3))
        self.assertFalse(model.readout._forward_hooks)


class RDMTests(unittest.TestCase):
    def test_figure4_correlation_uses_lstm_alignment_not_input_subtracted_gain(self):
        scores = np.zeros((8, 6, 6))
        scores[:, :, 3] = np.arange(6) * 0.01
        scores[:, :, 0] = [0, 0.05, 0, 0.1, 0, 0.15]
        with tempfile.TemporaryDirectory() as directory:
            with patch('analysis_helpers.science_paper.compute_gpn_correlations', return_value=scores):
                result = plot_fig4(None, None, Path(directory) / 'test')
        self.assertEqual(result['lstm_alignment_glimpse_spearman'], 1.0)

    def test_zero_paired_contrast_keeps_fdr_results_finite(self):
        rng = np.random.default_rng(1955)
        results = {f'GPN-{variant}-{backbone}': {'subjects': rng.random(8)}
                   for backbone in BACKBONES for variant in ['B', 'S', 'R', 'RS']}
        results['GPN-RS-DVD']['subjects'] = results['GPN-S-DVD']['subjects'].copy()
        statistics = sweep_statistics(results)
        rows = statistics['contrasts']['RS-S']
        dvd = next(row for row in rows if row['backbone'] == 'DVD')
        self.assertEqual((dvd['W'], dvd['p']), (0.0, 1.0))
        self.assertTrue(all(np.isfinite(row['p_fdr']) for row in rows))

    def test_metadata_cannot_be_selected_as_a_model_layer(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal((3, 15))
        rdms = {'rn50_glimpse': rng.standard_normal((6, 15)),
                'lstm_out': np.tile(y[0], (6, 1)),
                'semantic_embed': y[0], 'scene_embed': y[0]}
        self.assertEqual(model_layers(rdms, 15), ['rn50_glimpse', 'lstm_out'])
        best = best_alignment(y, rdms)
        expected = compute_gpn_correlations(y, rdms, ['rn50_glimpse', 'lstm_out'])
        glimpse, layer = np.unravel_index(expected.mean(axis=0).argmax(), (6, 2))
        np.testing.assert_array_equal(best['subjects'], expected[:, glimpse, layer])

    def test_malformed_rdm_is_rejected(self):
        with self.assertRaises(ValueError):
            model_layers({'rn50_glimpse': np.zeros((5, 15))}, 15)


class DatasetContractTests(unittest.TestCase):
    def test_new_backbone_keeps_existing_eight_item_batch_and_scene_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = np.arange(3 * 2 * 7 * 4 * 8).reshape(3, 2, 7, 4, 8)
            with h5py.File(root / 'coco_NSD_dg3fix91_r50v7ap_7fix_test_515.h5', 'w') as f:
                f['dg3_fix_actvs'] = values
                f['next_fix_coords'] = np.ones((3, 2, 6, 4, 2))
                f['next_fix_rel_coords'] = np.ones((3, 2, 6, 4, 2)) * 2
                f['mpnet_embeddings'] = np.ones((3, 768))
                f['full_image_actvs'] = np.ones((3, 8))
            with h5py.File(root / 'coco-test-515.h5', 'w') as f:
                f['multihot_labels'] = np.ones((3, 91))
            for in_memory in [0, 1]:
                with self.subTest(in_memory=in_memory), contextlib.redirect_stdout(io.StringIO()):
                    dataset = CocoGaze_t515(str(root) + '/', 'dg3', 6, in_memory, bbv=7)
                    sample = dataset[3]  # scene 1, trace 1
                    self.assertEqual(len(dataset), 6)
                    self.assertEqual(len(sample), 8)
                    np.testing.assert_array_equal(sample[0].numpy(), values[1, 1, :, 0])
                    np.testing.assert_array_equal(sample[2][0].numpy(), [0, 0])
                    self.assertEqual((sample[6].item(), sample[7].item()), (1, 1))


class PublishedCgsnLossTests(unittest.TestCase):
    def test_bce_opt_in_matches_cluster_objective_and_gradient(self):
        import ast
        from train.models.helper_funcs import compute_losses
        from train.helpers.helper_funcs import create_cpc_matrix

        logits = torch.linspace(-1, 1, 4 * 3 * 91, dtype=torch.float32).reshape(4, 3, 91).requires_grad_()
        labels = torch.zeros(4, 91)
        labels[torch.arange(4), torch.arange(4)] = 1
        outputs = [None, None, logits, None]
        mask = create_cpc_matrix(12, 3)
        hyp = {'optimizer': {'losses': {'glimpse_loss': 0, 'semantic_loss': 0,
                                        'scene_loss': 1, 'gazeloc_loss': 0}}}
        actual, floor = compute_losses(outputs, None, None, None, labels, mask, hyp, True,
                                       paper_cgsn_bce=True)
        targets = labels.unsqueeze(1).repeat(1, 3, 1).reshape(-1, 91)
        expected = torch.nn.BCEWithLogitsLoss()(logits.reshape(-1, 91), targets.float())
        assert_bitwise_equal(actual.detach().numpy(), expected.detach().numpy())
        assert_bitwise_equal(torch.autograd.grad(actual, logits, retain_graph=True)[0].numpy(),
                             torch.autograd.grad(expected, logits, retain_graph=True)[0].numpy())
        self.assertEqual(float(floor.detach()), 0.0)
        legacy, _ = compute_losses(outputs, None, None, None, labels, mask, hyp, True)
        self.assertNotEqual(float(actual.detach()), float(legacy.detach()))

        source_root = os.environ.get('GPN_CLUSTER_SOURCE')
        if source_root:
            source = Path(source_root) / 'train/models/helper_funcs.py'
            tree = ast.parse(source.read_text())
            function = next(node for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == 'compute_losses')
            namespace = {'torch': torch}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
            cluster, cluster_floor = namespace['compute_losses'](
                outputs, None, None, None, labels, mask, hyp, True)
            assert_bitwise_equal(actual.detach().numpy(), cluster.detach().numpy())
            assert_bitwise_equal(floor.detach().numpy(), cluster_floor.detach().numpy())
            assert_bitwise_equal(torch.autograd.grad(actual, logits, retain_graph=True)[0].numpy(),
                                 torch.autograd.grad(cluster, logits)[0].numpy())

    def test_training_cli_rejects_bce_without_scene_prediction(self):
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[1]
        check = subprocess.run([sys.executable, str(root / 'train/train_net.py'),
                                '--paper_cgsn_bce', '--scene_loss', '0'],
                               capture_output=True, text=True)
        self.assertNotEqual(check.returncode, 0)
        self.assertIn('--paper_cgsn_bce requires --scene_loss 1', check.stderr)


@unittest.skipUnless(os.environ.get('GPN_SCIENCE_BASELINE'), 'set GPN_SCIENCE_BASELINE for original-release regression')
class OriginalReleaseTests(unittest.TestCase):
    def test_original_rsa_and_confidence_intervals_are_bitwise_identical(self):
        import pickle
        from analysis_helpers.analysis_helper_funcs import mean_and_bootstrap_ci
        root = Path(os.environ['GPN_SCIENCE_BASELINE'])
        records = json.loads((root / 'legacy_baseline.json').read_text())
        layers = json.loads((root / 'rsa_baseline_layers.json').read_text())
        data_root = Path(next(iter(records.values()))['hyp']['dataset']['dataset_path']).parent
        with (data_root / 'rdms/streams_all_neural_rdms_correlation_special515.pkl').open('rb') as stream:
            neural = pickle.load(stream)
        with np.load(root / 'rsa_baseline.npz') as expected:
            for label, record in records.items():
                with (data_root / 'rdms' / ('gpn_rdms_' + record['identifier'] + '.pkl')).open('rb') as stream:
                    rdms = pickle.load(stream)
                for roi in ['ventral', 'parietal']:
                    with self.subTest(model=label, roi=roi):
                        y = np.stack([subject[roi] for subject in neural.values()])
                        scores = compute_gpn_correlations(y, rdms, layers[label])
                        assert_bitwise_equal(scores, expected[label + '::' + roi])
                        best = best_alignment(y, rdms)
                        i, j = np.unravel_index(scores.mean(axis=0).argmax(), scores.shape[1:])
                        self.assertEqual((best['glimpse'], best['layer']), (i, layers[label][j]))
                        assert_bitwise_equal(best['subjects'], scores[:, i, j])
                        if label == 'GPN-R-SimCLR' and roi == 'ventral':
                            selected = ['rn50_glimpse', 'lstm_h_0', 'lstm_h_1', 'lstm_out', 'glimpse_output', 'RN50_gl_seqadd']
                            for index, layer in enumerate(selected):
                                for n, value in enumerate(mean_and_bootstrap_ci(scores[:, :, layers[label].index(layer)])):
                                    assert_bitwise_equal(value, expected[f'figure4_ci::{index}::{n}'])

    def test_all_original_checkpoint_outputs_are_bitwise_identical(self):
        root = Path(os.environ['GPN_SCIENCE_BASELINE'])
        records = json.loads((root / 'legacy_baseline.json').read_text())
        checked = 0
        with np.load(root / 'legacy_baseline.npz') as expected:
            for label, record in records.items():
                with self.subTest(model=label), contextlib.redirect_stdout(io.StringIO()):
                    hyp = hypothesis(label, record['hyp']['dataset']['dataset_path'])
                    self.assertEqual(hyp, record['hyp'])
                    model, identifier = get_network_model(hyp)
                    self.assertEqual(identifier, record['identifier'])
                    model.load_state_dict(torch.load(record['checkpoint'], map_location='cpu', weights_only=False), strict=True)
                    model.eval()
                    x = torch.from_numpy(expected[label + '::input'])
                    c = torch.from_numpy(expected[label + '::coords'])
                    with torch.no_grad():
                        acts, predictions = model(x, c)
                    for key, value in acts.items():
                        assert_bitwise_equal(value.numpy(), expected[label + '::act::' + key])
                        checked += 1
                    for slot, value in enumerate(predictions):
                        if isinstance(value, torch.Tensor):
                            assert_bitwise_equal(value.numpy(), expected[label + '::pred::' + str(slot)])
                            checked += 1
        self.assertEqual(checked, 140)  # 168 saved arrays, including 28 unchanged inputs.


@unittest.skipUnless(os.environ.get('GPN_PAPER_DATA_ROOT'), 'set GPN_PAPER_DATA_ROOT for published-artifact checks')
class PublishedArtifactTests(unittest.TestCase):
    def test_added_encoder_paths_match_version11_bit_for_bit(self):
        import ast
        source_root = Path(os.environ.get('GPN_VERSION11_ROOT', os.environ['GPN_PAPER_DATA_ROOT']))
        source = source_root / 'analysis_helpers/analysis_helper_funcs.py'
        if not source.is_file():
            self.skipTest('set GPN_VERSION11_ROOT to compare encoder paths with the version11 source')
        try:
            import cornet
        except ImportError:
            cornet = None  # optional dependency for backbone 13
        from torchvision import models, transforms
        from analysis_helpers.analysis_helper_funcs import get_RN50_embeddings
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_RN50_embeddings')
        inputs = np.random.default_rng(1955).integers(0, 256, (2, 91, 91, 3), dtype=np.uint8)
        constructors = ['efficientnet_b1', 'vit_b_32', 'resnet101', 'convnext_tiny', 'swin_t', 'maxvit_t']
        originals = {name: getattr(models, name) for name in constructors}
        original_cornet = cornet.cornet_s if cornet is not None else None
        with contextlib.ExitStack() as stack:
            for name, constructor in originals.items():
                stack.enter_context(patch.object(models, name, side_effect=lambda *a, _ctor=constructor, **kw: _ctor(weights=None)))
            if cornet is not None:
                stack.enter_context(patch.object(cornet, 'cornet_s', side_effect=lambda **kw: original_cornet(pretrained=False, map_location='cpu')))
            stack.enter_context(patch('torch.cuda.is_available', return_value=False))
            stack.enter_context(patch('torch.backends.mps.is_available', return_value=False))
            namespace = dict(vars(models), np=np, torch=torch, transforms=transforms, cornet=cornet)
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
            for bbv in range(7, 14 if cornet is not None else 13):
                with self.subTest(backbone=bbv):
                    torch.manual_seed(1955)
                    expected = namespace['get_RN50_embeddings'](inputs, bbv=bbv)
                    torch.manual_seed(1955)
                    actual = get_RN50_embeddings(inputs, bbv=bbv)
                    assert_bitwise_equal(actual, expected)

    def test_all_46_published_checkpoints_load_strictly_and_run(self):
        root = Path(os.environ['GPN_PAPER_DATA_ROOT'])
        for label, expected_name in MODEL_IDS.items():
            with self.subTest(model=label), contextlib.redirect_stdout(io.StringIO()):
                hyp = hypothesis(label, str(root / 'datasets') + '/')
                model, identifier = get_network_model(hyp)
                self.assertEqual(identifier, expected_name)
                path = root / 'train/logs/net_params' / identifier / (identifier + '.pth')
                model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False), strict=True)
                model.eval()
                with torch.no_grad():
                    _, outputs = model(torch.zeros(1, 2, feature_dim(hyp['dataset']['bbv'])), torch.zeros(1, 2, 2))
                for value in outputs:
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.isfinite(value).all())


if __name__ == '__main__':
    unittest.main()

"""Regression tests for comparing EEGNet with and without temporal graph filtering."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from .cli import build_parser, run
from .data import TrialDataset, save_bundle
from .eegnet import EEGNet
from .torch_benchmark import evaluate_agfl
from .waveforms import prepare_waveform_split


def waveform_trials():
    subjects = np.repeat(["p0", "p1", "p2"], 4)
    labels = np.tile([0, 1], 6)
    table = pd.DataFrame({
        "Subject_ID": subjects, "Trial_ID": np.tile(["0", "1", "2", "3"], 3),
        "Label": labels,
    })
    rng = np.random.default_rng(28)
    waveforms = rng.normal(0, 2e-6, (len(table), 2, 96))
    waveforms[:, 0, 36:40] += labels[:, None] * 3e-6
    return TrialDataset(table, waveforms.astype(np.float32), sfreq=128., tmin=-.2,
                        channel_names=("Fz", "Cz"), provenance={"status": "synthetic"})


class EEGNetArchitectureTests(unittest.TestCase):
    def test_waveform_scaling_is_channelwise_and_uses_only_training_rows(self):
        epochs = np.array([
            [[1., 3., 2., 5.], [20., 24., 21., 23.]],
            [[111., 121., 115., 118.], [320., 330., 315., 335.]],
            [[2., 6., 3., 4.], [27., 29., 26., 28.]],
        ], dtype=np.float32) * 1e-6
        train = np.array([2, 0])
        held_out = np.array([1])
        split = prepare_waveform_split(epochs, train, held_out, batch_size=1)
        reference = epochs[train].astype(np.float64)
        mean = reference.mean(axis=(0, 2), keepdims=True)
        std = reference.std(axis=(0, 2), keepdims=True)
        for rows, dataset in ((train, split.train), (held_out, split.held_out)):
            expected = (epochs[rows] - mean) / std
            actual = dataset[np.arange(len(rows))]
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_paired_weights_and_temporal_graph(self):
        waveforms = torch.arange(2 * 3 * 436, dtype=torch.float32).reshape(2, 3, 436) / 1000
        options = dict(n_channels=3, n_samples=436, sfreq=256., dropout=0.)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.random.default_generator.manual_seed(12)
            baseline = EEGNet(**options, config="none").eval()
            self.assertIsNone(baseline.graph)
            logits = baseline(waveforms)
            self.assertEqual(tuple(logits.shape), (2, 2))
            self.assertTrue(torch.isfinite(logits).all().item())
            tokens = baseline.encode_tokens(waveforms)
            self.assertEqual(tuple(tokens.shape), (2, 13, 16))
            for config, hops in (("k0", 0), ("inter", 2)):
                with self.subTest(config=config):
                    torch.random.default_generator.manual_seed(12)
                    arm = EEGNet(**options, config=config, hops=2).eval()
                    for name, weights in baseline.state_dict().items():
                        torch.testing.assert_close(arm.state_dict()[name], weights, rtol=0, atol=0)
                    self.assertEqual(arm.graph.hops, hops)
                    torch.testing.assert_close(arm.encode_tokens(waveforms), tokens)
                    graph_inputs = []
                    hook = arm.graph.register_forward_pre_hook(
                        lambda module, args: graph_inputs.append(args[0].detach().clone())
                    )
                    try:
                        graph_logits = arm(waveforms)
                    finally:
                        hook.remove()
                    self.assertEqual(tuple(graph_logits.shape), (2, 2))
                    self.assertTrue(torch.isfinite(graph_logits).all().item())
                    self.assertEqual(len(graph_inputs), 1)
                    torch.testing.assert_close(graph_inputs[0], tokens)

    def test_requires_multiple_temporal_nodes(self):
        with self.assertRaises(ValueError):
            EEGNet(3, 32, 256., config="inter")


class EEGNetEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = waveform_trials()
        cls.options = dict(backbone="eegnet", input_mode="eeg", grouping="loso",
                           configs=("none", "k0", "inter"), seeds=(9,),
                           validation_epochs=(1, 2), inner_splits=2, batch_size=3, n_threads=1)
        cls.result = evaluate_agfl(cls.data, **cls.options)

    def test_cli_routes_waveforms_and_reports_all_paired_comparisons(self):
        with TemporaryDirectory() as temp:
            bundle, output = Path(temp) / "bundle", Path(temp) / "results"
            save_bundle(self.data, bundle)
            args = build_parser().parse_args([
                "eegnet", "--input", str(bundle), "--output", str(output), "--allow-synthetic",
                "--seeds", "9", "--validation-epochs", "1", "2", "--inner-splits", "2", "--batch-size", "3",
            ])
            with patch(f"{evaluate_agfl.__module__}.evaluate_agfl", return_value=self.result) as evaluate:
                run(args)
            settings = evaluate.call_args.kwargs
            self.assertEqual(settings["backbone"], "eegnet")
            self.assertEqual(settings["batch_size"], 3)
            self.assertEqual(settings["configs"], ("none", "inter", "k0"))
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual({(p["baseline"], p["config"]) for p in metrics["paired_comparisons"]},
                             {("none", "inter"), ("none", "k0"), ("k0", "inter")})
            self.assertEqual(json.loads((output / "eegnet_config.json").read_text())["backbone"], "eegnet")

    def test_nested_folds_and_paired_predictions_on_waveforms(self):
        keys = ["Subject_ID", "Trial_ID"]
        truth = self.data.table.set_index(keys).Label.sort_index()
        predictions = self.result["predictions"]
        self.assertEqual(set(predictions.Config), {"none", "k0", "inter"})
        self.assertEqual(set(predictions.Feature_Set), {"eeg_waveform"})
        self.assertEqual(set(predictions.Model), {
            "eegnet_none::seed9", "eegnet_k0::seed9", "eegnet_inter::seed9",
        })
        self.assertTrue(predictions.Probability.between(0, 1).all())
        folds = self.result["folds"].set_index("Fold")
        for _, arm in predictions.groupby("Config"):
            pd.testing.assert_series_equal(arm.set_index(keys).Label.sort_index(), truth)
            self.assertTrue(arm.Subject_ID.eq(arm.Fold.map(folds.Test_Subject)).all())
        validation_log = self.result["validation"]
        self.assertEqual(len(validation_log), len(folds) * 2 * 3 * 2)
        for row in validation_log.itertuples():
            train, validation = set(row.Train_Subjects), set(row.Validation_Subjects)
            self.assertTrue(train.isdisjoint(validation))
            self.assertEqual(train | validation, set(self.data.table.Subject_ID) - {row.Test_Subject})
        selections = self.result["selection"]
        self.assertEqual(len(selections), len(folds) * 3)
        self.assertEqual(set(selections.Status), {"ok"})
        self.assertTrue(selections.Selected_Epochs.isin([1, 2]).all())

    def test_held_out_waveforms_and_labels_cannot_affect_other_test_trials(self):
        # p0 stays entirely out of fitting. Keep its first trial identical while
        # shifting other recordings by hundreds of their training standard deviations.
        table = self.data.table.copy()
        held_out = table.Subject_ID.eq("p0")
        table.loc[held_out, "Label"] = 1 - table.loc[held_out, "Label"]
        epochs = self.data.epochs.copy()
        epochs[held_out & table.Trial_ID.ne("0")] += 1e-3
        changed = evaluate_agfl(replace(self.data, table=table, epochs=epochs), **self.options)
        for name in ("validation", "selection"):
            records = [result[name].loc[result[name].Test_Subject.eq("p0")].reset_index(drop=True)
                       for result in (self.result, changed)]
            pd.testing.assert_frame_equal(*records)
        predictions = []
        for result in (self.result, changed):
            frame = result["predictions"]
            fixed_trial = frame[frame.Subject_ID.eq("p0") & frame.Trial_ID.eq("0")]
            predictions.append(fixed_trial.set_index("Config").Probability.sort_index())
        self.assertEqual(len(predictions[0]), 3)
        pd.testing.assert_series_equal(*predictions, check_exact=True)

    def test_rejects_feature_tables_and_non_eeg_requests(self):
        with self.assertRaises(ValueError):
            evaluate_agfl(replace(self.data, epochs=None), backbone="eegnet", epochs=1)
        mixed = self.data.table.assign(EMG_Test_Mean=np.arange(len(self.data.table), dtype=float))
        with self.assertRaises(ValueError):
            evaluate_agfl(replace(self.data, table=mixed), backbone="eegnet", input_mode="eeg_emg", epochs=1)


if __name__ == "__main__":
    unittest.main()

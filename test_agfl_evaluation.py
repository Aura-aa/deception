"""Regression checks for the participant-held-out graph comparison."""

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from .data import TrialDataset
from .cli import build_parser
from .torch_benchmark import evaluate_agfl


def eeg_trials():
    rng = np.random.default_rng(648)
    subjects, trials = 4, 6
    labels = np.tile([0, 1], subjects * trials // 2)
    table = pd.DataFrame({
        "Subject_ID": np.repeat([f"p{i}" for i in range(subjects)], trials),
        "Trial_ID": np.tile([str(i) for i in range(trials)], subjects),
        "Label": labels,
    })
    for window in range(1, 5):
        for band in ("theta", "alpha", "beta", "gamma"):
            for channel in ("Fz", "Cz"):
                table[f"EEG_{channel}_SW{window}_{band}_LogPower"] = (
                    rng.normal(size=len(table)) + 0.3 * labels
                )
    # A missing signal value exercises estimation of the training-fold median.
    table.loc[1, "EEG_Fz_SW1_theta_LogPower"] = np.nan
    return TrialDataset(table, provenance={"status": "synthetic"})


class AGFLEvaluationTests(unittest.TestCase):
    def test_cli_defaults_and_legacy_feature_guard(self):
        args = build_parser().parse_args(["agfl", "--input", "bundle", "--output", "out"])
        self.assertEqual(args.input_mode, "eeg")
        self.assertEqual(args.grouping, "loso")
        self.assertEqual(args.configs, ["none", "inter"])
        self.assertIsNone(args.epochs)
        old = replace(self.data, provenance={"feature_extraction": {"version": 2}})
        with self.assertRaisesRegex(ValueError, "features command"):
            evaluate_agfl(old)

    @classmethod
    def setUpClass(cls):
        cls.data = eeg_trials()
        cls.options = dict(
            input_mode="eeg", grouping="loso", seeds=(7, 13),
            configs=("none", "inter"), validation_epochs=(1, 2),
            inner_splits=2, d=3, a=3, n_threads=1,
        )
        cls.result = evaluate_agfl(cls.data, **cls.options)

    def test_baseline_and_graph_predict_the_same_held_out_trials(self):
        predictions = self.result["predictions"]
        models = {"none::seed7", "inter::seed7", "none::seed13", "inter::seed13"}
        self.assertEqual(set(predictions.Model), models)
        self.assertEqual(set(predictions.Feature_Set), {"eeg"})
        self.assertTrue(predictions.Probability.between(0, 1).all())
        np.testing.assert_array_equal(predictions.Prediction, predictions.Probability >= 0.5)
        keys = ["Subject_ID", "Trial_ID"]
        truth = self.data.table.set_index(keys).Label.sort_index()
        folds = self.result["folds"].set_index("Fold")
        for _, part in predictions.groupby("Model"):
            pd.testing.assert_series_equal(part.set_index(keys).Label.sort_index(), truth)
            self.assertTrue(part.Subject_ID.eq(part.Fold.map(folds.Test_Subject)).all())
        self.assertEqual(set(folds.Test_Subject), set(self.data.table.Subject_ID))
        self.assertTrue(folds.Test_Subject.is_unique)

    def test_nested_validation_partitions_are_disjoint_and_paired(self):
        validation = self.result["validation"]
        selection = self.result["selection"]
        models = set(self.result["predictions"].Model)
        for outer in self.result["folds"].itertuples():
            train = set(outer.Train_Subjects)
            self.assertEqual(train, set(self.data.table.Subject_ID) - {outer.Test_Subject})
            records = validation.loc[validation.Fold.eq(outer.Fold)]
            self.assertEqual(set(records.Test_Subject), {outer.Test_Subject})
            inner_test_subjects = []
            for _, inner in records.groupby("Inner_Fold"):
                expected_validation = set(inner.iloc[0].Validation_Subjects)
                inner_test_subjects.extend(expected_validation)
                self.assertEqual(set(inner.Model), models)
                self.assertEqual(set(inner.Epochs), {1, 2})
                self.assertFalse(inner.duplicated(["Model", "Epochs"]).any())
                self.assertEqual(len(inner), len(models) * 2)
                for row in inner.itertuples():
                    self.assertEqual(set(row.Validation_Subjects), expected_validation)
                    self.assertEqual(set(row.Train_Subjects), train - expected_validation)
            self.assertCountEqual(inner_test_subjects, train)
            chosen = selection.loc[selection.Fold.eq(outer.Fold)]
            self.assertEqual(set(chosen.Model), models)
            self.assertEqual(set(chosen.Status), {"ok"})
            for row in chosen.itertuples():
                arm = records[records.Model.eq(row.Model)]
                counts = arm.groupby("Epochs", sort=True)[
                    ["Class0_Correct", "Class0_N", "Class1_Correct", "Class1_N"]
                ].sum()
                pooled = (counts.Class0_Correct / counts.Class0_N +
                          counts.Class1_Correct / counts.Class1_N) / 2
                # Sorted epoch counts make idxmax choose the shorter tied run.
                self.assertEqual(row.Selected_Epochs, pooled.idxmax())
                self.assertAlmostEqual(row.Inner_Balanced_Accuracy, pooled.max())

    def test_other_test_trials_cannot_change_selection_or_a_prediction(self):
        # Keep p0's first trial fixed, change features of its other trials and
        # reverse all its labels. None of that may influence its fitted model.
        altered = self.data.table.copy()
        held_out = altered.Subject_ID.eq("p0")
        changed_rows = held_out & altered.Trial_ID.ne("0")
        eeg = altered.columns[altered.columns.str.startswith("EEG_")]
        altered.loc[changed_rows, eeg] = 5000 + altered.loc[changed_rows, eeg]
        altered.loc[held_out, "Label"] = 1 - altered.loc[held_out, "Label"]
        # These distractors simulate extra modalities in an otherwise EEG-only bundle.
        altered["EMG_Test_Mean"] = altered.Label * 1e6
        altered["Beh_RT_Test"] = altered.Label
        options = dict(self.options, seeds=(7,))
        result = evaluate_agfl(replace(self.data, table=altered), **options)
        for key in ("validation", "selection"):
            before = self.result[key]
            before = before[before.Test_Subject.eq("p0") & before.Model.str.endswith("seed7")]
            after = result[key]
            after = after[after.Test_Subject.eq("p0")]
            pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
        probabilities = []
        for run in (self.result, result):
            frame = run["predictions"]
            keep = frame.Subject_ID.eq("p0") & frame.Trial_ID.eq("0") & frame.Model.str.endswith("seed7")
            probabilities.append(frame[keep].set_index("Model").Probability.sort_index())
        self.assertEqual(len(probabilities[0]), 2)
        pd.testing.assert_series_equal(*probabilities, check_exact=True)

    def test_future_emg_support_handles_a_single_physiology_node(self):
        table = self.data.table.copy()
        table["EMG_Test_Mean"] = np.linspace(-1, 1, len(table))
        result = evaluate_agfl(
            replace(self.data, table=table), input_mode="eeg_emg",
            configs=("intra", "combined"), epochs=1, d=3, a=3, n_threads=1,
        )
        predictions = result["predictions"]
        self.assertEqual(set(predictions.Feature_Set), {"eeg_emg"})
        self.assertEqual(len(predictions), len(table) * 2)
        self.assertTrue(predictions.Probability.between(0, 1).all())

    def test_missing_required_observations_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_agfl(self.data, input_mode="eeg_emg", epochs=1)
        incomplete = self.data.table.drop(columns=[
            "EEG_Fz_SW4_gamma_LogPower", "EEG_Cz_SW4_gamma_LogPower",
        ])
        with self.assertRaises(ValueError):
            evaluate_agfl(replace(self.data, table=incomplete), input_mode="eeg", epochs=1)
        two_people = self.data.table[self.data.table.Subject_ID.isin(["p0", "p1"])].copy()
        with self.assertRaises(ValueError):
            evaluate_agfl(replace(self.data, table=two_people), validation_epochs=(1, 2))


if __name__ == "__main__":
    unittest.main()

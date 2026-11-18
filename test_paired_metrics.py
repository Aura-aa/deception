"""Regression cases for the participant-level AGFL comparison."""

import unittest

import pandas as pd

from .metrics import paired_agfl_comparisons


class PairedAGFLMetricsTests(unittest.TestCase):
    def test_pair_by_trial_and_weight_participants_equally(self):
        rows = []
        # The improvement is 1 for A and 0 for B. B contributes more trials.
        for subject, labels in (("A", [0, 1]), ("B", [0, 0, 0, 1]), ("C", [1, 1])):
            for seed in (1, 2):
                for trial, label in enumerate(labels):
                    for config in ("none", "inter"):
                        prediction = 1 - label if subject == "A" and config == "none" else label
                        rows.append({"Seed": seed, "Fold": subject, "Subject_ID": subject,
                                     "Trial_ID": str(trial), "Label": label, "Prediction": prediction,
                                     "Config": config, "Feature_Set": "eeg"})
        data = pd.DataFrame(rows)
        details, summary = paired_agfl_comparisons(data.iloc[::-1])
        self.assertEqual(len(details), 6)
        self.assertEqual(summary[0]["participants_total"], 3)
        self.assertEqual(summary[0]["participants_scored"], 2)
        self.assertEqual(summary[0]["mean_paired_balanced_accuracy_difference"], 0.5)
        self.assertTrue(details.loc[details.Subject_ID == "C", "Delta_Balanced_Accuracy"].isna().all())

        data.loc[(data.Subject_ID == "A") & (data.Seed == 2) & (data.Config == "inter"), "Prediction"] = None
        _, partial = paired_agfl_comparisons(data)
        self.assertEqual(partial[0]["participants_scored"], 1)
        self.assertEqual(partial[0]["mean_paired_balanced_accuracy_difference"], 0.0)
        with self.assertRaisesRegex(ValueError, "identical trials"):
            paired_agfl_comparisons(data.iloc[1:])

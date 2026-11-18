"""Regression checks for raw EEG preparation and short-window spectral features."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from .data import TrialDataset, load_bundle
from .features import EEG_BANDS, EEG_SUBWINDOWS, eeg_features, extract_features
from .preparation import prepare_dataset, validate_alignment, write_alignment_template


class PublicEEGPreparationTest(unittest.TestCase):
    def test_raw_eeg_only_and_optional_emg(self):
        import mne

        with TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / "sub-S01" / "ses-01"
            eeg_dir, beh_dir = session / "eeg", session / "beh"
            eeg_dir.mkdir(parents=True)
            beh_dir.mkdir()
            stem = "sub-S01_ses-01_task-GG_run-01"
            pd.DataFrame({
                "Trial_Order": [10, 2], "Trial_Condition": ["Probable_Lie", "Probable_Truth"],
            }).to_csv(beh_dir / f"{stem}_beh.tsv", sep="\t", index=False)
            names = ["Cz", "124", "Sync"]
            samples = np.arange(30 * 256) / 256
            info = mne.create_info(names, 256., ["eeg"] * 3)
            raw = mne.io.RawArray(np.tile(1e-5 * np.sin(2 * np.pi * 10 * samples), (3, 1)), info)
            raw.save(eeg_dir / f"{stem}_eeg.fif", verbose="ERROR")
            pd.DataFrame({"name": names, "type": ["EEG"] * 3}).to_csv(
                eeg_dir / f"{stem}_channels.tsv", sep="\t", index=False)
            (eeg_dir / f"{stem}_eeg.json").write_text('{"SamplingFrequency": 256}')
            pd.DataFrame({"onset": [8., 17.]}).to_csv(eeg_dir / f"{stem}_events.tsv", sep="\t", index=False)

            path = root / "alignment.csv"
            alignment = write_alignment_template(root, path)
            self.assertEqual(alignment["Trial_ID"].tolist(), ["2", "10"])
            self.assertNotIn("Physio_Onset", alignment)
            alignment["EEG_Onset"] = [8.25, 17.25]
            alignment["Event_Phase"] = "correctness_prompt"
            alignment.to_csv(path, index=False)
            prepare_dataset(root, root / "eeg-only", path)
            data = load_bundle(root / "eeg-only")
            self.assertEqual(data.table["Label"].tolist(), [0, 1])
            self.assertEqual(data.channel_names, ("Cz",))
            self.assertEqual(data.table["QC_Source_Row"].tolist(), [1, 0])
            np.testing.assert_allclose(data.table["EEG_Onset"], [8.25, 17.25])
            np.testing.assert_allclose(data.table["QC_EEG_NearestEventDistance_s"], .25)
            self.assertNotIn("Physio_Onset", data.table)
            self.assertEqual(data.provenance["feature_extraction"]["version"], 3)

            with self.assertRaisesRegex(ValueError, "Physio_Onset"):
                validate_alignment(alignment, data.table, include_physiology=True)
            template = write_alignment_template(root, root / "multi-template.csv", include_physiology=True)
            self.assertIn("Physio_Onset", template)
            alignment["Physio_Onset"] = [10.25, 19.25]
            alignment.to_csv(path, index=False)
            (beh_dir / f"{stem}_physio.json").write_text(json.dumps({
                "Columns": ["EMG"], "StartTime": 2., "SamplingFrequency": 100.,
            }))
            pd.Series(np.ones(3000)).to_csv(beh_dir / f"{stem}_physio.tsv", header=False, index=False)
            multi = prepare_dataset(root, root / "with-emg", path, include_physiology=True)
            np.testing.assert_allclose(multi.table["EMG_EMG_Mean"], 1.)
            np.testing.assert_allclose(multi.epochs, data.epochs)

    def test_short_window_band_powers_scale_with_signal_energy(self):
        sfreq = 256.
        tmin = -51 / sfreq
        signal = np.random.default_rng(21).normal(size=436)
        epochs = np.stack([signal, 2 * signal])[:, None, :]
        features = eeg_features(epochs, sfreq, ["Cz"], tmin=tmin, subwindows=EEG_SUBWINDOWS)
        for band in EEG_BANDS:
            for window in EEG_SUBWINDOWS:
                power = features[f"EEG_Cz_SW{window}_{band}_LogPower"]
                self.assertAlmostEqual(power.iloc[1] - power.iloc[0], np.log10(4.))

    def test_table_only_reextraction_preserves_legacy_spectral_version(self):
        table = pd.DataFrame({"Subject_ID": ["A"], "Trial_ID": ["1"], "Label": [1],
                              "EEG_Cz_SW1_theta_LogPower": [-300.]})
        dataset = TrialDataset(table=table, provenance={"feature_extraction": {"version": 2}})
        result = extract_features(dataset)
        self.assertEqual(result.provenance["feature_extraction"]["version"], 2)
        pd.testing.assert_frame_equal(result.table, table)

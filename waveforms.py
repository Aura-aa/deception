"""Training-fold channel normalization for on-demand EEG waveform batches."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch


def _trial_rows(rows, n_trials, name):
    rows = np.asarray(rows)
    if rows.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array of trial indices.")
    if rows.dtype == np.bool_:
        if rows.size != n_trials:
            raise ValueError(f"{name} mask length must equal the number of trials.")
        rows = np.flatnonzero(rows)
    if not rows.size or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError(f"{name} must contain at least one integer trial index.")
    if np.any(rows < 0) or np.any(rows >= n_trials) or np.unique(rows).size != rows.size:
        raise ValueError(f"{name} must contain distinct in-range trial indices.")
    return rows.astype(np.intp, copy=True)


def _read_batch(epochs, rows):
    # Fancy indexing a memory map reads just the requested trials. The copy
    # allows centering in place regardless of the source dtype and writability.
    batch = np.array(epochs[rows], dtype=np.float64, copy=True)
    if not np.isfinite(batch).all():
        raise ValueError("EEG waveform batches must contain finite samples.")
    return batch


@dataclass(frozen=True)
class WaveformDataset:
    epochs: np.ndarray
    rows: np.ndarray
    origin: np.ndarray
    centered_mean: np.ndarray
    scale: np.ndarray
    varying: np.ndarray

    def __len__(self):
        return self.rows.size

    def __getitem__(self, index):
        rows = np.atleast_1d(self.rows[index])
        if rows.ndim != 1:
            raise IndexError("Waveform batch indices must select a one-dimensional list of trials.")
        batch = _read_batch(self.epochs, rows)
        batch -= self.origin[None, :, None]
        batch -= self.centered_mean[None, :, None]
        batch[:, ~self.varying, :] = 0.
        batch /= self.scale[None, :, None]
        result = batch.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Normalized EEG waveforms exceed the finite float32 range.")
        return torch.from_numpy(result)


@dataclass(frozen=True)
class WaveformSplit:
    train: WaveformDataset
    held_out: WaveformDataset
    pca_components: None = None


def prepare_waveform_split(epochs, train_rows, held_out_rows, *, batch_size):
    """Fit one mean and population standard deviation per channel on training samples.

    Variances are merged using within-batch and between-batch sums of squares.
    Subtracting a fixed training sample before accumulating means also preserves
    truly constant training channels and improves precision for large offsets.
    """
    if getattr(epochs, "ndim", None) != 3 or min(epochs.shape) < 1:
        raise ValueError("Expected nonempty EEG epochs shaped (trials, channels, samples).")
    if not (np.issubdtype(epochs.dtype, np.floating) or np.issubdtype(epochs.dtype, np.integer)):
        raise ValueError("EEG waveforms must contain real numeric samples.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("Waveform batch_size must be a positive integer.")
    train_rows = _trial_rows(train_rows, len(epochs), "train_rows")
    held_out_rows = _trial_rows(held_out_rows, len(epochs), "held_out_rows")
    if np.intersect1d(train_rows, held_out_rows, assume_unique=True).size:
        raise ValueError("Waveform training and held-out trial indices must be disjoint.")

    origin = np.array(epochs[train_rows[0], :, 0], dtype=np.float64, copy=True)
    mean = np.zeros_like(origin)
    sum_squares = np.zeros_like(origin)
    count = 0
    for start in range(0, train_rows.size, batch_size):
        batch = _read_batch(epochs, train_rows[start:start + batch_size])
        batch -= origin[None, :, None]
        batch_mean = batch.mean(axis=(0, 2))
        batch -= batch_mean[None, :, None]
        np.square(batch, out=batch)
        batch_count = batch.shape[0] * batch.shape[2]
        total = count + batch_count
        delta = batch_mean - mean
        sum_squares += batch.sum(axis=(0, 2)) + delta**2 * (count * batch_count / total)
        mean += delta * (batch_count / total)
        count = total
    scale = np.sqrt(sum_squares / count)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Training EEG channel statistics exceed the finite float64 range.")
    varying = scale > 0
    if not varying.any():
        raise ValueError("All EEG channels are constant in this training fold.")
    scale[~varying] = 1.
    return WaveformSplit(
        WaveformDataset(epochs, train_rows, origin, mean, scale, varying),
        WaveformDataset(epochs, held_out_rows, origin, mean, scale, varying),
    )

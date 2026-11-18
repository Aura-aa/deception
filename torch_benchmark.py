"""Participant-held-out evaluation of the AGFL graph and its matched controls."""
from __future__ import annotations

from itertools import product
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from .agfl import AGFLModel
from .data import PHYSIOLOGY_PREFIXES
from .eegnet import EEGNet
from .evaluation import (_feature_blocks, _group_scaler, _inner_cv, _outer_splits,
                         _validate_training_observations, _vector_groups)
from .features import EEG_BANDS, EEG_SUBWINDOWS
from .waveforms import prepare_waveform_split


CONFIGS = ("none", "k0", "intra", "inter", "combined")


def _input_groups(dataset, input_mode):
    if input_mode not in ("eeg", "eeg_emg", "eeg_physiology"):
        raise ValueError("AGFL input_mode must be 'eeg', 'eeg_emg', or 'eeg_physiology'.")
    version = dataset.provenance.get("feature_extraction", {}).get("version")
    if version is not None and int(version) < 3:
        raise ValueError("These EEG features predate the corrected sub-window band powers. "
                         "Run the features command on the aligned epochs and use the new bundle.")
    table = dataset.table
    expected_names = [f"eeg_sw{window}_{band}" for window in EEG_SUBWINDOWS for band in EEG_BANDS]
    suffixes = tuple(f"_SW{window}_{band}_LogPower" for window in EEG_SUBWINDOWS for band in EEG_BANDS)
    columns = sorted(name for name in table if name.startswith("EEG_") and name.endswith(suffixes))
    if input_mode != "eeg":
        prefixes = ("EMG_",) if input_mode == "eeg_emg" else PHYSIOLOGY_PREFIXES
        physiology = sorted(name for name in table if name.startswith(prefixes))
        if not physiology:
            raise ValueError(f"{input_mode} requires recorded features with a prefix from {prefixes}.")
        columns.extend(physiology)
    groups = _vector_groups(columns)
    names = [name for name, _ in groups]
    missing = [name for name in expected_names if name not in names]
    if missing:
        raise ValueError(f"AGFL requires all 16 EEG window/band groups. Missing: {', '.join(missing)}.")
    channels = [{columns[i].removeprefix("EEG_").removesuffix(suffix) for i in indices}
                for (_, indices), suffix in zip(groups, suffixes)]
    if any(group != channels[0] for group in channels):
        raise ValueError("All 16 EEG groups must have the same named channels.")
    sizes = [sum(name.startswith("eeg_") for name in names)]
    if input_mode != "eeg":
        sizes.append(sum(not name.startswith("eeg_") for name in names))
    return columns, groups, sizes


class EncodedSplit(NamedTuple):
    train: torch.Tensor
    held_out: torch.Tensor
    mean: torch.Tensor
    components: torch.Tensor

    @property
    def pca_components(self):
        return len(self.components)


def _encode(train, held_out, groups, feature_blocks, width):
    """Fit both encoding stages exclusively on these training observations."""
    _validate_training_observations(train, feature_blocks)
    # A constant/unobserved training coordinate must not activate at test time.
    supported = np.array([np.unique(col[np.isfinite(col)]).size > 1 for col in train.T])
    train = np.where(supported, train, 0.)
    held_out = np.where(supported, held_out, 0.)
    encoder = _group_scaler(groups, width, equal_width=True)
    train_flat = encoder.fit_transform(train)
    pca = PCA(n_components=.95, svd_solver="full").fit(train_flat)
    shape = (-1, len(groups), width)
    return EncodedSplit(
        torch.tensor(train_flat.reshape(shape), dtype=torch.float32),
        torch.tensor(encoder.transform(held_out).reshape(shape), dtype=torch.float32),
        torch.tensor(pca.mean_, dtype=torch.float32),
        torch.tensor(pca.components_, dtype=torch.float32),
    )


def _train(split, labels, config, seed, checkpoints, make_model, options):
    """Train one fresh model. Only inner validation requests multiple checkpoints."""
    counts = np.bincount(labels, minlength=2)
    if not counts.all():
        raise ValueError("An AGFL training fold has only one label class.")
    weights = torch.tensor(len(labels) / (2. * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
    target = torch.tensor(labels, dtype=torch.long)
    torch.random.default_generator.manual_seed(seed)
    device = options["device"]
    model = make_model(split, config).to(device)
    waveform_model = isinstance(model, EEGNet)
    if waveform_model:
        options["architectures"][config] = model.architecture
    # Graph initialization consumes extra random numbers. Reset the training
    # stream and isolate the minibatch generator so paired arms see the same
    # input order and encoder dropout masks.
    torch.random.default_generator.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    criterion = criterion.to(device)
    batch_size = options["batch_size"]
    scores = {}
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        if batch_size is None:
            batches = (slice(None),)
        else:
            order = torch.randperm(len(target), generator=generator).numpy()
            batches = (order[start:start + batch_size] for start in range(0, len(target), batch_size))
        for indices in batches:
            optimizer.zero_grad()
            logits = model(split.train[indices].to(device))
            batch_target = target[indices].to(device)
            # The training-fold class weights have a mean of one. Averaging
            # by batch size preserves that objective even in one-class batches.
            loss = criterion(logits, batch_target) / len(batch_target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
            if waveform_model:
                model.constrain_weights()
        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                n = len(split.held_out)
                step = batch_size or n
                probabilities = []
                for start in range(0, n, step):
                    batch = split.held_out[start:start + step].to(device)
                    probabilities.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
                probability = np.concatenate(probabilities)
            if not np.isfinite(probability).all():
                raise FloatingPointError("Non-finite held-out probabilities.")
            scores[epoch] = probability
    return scores


def _model_name(backbone, config, seed):
    prefix = "eegnet_" if backbone == "eegnet" else ""
    return f"{prefix}{config}::seed{seed}"


def _tune(outer_rows, y, subjects, targets, arms, prepare_split, make_model, options, fold_info):
    candidates = options["validation_epochs"]
    # The grouped splitter uses only outer-training participants. Within-person
    # label permutations preserve both the groups and their class counts.
    splits = _inner_cv(y, subjects, options["inner_splits"], grouped=True)
    tallies = {(arm, epoch): np.zeros((2, 2), dtype=int) for arm in arms for epoch in candidates}
    rows = []
    for fold, (train, valid) in enumerate(splits):
        split = prepare_split(outer_rows[train], outer_rows[valid])
        info = {**fold_info, "Inner_Fold": fold, "Train_Subjects": tuple(sorted(set(subjects[train]))),
                "Validation_Subjects": tuple(sorted(set(subjects[valid]))),
                "Train_Rows": len(train), "Validation_Rows": len(valid),
                "PCA_Components": split.pca_components}
        for config, seed in arms:
            target = targets[seed]
            scores = _train(split, target[train], config, seed, candidates, make_model, options)
            labels = target[valid]
            counts = np.bincount(labels, minlength=2)
            for epoch, probability in scores.items():
                correct = np.bincount(labels[(probability >= .5) == labels], minlength=2)
                tallies[((config, seed), epoch)] += np.stack((correct, counts))
                rows.append({**info, "Model": _model_name(options["backbone"], config, seed),
                             "Config": config, "Seed": seed,
                             "Epochs": epoch, "Class0_N": int(counts[0]), "Class1_N": int(counts[1]),
                             "Class0_Correct": int(correct[0]), "Class1_Correct": int(correct[1]),
                             "Balanced_Accuracy": float((correct / counts).mean()) if counts.all() else np.nan})
    winners = {}
    for arm in arms:
        candidate_scores = {}
        for epoch in candidates:
            correct, counts = tallies[arm, epoch]
            if not counts.all():
                raise ValueError("Grouped epoch selection requires both classes across inner validation trials.")
            candidate_scores[epoch] = float((correct / counts).mean())
        best = max(candidates, key=lambda epoch: (candidate_scores[epoch], -epoch))
        winners[arm] = (best, candidate_scores[best])
    return winners, rows


def _training_targets(labels, subjects, seeds, fold, shuffle):
    targets = {}
    for seed in seeds:
        target = labels.copy()
        if shuffle:
            rng = np.random.default_rng(np.random.SeedSequence([seed, fold]))
            for subject in np.unique(subjects):
                indices = np.flatnonzero(subjects == subject)
                target[indices] = rng.permutation(target[indices])
        targets[seed] = target
    return targets


def evaluate_agfl(dataset, *, backbone="bandpower", input_mode="eeg", grouping="loso", configs=None, seeds=(2026,),
                  d=16, a=16, hops=2, kappa=5, epochs=None, validation_epochs=None, inner_splits=3,
                  lr=1e-3, weight_decay=None, shuffle_labels=False, fold_seed=2026, n_threads=4,
                  batch_size=32, device="cpu"):
    """Compare matched AGFL arms. LOSO tunes epochs on training participants by default.

    Explicit epochs select a fixed stopping point. Within-participant evaluation
    supports fixed epochs (60 by default). Shuffled controls permute labels within
    each training participant and keep the original labels for outer testing.
    """
    dataset.validate()
    table = dataset.table
    if device not in ("cpu", "cuda"):
        raise ValueError("Select cpu or cuda as the training device.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if backbone == "eegnet":
        if input_mode != "eeg" or dataset.epochs is None:
            raise ValueError("EEGNet requires EEG-only input with aligned waveform epochs in the bundle.")
        if d != 16:
            raise ValueError("EEGNet-8,2 produces 16 learned features at each time position.")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("EEGNet batch_size must be a positive integer.")
        allowed_configs = ("none", "inter", "k0")
        default_configs = allowed_configs
        feature_set = "eeg_waveform"
        n_channels, n_samples = dataset.epochs.shape[1:]
        representation = {
            "channel_names": list(dataset.channel_names), "sfreq": float(dataset.sfreq),
            "n_samples": n_samples, "tmin": float(dataset.tmin),
            "tmax": float(dataset.tmin + (n_samples - 1) / dataset.sfreq),
            "event_phase": dataset.provenance.get("event_phase"),
            "encoder": "EEGNet-8,2 on aligned EEG waveforms; per-channel training-fold standardization",
            "nodes": "pooled time positions in order", "architectures": {},
        }

        def prepare_split(train_rows, held_out_rows):
            return prepare_waveform_split(dataset.epochs, train_rows, held_out_rows, batch_size=batch_size)

        def make_model(split, config):
            return EEGNet(n_channels, n_samples, dataset.sfreq, config=config, a=a, hops=hops, kappa=kappa)

    elif backbone == "bandpower":
        columns, groups, sizes = _input_groups(dataset, input_mode)
        values = table[columns].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("AGFL feature values contain infinity.")
        blocks = _feature_blocks(columns)
        allowed_configs = CONFIGS
        default_configs = ("none", "inter")
        feature_set = input_mode
        representation = {
            "group_names": [name for name, _ in groups], "modality_sizes": sizes,
            "feature_columns": columns, "encoder": "training-fold median imputation, scaling and equal-width vectors",
            "pca": "95% of training variance on encoded features before graph processing, shared across arms",
        }

        def prepare_split(train_rows, held_out_rows):
            return _encode(values[train_rows], values[held_out_rows], groups, blocks, d)

        def make_model(split, config):
            return AGFLModel(config, sizes, d, a, hops, kappa, split.mean, split.components)

    else:
        raise ValueError("Choose the bandpower or eegnet backbone.")
    if weight_decay is None:
        weight_decay = 0. if backbone == "eegnet" else 1e-2
    y = table.Label.to_numpy(dtype=int)
    subjects = table.Subject_ID.astype(str).to_numpy()
    configs = default_configs if configs is None else tuple(configs)
    seeds = tuple(seeds)
    if grouping not in ("loso", "within"):
        raise ValueError("AGFL grouping must be 'loso' or 'within'.")
    if not configs or len(set(configs)) != len(configs) or any(c not in allowed_configs for c in configs):
        raise ValueError(f"Provide distinct configurations from {allowed_configs} for {backbone}.")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("AGFL seeds must be distinct nonnegative integers.")
    if any(type(v) is not int or v < 1 for v in (d, a, kappa, n_threads)) or type(hops) is not int or hops < 0:
        raise ValueError("AGFL widths, kappa and thread count must be positive; hops must be nonnegative.")
    if type(fold_seed) is not int or fold_seed < 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("AGFL requires a nonnegative fold seed and positive finite learning rate.")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("AGFL weight decay must be finite and nonnegative.")
    if epochs is not None and validation_epochs is not None:
        raise ValueError("Select either fixed epochs or validation_epochs.")
    if epochs is None and validation_epochs is None:
        if grouping == "loso":
            validation_epochs = (20, 40, 60)
        else:
            epochs = 60
    if validation_epochs is not None:
        if grouping != "loso":
            raise ValueError("Grouped epoch selection requires LOSO; use fixed epochs for within-participant evaluation.")
        candidates = tuple(validation_epochs)
        if not candidates or any(type(e) is not int or e < 1 for e in candidates):
            raise ValueError("validation_epochs must contain positive integers.")
        candidates = sorted(set(candidates))
        if type(inner_splits) is not int or inner_splits < 2 or np.unique(subjects).size < 3:
            raise ValueError("Nested LOSO requires at least three subjects and at least two inner splits.")
    else:
        candidates = []
        if type(epochs) is not int or epochs < 1:
            raise ValueError("Fixed epochs must be a positive integer.")
    options = {**representation, "backbone": backbone, "device": device,
               "batch_size": batch_size if backbone == "eegnet" else None,
               "input_mode": input_mode, "grouping": grouping, "configs": list(configs),
               "seeds": list(seeds), "fold_seed": fold_seed, "epochs": epochs,
               "validation_epochs": candidates, "inner_splits": inner_splits if candidates else None,
               "selection_metric": "pooled balanced accuracy on inner out-of-fold predictions" if candidates else None,
               "tie_rule": "fewest epochs" if candidates else None, "threshold": .5, "shuffle_labels": bool(shuffle_labels),
               "shuffle_unit": "within each outer-training participant",
               "d": d, "a": a, "hops": hops, "kappa": kappa, "lr": lr,
               "weight_decay": weight_decay, "n_threads": n_threads, "deterministic": True,
               "optimizer": "Adam", "loss": "training-class-weighted cross-entropy"}
    arms = list(product(configs, seeds))
    rows, folds, selections, validation = [], [], [], []
    threads = torch.get_num_threads()
    debug_mode = torch.get_deterministic_debug_mode()
    try:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(n_threads)
        devices = [torch.cuda.current_device()] if device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            for fold, held_out, train, test in _outer_splits(y, subjects, grouping=grouping, seed=fold_seed):
                info = {"Fold": fold, "Test_Subject": held_out}
                train_info = {**info, "Train_Subjects": tuple(sorted(set(subjects[train]))),
                              "Train_Rows": len(train), "Test_Rows": len(test)}
                folds.append(train_info)
                labels = y[train]
                train_subjects = subjects[train]
                target = _training_targets(labels, train_subjects, seeds, fold, shuffle_labels)
                skipped = np.bincount(labels, minlength=2).min() < 2
                if not skipped:
                    winners = {arm: (epochs, np.nan) for arm in arms}
                    if candidates:
                        winners, details = _tune(train, labels, train_subjects, target,
                                                 arms, prepare_split, make_model, options, info)
                        validation.extend(details)
                    outer = prepare_split(train, test)
                for config, seed in arms:
                    identity = {"Model": _model_name(backbone, config, seed), "Config": config, "Seed": seed}
                    block = table.iloc[test][["Subject_ID", "Trial_ID", "Label"]].copy()
                    block = block.assign(**identity, Feature_Set=feature_set, Fold=fold)
                    record = {**train_info, **identity}
                    if skipped:
                        block["Probability"] = block["Prediction"] = np.nan
                        selections.append({**record, "Status": "skipped", "Selected_Epochs": None,
                                           "Inner_Balanced_Accuracy": np.nan, "PCA_Components": None,
                                           "Reason": "fewer than two training trials in a class"})
                    else:
                        best, score = winners[config, seed]
                        probability = _train(outer, target[seed], config, seed, [best], make_model, options)[best]
                        block["Probability"] = probability
                        block["Prediction"] = (probability >= .5).astype(int)
                        selections.append({**record, "Status": "ok", "Selected_Epochs": best, "Reason": "",
                                           "Inner_Balanced_Accuracy": score, "PCA_Components": outer.pca_components})
                    rows.append(block)
    finally:
        torch.set_num_threads(threads)
        torch.set_deterministic_debug_mode(debug_mode)
    validation_columns = ["Fold", "Test_Subject", "Inner_Fold", "Train_Subjects", "Validation_Subjects",
                          "Train_Rows", "Validation_Rows", "PCA_Components", "Model", "Config", "Seed",
                          "Epochs", "Class0_N", "Class1_N", "Class0_Correct", "Class1_Correct", "Balanced_Accuracy"]
    return {"predictions": pd.concat(rows, ignore_index=True), "folds": pd.DataFrame(folds),
            "selection": pd.DataFrame(selections), "validation": pd.DataFrame(validation, columns=validation_columns),
            "configuration": options}

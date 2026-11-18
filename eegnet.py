"""Compact EEGNet for trial waveforms, with optional AGFL over time positions."""
from __future__ import annotations

import math
from collections import OrderedDict
from numbers import Integral, Real

import torch
from torch import nn

from .agfl import AGFLLayer, build_edge_mask


class EEGNet(nn.Module):
    """EEGNet-8,2 backbone with the same convolutional encoder and head in each arm.

    Inputs have shape (batch, EEG channels, samples). Each node is a pooled time
    position with 16 learned coordinates. The shorter second pooling interval
    keeps about eight nodes per second for the optional temporal graph. A
    1.7-second trial at 256 Hz therefore has 13 positions. The authors' 128 Hz
    implementation uses a second pool of 8. The baseline and graph arms here
    both use a pool of 4 at the same intermediate 32 Hz rate.
    """

    spatial_max_norm = 1.0
    classifier_max_norm = 0.25

    def __init__(self, n_channels, n_samples, sfreq, *, config="none", a=16,
                 hops=2, kappa=5, dropout=0.5):
        super().__init__()
        if config not in ("none", "k0", "inter"):
            raise ValueError("EEGNet config must be 'none', 'k0', or 'inter'.")
        for name, value in (("n_channels", n_channels), ("n_samples", n_samples),
                            ("a", a), ("kappa", kappa)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(hops, bool) or not isinstance(hops, Integral) or hops < 0:
            raise ValueError("hops must be a nonnegative integer.")
        if config == "inter" and hops == 0:
            raise ValueError("The inter graph needs at least one hop. Use k0 for zero hops.")
        if isinstance(sfreq, bool) or not isinstance(sfreq, Real) or not math.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("sfreq must be a positive, finite sampling rate in Hz.")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.config = config
        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.sfreq = float(sfreq)
        temporal_filters, depth_multiplier = 8, 2
        self.token_width = temporal_filters * depth_multiplier
        # At 128 Hz the first kernel is 64 samples and the first pool is 4.
        # The separable kernel and second pool operate at the reduced rate.
        temporal_kernel = max(1, round(self.sfreq / 2))
        pool1 = max(1, round(self.sfreq / 32))
        reduced_sfreq = self.sfreq / pool1
        separable_kernel = max(1, round(reduced_sfreq / 2))
        pool2 = max(1, round(reduced_sfreq / 8))
        self.n_tokens = (self.n_samples // pool1) // pool2
        if self.n_tokens < 2:
            raise ValueError(f"EEGNet needs at least {2 * pool1 * pool2} samples at {sfreq} Hz "
                             "to compare graph arms over at least two time positions.")

        self.encoder = nn.Sequential(OrderedDict([
            # Explicit same padding preserves time length for even kernels.
            ("temporal_pad", nn.ZeroPad2d(((temporal_kernel - 1) // 2, temporal_kernel // 2, 0, 0))),
            ("temporal", nn.Conv2d(1, temporal_filters, (1, temporal_kernel), bias=False)),
            ("temporal_bn", nn.BatchNorm2d(temporal_filters, eps=1e-3, momentum=0.01)),
            ("spatial", nn.Conv2d(temporal_filters, self.token_width, (self.n_channels, 1),
                                  groups=temporal_filters, bias=False)),
            ("spatial_bn", nn.BatchNorm2d(self.token_width, eps=1e-3, momentum=0.01)),
            ("spatial_elu", nn.ELU()), ("pool1", nn.AvgPool2d((1, pool1))), ("dropout1", nn.Dropout(dropout)),
            ("separable_pad", nn.ZeroPad2d(((separable_kernel - 1) // 2, separable_kernel // 2, 0, 0))),
            ("depthwise_time", nn.Conv2d(self.token_width, self.token_width, (1, separable_kernel),
                                         groups=self.token_width, bias=False)),
            ("pointwise", nn.Conv2d(self.token_width, self.token_width, 1, bias=False)),
            ("separable_bn", nn.BatchNorm2d(self.token_width, eps=1e-3, momentum=0.01)),
            ("separable_elu", nn.ELU()), ("pool2", nn.AvgPool2d((1, pool2))), ("dropout2", nn.Dropout(dropout)),
        ]))
        # Instantiate shared parameters first so a paired seed gives identical
        # encoder/head weights. No forward pass is needed to determine their shapes.
        self.head = nn.Linear(self.n_tokens * self.token_width, 2)
        self.graph = None
        if config != "none":
            self.graph = AGFLLayer(self.token_width, int(a), 0 if config == "k0" else int(hops), int(kappa))
        self.register_buffer("edge_mask", build_edge_mask(self.n_tokens) if self.graph is not None else None,
                             persistent=False)
        self.architecture = {
            "backbone": "EEGNet-8,2", "config": config, "n_channels": self.n_channels,
            "n_samples": self.n_samples, "sfreq": self.sfreq, "temporal_filters": temporal_filters,
            "depth_multiplier": depth_multiplier, "temporal_kernel_samples": temporal_kernel,
            "separable_kernel_pooled_samples": separable_kernel, "pool1": pool1, "pool2": pool2,
            "n_tokens": self.n_tokens, "token_width": self.token_width,
            "token_stride_seconds": (pool1 * pool2) / self.sfreq, "dropout": float(dropout),
            "graph_hops": int(hops) if config == "inter" else 0,
            "attention_width": int(a) if self.graph is not None else None,
            "kappa": int(kappa) if config == "inter" else None,
            "reference_doi": "10.1088/1741-2552/aace8c",
            "reference_implementation": "https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py",
            "pooling_adaptation": "Second pool targets 8 time positions/second for 1.7-second trials.",
            "spatial_filter_max_norm": self.spatial_max_norm,
            "classifier_max_norm": self.classifier_max_norm,
        }

    @torch.no_grad()
    def constrain_weights(self):
        """Project each spatial filter and each classifier row after an optimizer step."""
        for weight, limit in ((self.encoder.spatial.weight, self.spatial_max_norm),
                              (self.head.weight, self.classifier_max_norm)):
            axes = tuple(range(1, weight.ndim))
            norms = torch.linalg.vector_norm(weight, dim=axes, keepdim=True)
            weight.div_((norms / limit).clamp_min_(1.0))

    def encode_tokens(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Return ordered time nodes (batch, positions, learned features)."""
        if waveforms.ndim != 3 or tuple(waveforms.shape[1:]) != (self.n_channels, self.n_samples):
            raise ValueError(f"Expected EEG waveforms shaped (batch, {self.n_channels}, {self.n_samples}).")
        features = self.encoder(waveforms.unsqueeze(1))
        return features.squeeze(2).transpose(1, 2)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(waveforms)
        if self.graph is not None:
            tokens = self.graph(tokens, self.edge_mask)
        return self.head(tokens.flatten(start_dim=1))

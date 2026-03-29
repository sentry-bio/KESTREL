"""
kestrel_bio.spectrum_branch
===========================
Inference-only: SpectrumBranch model, adaptive pyramid CLR feature
extraction, and hyperbolic Fréchet mean ensemble.

No training code, no dataset classes — just what's needed to load a
checkpoint and run predictions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# Hyperbolic operations
# ---------------------------------------------------------------------------

class HyperbolicOps:
    """Minimal Poincaré ball operations (curvature κ, radius bound 1/√κ)."""

    def __init__(self, kappa: float = 1.25, eps: float = 1e-5):
        self.kappa = kappa
        self.eps   = eps

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        k      = self.kappa
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_k = k ** 0.5
        coeff  = torch.tanh(sqrt_k * v_norm / 2) / (sqrt_k * v_norm)
        return coeff * v

    def project(self, x: torch.Tensor, max_norm: float = 0.95) -> torch.Tensor:
        max_r = max_norm / (self.kappa ** 0.5)
        norm  = x.norm(dim=-1, keepdim=True)
        return x * (max_r / norm.clamp(min=self.eps)).clamp(max=1.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpectrumBranch(nn.Module):
    """
    MLP: k-mer CLR features → Poincaré ball coordinates.

    Architecture: Input → [Linear → LayerNorm → GELU → Dropout] × N
                → tangent → expmap0 → project → Poincaré ball
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int   = 512,
        embed_dim:  int   = 128,
        n_layers:   int   = 3,
        dropout:    float = 0.1,
        kappa:      float = 1.25,
    ):
        super().__init__()
        self.kappa = kappa
        self.h_ops = HyperbolicOps(kappa=kappa)

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        dim = hidden_dim
        for _ in range(n_layers - 1):
            next_dim = max(embed_dim, dim // 2)
            layers.extend([
                nn.Linear(dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            dim = next_dim
        layers.append(nn.Linear(dim, embed_dim))
        self.mlp         = nn.Sequential(*layers)
        self.domain_head = nn.Linear(embed_dim, 3)   # Bacteria / Archaea / Eukaryota

    def forward(self, spectrum: torch.Tensor) -> dict:
        tangent       = self.mlp(spectrum)
        coords        = self.h_ops.project(self.h_ops.expmap0(tangent))
        domain_logits = self.domain_head(tangent)
        return {"coords": coords, "tangent": tangent, "domain_logits": domain_logits}


# ---------------------------------------------------------------------------
# Fast numpy feature extraction
# ---------------------------------------------------------------------------

def build_kmer_tables(k_values: List[int]) -> dict:
    """Pre-compute per-k stride-tricks index tables."""
    tables = {}
    for k in k_values:
        vocab   = 4 ** k
        mapping = np.full(128, -1, dtype=np.int8)
        mapping[65] = 0; mapping[67] = 1; mapping[71] = 2; mapping[84] = 3
        powers  = 4 ** np.arange(k - 1, -1, -1, dtype=np.int32)
        tables[k] = (mapping, powers, vocab)
    return tables


def _kmer_spectrum_fast(seq: str, k: int, tables: dict) -> np.ndarray:
    from numpy.lib.stride_tricks import as_strided
    mapping, powers, vocab = tables[k]
    raw     = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    raw     = np.minimum(raw, 127)
    encoded = mapping[raw].astype(np.int16)
    L = len(encoded)
    if L < k:
        return np.zeros(vocab, dtype=np.float32)
    s       = encoded.strides[0]
    windows = as_strided(encoded, shape=(L - k + 1, k), strides=(s, s))
    valid   = (windows >= 0).all(axis=1)
    vw      = windows[valid].astype(np.int32)
    if len(vw) == 0:
        return np.zeros(vocab, dtype=np.float32)
    hashes = (vw * powers).sum(axis=1)
    counts = np.bincount(hashes, minlength=vocab).astype(np.float32)
    t = counts.sum()
    if t > 0:
        counts /= t
    return counts


def _clr_fast(seq: str, k_values: List[int], tables: dict) -> np.ndarray:
    parts = []
    for k in k_values:
        s  = _kmer_spectrum_fast(seq, k, tables) + 1e-8
        ls = np.log(s)
        parts.append(ls - ls.mean())
    return np.concatenate(parts)


def _n_sub_windows(L: int, max_sub: int, min_win_bp: int) -> int:
    for n in [4, 2]:
        if n <= max_sub and L // n >= min_win_bp:
            return n
    return 0


def make_extract_fn(k_values, max_sub_windows, min_win_bp, max_dna_len, spectrum_dim):
    """Return a vectorised feature function bound to checkpoint parameters."""
    tables = build_kmer_tables(k_values)

    def extract(dna: str) -> np.ndarray:
        L     = len(dna)
        n_sub = _n_sub_windows(L, max_sub_windows, min_win_bp)
        parts = [_clr_fast(dna, k_values, tables)]
        if n_sub > 0:
            step = L // n_sub
            for i in range(n_sub):
                start = i * step
                end   = (start + step) if i < n_sub - 1 else L
                win   = dna[start:end]
                parts.append(_clr_fast(win, k_values, tables) if len(win) >= 4
                              else np.zeros(spectrum_dim, dtype=np.float32))
        for _ in range(max_sub_windows - n_sub):
            parts.append(np.zeros(spectrum_dim, dtype=np.float32))
        mask          = np.array([1.0] * n_sub + [0.0] * (max_sub_windows - n_sub),
                                  dtype=np.float32)
        length_scalar = np.array([np.log(max(L, 1) / max_dna_len)], dtype=np.float32)
        return np.concatenate(parts + [mask, length_scalar]).astype(np.float32)

    return extract


# ---------------------------------------------------------------------------
# Hyperbolic Fréchet mean
# ---------------------------------------------------------------------------

def hyperbolic_frechet_mean(
    tangent_vecs: np.ndarray,
    kappa:        float,
    n_iter:       int = 10,
) -> np.ndarray:
    """
    Iterative Fréchet mean in the Poincaré ball (curvature κ).

    Args:
        tangent_vecs: (N, D) float32 — per-fragment tangent predictions.
        kappa:        curvature (positive scalar).
        n_iter:       Riemannian gradient iterations.

    Returns:
        (D,) float32 mean tangent vector.
    """
    tangent_vecs = np.asarray(tangent_vecs, dtype=np.float64)
    N, D = tangent_vecs.shape
    if N == 1:
        return tangent_vecs[0].astype(np.float32)

    c = 1.0 / kappa

    def exp0(v):
        norm = np.linalg.norm(v) + 1e-10
        return (np.tanh(np.sqrt(c) * norm / 2) / (np.sqrt(c) * norm)) * v

    def log_map(x, y):
        c_xx = c * np.dot(x, x)
        c_xy = c * np.dot(x, y)
        c_yy = c * np.dot(y, y)
        mob  = ((1 + 2*c_xy + c_yy) * x + (1 - c_xx) * y) / (1 + 2*c_xy + c_xx*c_yy + 1e-10)
        norm_mob = np.linalg.norm(mob) + 1e-10
        lx   = 2.0 / (1 - c_xx + 1e-10)
        return (2.0 / (np.sqrt(c) * lx)) * np.arctanh(np.sqrt(c) * norm_mob) / norm_mob * mob

    points = np.stack([exp0(v) for v in tangent_vecs])
    mu     = exp0(tangent_vecs.mean(axis=0))

    for _ in range(n_iter):
        logs = np.stack([log_map(mu, p) for p in points])
        mu   = exp0(log_map(np.zeros(D), mu) + logs.mean(axis=0))

    norm_mu = np.linalg.norm(mu) + 1e-10
    return ((2.0 / np.sqrt(c)) * np.arctanh(np.sqrt(c) * norm_mu) / norm_mu * mu).astype(np.float32)

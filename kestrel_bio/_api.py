"""
kestrel_bio._api — high-level Python API
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from kestrel_bio.spectrum_branch import (
    SpectrumBranch,
    make_extract_fn,
    hyperbolic_frechet_mean,
)

DOMAIN_NAMES = {0: "Bacteria", 1: "Archaea", 2: "Eukaryota"}


def load_model(model_path: str, device: str = "cpu"):
    """
    Load a KESTREL checkpoint.

    Returns:
        (model, extract_fn, kappa) — pass all three to classify_sequence.

    Example:
        model, extract, kappa = load_model("kestrel_nano_best.pt")
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    model = SpectrumBranch(
        input_dim  = ckpt["input_dim"],
        hidden_dim = ckpt["hidden_dim"],
        embed_dim  = ckpt["embed_dim"],
        n_layers   = ckpt.get("n_layers", 4),
        kappa      = ckpt["kappa"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    k_values     = ckpt.get("k_values", [4, 5, 6])
    max_sub      = ckpt.get("max_sub_windows", 4)
    min_win_bp   = ckpt.get("min_win_bp", 100)
    max_dna_len  = ckpt.get("max_dna_len", 2611)
    spectrum_dim = sum(4**k for k in k_values)
    extract      = make_extract_fn(k_values, max_sub, min_win_bp, max_dna_len, spectrum_dim)

    return model, extract, ckpt["kappa"]


def classify_sequence(
    model,
    extract_fn,
    kappa:      float,
    dna:        str,
    n_ensemble: int = 1,
    frag_bp:    int = 509,
    device:     str = "cpu",
) -> dict:
    """
    Classify a single DNA string.

    Args:
        model:       SpectrumBranch (from load_model)
        extract_fn:  feature function (from load_model)
        kappa:       curvature scalar (from load_model)
        dna:         DNA sequence string
        n_ensemble:  number of random sub-fragments to ensemble
        frag_bp:     sub-fragment length when n_ensemble > 1
        device:      "cpu" or "cuda"

    Returns:
        dict with keys: domain, domain_conf, domain_probs, coords,
                        radial_depth, length_bp, n_ensemble
    """
    dna = dna.upper().replace(" ", "").replace("\n", "")
    L   = len(dna)
    if L < 4:
        raise ValueError(f"Sequence too short ({L} bp, minimum 4 bp)")

    if n_ensemble == 1:
        frags = [extract_fn(dna)]
    else:
        rng      = np.random.default_rng(hash(dna[:64]) & 0xFFFFFFFF)
        frags    = []
        frag_len = min(L, frag_bp)
        for _ in range(n_ensemble):
            start = int(rng.integers(0, max(1, L - frag_len + 1)))
            frag  = dna[start: start + frag_len]
            frags.append(extract_fn(frag if len(frag) >= 4 else dna[:max(4, frag_len)]))

    feat = torch.from_numpy(np.stack(frags)).to(device)
    with torch.no_grad():
        out = model(feat)

    tang_vecs  = out["tangent"].cpu().numpy()
    dom_logits = out["domain_logits"].cpu().numpy()

    tang_mean = (hyperbolic_frechet_mean(tang_vecs, kappa)
                 if n_ensemble > 1 else tang_vecs[0])
    dom_probs = torch.softmax(torch.from_numpy(dom_logits.mean(axis=0)), dim=0).numpy()
    dom_pred  = int(dom_probs.argmax())
    dom_conf  = float(dom_probs[dom_pred])

    c      = 1.0 / kappa
    radial = float(np.tanh(np.sqrt(c) * np.linalg.norm(tang_mean) / 2))

    return {
        "domain":       DOMAIN_NAMES.get(dom_pred, str(dom_pred)),
        "domain_conf":  round(dom_conf, 4),
        "domain_probs": {DOMAIN_NAMES[i]: round(float(dom_probs[i]), 4)
                         for i in range(len(dom_probs))},
        "coords":       tang_mean.tolist(),
        "radial_depth": round(radial, 4),
        "length_bp":    L,
        "n_ensemble":   n_ensemble,
    }

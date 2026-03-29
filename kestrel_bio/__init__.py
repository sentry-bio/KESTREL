"""
kestrel-bio — Fast k-mer phylogenetic classifier
=================================================
CPU-native companion to BiosphereAtlas. Places DNA sequences into
the Poincaré ball coordinate space using adaptive pyramid CLR features.

Quick start:

    # CLI (installed via pip)
    kestrel download nano
    kestrel classify --model ~/.kestrel/models/kestrel_nano_best.pt reads.fastq

    # Python API
    from kestrel_bio import load_model, classify_sequence

    model, extract, kappa = load_model("/path/to/kestrel_nano_best.pt")
    result = classify_sequence(model, extract, kappa, "ATCGATCG...")
    print(result["domain"], result["domain_conf"])

Public API:
    SpectrumBranch       — the MLP model class
    make_extract_fn      — build a feature extraction function from checkpoint params
    hyperbolic_frechet_mean — manifold-aware ensemble averaging
    load_model           — load a checkpoint, return (model, extract_fn, kappa)
    classify_sequence    — classify a single DNA string
"""

from kestrel_bio.spectrum_branch import (
    SpectrumBranch,
    HyperbolicOps,
    make_extract_fn,
    build_kmer_tables,
    hyperbolic_frechet_mean,
)
from kestrel_bio._api import load_model, classify_sequence

__version__ = "1.0.0"
__all__ = [
    "SpectrumBranch",
    "HyperbolicOps",
    "make_extract_fn",
    "build_kmer_tables",
    "hyperbolic_frechet_mean",
    "load_model",
    "classify_sequence",
]

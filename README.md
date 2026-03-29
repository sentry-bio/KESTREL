# KESTREL

**Fast k-mer phylogenetic classifier** — CPU-native companion to [BiosphereAtlas](https://biosphereatlas.com).

Places DNA sequences into the Poincare ball coordinate space (κ = 5/4) using adaptive pyramid CLR features from k-mer spectra. No GPU required. No database required.

## Quick Start

```bash
pip install kestrel-bio
kestrel download nano
kestrel classify reads.fastq
```

## Python API

```python
from kestrel_bio import load_model, classify_sequence

model, extract, kappa = load_model("kestrel_nano_best.pt")
result = classify_sequence(model, extract, kappa, "ATCGATCG...")

print(result["domain"])       # "Bacteria"
print(result["domain_conf"])  # 0.987
print(result["coords"])       # 129-dim tangent vector
```

## With Canonical Coordinate System

When paired with the [Canonical Coordinate System](https://github.com/sentry-bio/canonical-coordinate-system), KESTREL provides interpretable taxonomic addresses:

```python
from kestrel_bio import load_model
from canonical_hybrid import HybridEngine

model, extract, kappa = load_model("kestrel_nano_best.pt")
engine = HybridEngine.load("tessellation_K25")

result = engine.classify_kestrel("ATCGATCG...", model, extract, kappa)
print(result.cell_domain)     # "Bacteria"
print(result.cell_family)     # "Lactobacillaceae"
print(result.cell_confidence) # 2.3
```

## Models

| Model | Params | Size | cos @ 2.6kbp | Domain acc @ 509bp |
|-------|--------|------|-------------|-------------------|
| **KESTREL v5** | 28.2M | 108 MB | 0.978 | 94.4% |
| **KESTREL nano** | 13.9M | 54 MB | 0.972 | 94.4% |

Both models are available via `kestrel download` or from [Zenodo](https://doi.org/10.5281/zenodo.PENDING).

## CLI

```bash
# Download model weights
kestrel download nano       # 54 MB, recommended
kestrel download full       # 108 MB, highest accuracy

# Classify sequences
kestrel classify reads.fastq --model ~/.kestrel/models/kestrel_nano_best.pt

# Model info
kestrel info --model ~/.kestrel/models/kestrel_nano_best.pt
```

## How It Works

1. **K-mer spectrum extraction**: Adaptive pyramid CLR features from k-mer frequencies (k=4,5,6)
2. **MLP projection**: 4-layer network projects features to 129-dim tangent space
3. **Poincare ball embedding**: Exponential map places sequences on the hyperbolic manifold at κ = 5/4
4. **Domain classification**: Linear head on tangent vectors → Bacteria/Archaea/Eukaryota
5. **Canonical addressing** (optional): Voronoi tessellation maps tangent vectors to interpretable cell IDs

The tangent-space embedding is compatible with BiosphereAtlas — both models map to the same manifold (cos = 0.974 agreement over 47K genomes).

## Architecture

```
DNA sequence
  ↓ [k-mer CLR features, k=4,5,6]
  ↓ [adaptive pyramid windowing]
Feature vector (26,885-dim)
  ↓ [LayerNorm → Linear → GELU] × 4
  ↓ [Linear → Poincare projection]
Tangent vector (129-dim) + Domain logits (3-dim)
```

## Performance

Tested on 5,000 genomes against BiosphereAtlas v15.5 teacher embeddings:

- **cos(KESTREL, Atlas) = 0.974** — near-identical manifold directions
- **Domain accuracy: 99.2%** — reliable three-domain classification
- **Voronoi cell agreement: 85.6%** at K=25 resolution
- **Latency: <1ms** on CPU (numpy only, no GPU)

## License

MIT. Created by [Sentry Bio](https://sentry.bio).

## Citation

```bibtex
@software{kestrel2026,
  title={KESTREL: Fast k-mer phylogenetic classifier},
  author={Fenn, Rohit and Fenn, Amit},
  year={2026},
  url={https://github.com/sentry-bio/KESTREL},
}
```

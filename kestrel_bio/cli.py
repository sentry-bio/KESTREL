#!/usr/bin/env python3
"""
kestrel_bio.cli — KESTREL command-line interface
=================================================

Commands:
  classify   Classify sequences from FASTA / FASTQ
  info       Print model metadata
  download   Download model weights from Zenodo
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from kestrel_bio.spectrum_branch import (
    SpectrumBranch,
    make_extract_fn,
    hyperbolic_frechet_mean,
)

DOMAIN_NAMES = {0: "Bacteria", 1: "Archaea", 2: "Eukaryota"}

# Zenodo record — fill in after deposit
_ZENODO_RECORD = "PENDING"   # e.g. "10.5281/zenodo.1234567"
_MODELS = {
    "nano": {
        "filename": "kestrel_nano_best.pt",
        "url":      f"https://zenodo.org/record/{_ZENODO_RECORD}/files/kestrel_nano_best.pt",
        "sha256":   "PENDING",   # fill in after deposit
        "size_mb":  54,
        "description": "13.9M params — default, 13ms CPU, 94% domain acc @ 509bp",
    },
    "full": {
        "filename": "kestrel_v5_best.pt",
        "url":      f"https://zenodo.org/record/{_ZENODO_RECORD}/files/kestrel_v5_best.pt",
        "sha256":   "PENDING",
        "size_mb":  108,
        "description": "28.2M params — higher accuracy, ~20ms CPU, 0.978 cos @ 2611bp",
    },
}

DEFAULT_MODEL_DIR = Path.home() / ".kestrel" / "models"


# ---------------------------------------------------------------------------
# FASTA / FASTQ parser
# ---------------------------------------------------------------------------

def _open(path: str):
    if path == "-":
        return sys.stdin
    p = Path(path)
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


def iter_sequences(path: str):
    """Yield (id, sequence) from FASTA or FASTQ (auto-detected, gzip ok)."""
    with _open(path) as fh:
        first = fh.readline()
        if not first:
            return
        if hasattr(fh, "seek"):
            fh.seek(0)

    with _open(path) as fh:
        first_char = None
        buf_id = buf_seq = None
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if first_char is None:
                first_char = line[0]

            if first_char == ">":
                if line.startswith(">"):
                    if buf_id is not None:
                        yield buf_id, buf_seq
                    buf_id  = line[1:].split()[0]
                    buf_seq = ""
                else:
                    buf_seq += line.upper()
            elif first_char == "@":
                seq_id = line[1:].split()[0]
                seq    = next(fh).rstrip().upper()
                next(fh)   # +
                next(fh)   # qual
                yield seq_id, seq

        if first_char == ">" and buf_id is not None:
            yield buf_id, buf_seq


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str):
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


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def cmd_classify(args):
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model, extract, kappa = load_model(args.model, device)

    teacher_coords = teacher_acc_idx = None
    if args.teacher:
        t = np.load(args.teacher, allow_pickle=False)
        teacher_acc_idx = {a: i for i, a in enumerate(t["accessions"].tolist())}
        teacher_coords  = t["coords"]

    n_ensemble = args.ensemble

    if args.output_format == "tsv":
        hdr = ["id", "length_bp", "domain", "domain_conf", "radial_depth"]
        if teacher_coords is not None:
            hdr.append("cos_to_teacher")
        if args.coords:
            hdr.append("coords")
        print("\t".join(hdr))

    results = []
    n_seqs  = 0
    t0      = time.time()

    for seq_id, seq in iter_sequences(args.input):
        L = len(seq)
        if L < 4:
            continue

        if n_ensemble == 1:
            frags = [extract(seq)]
        else:
            rng      = np.random.default_rng(hash(seq_id) & 0xFFFFFFFF)
            frags    = []
            frag_len = min(L, args.frag_bp)
            for _ in range(n_ensemble):
                start = int(rng.integers(0, max(1, L - frag_len + 1)))
                frag  = seq[start: start + frag_len]
                frags.append(extract(frag if len(frag) >= 4 else seq[:max(4, frag_len)]))

        feat_batch = torch.from_numpy(np.stack(frags)).to(device)
        with torch.no_grad():
            out = model(feat_batch)

        tang_vecs  = out["tangent"].cpu().numpy()
        dom_logits = out["domain_logits"].cpu().numpy()

        tang_mean = hyperbolic_frechet_mean(tang_vecs, kappa) if n_ensemble > 1 else tang_vecs[0]
        dom_probs = torch.softmax(torch.from_numpy(dom_logits.mean(axis=0)), dim=0).numpy()
        dom_pred  = int(dom_probs.argmax())
        dom_conf  = float(dom_probs[dom_pred])

        c      = 1.0 / kappa
        radial = float(np.tanh(np.sqrt(c) * np.linalg.norm(tang_mean) / 2))

        row = {
            "id":           seq_id,
            "length_bp":    L,
            "domain":       DOMAIN_NAMES.get(dom_pred, str(dom_pred)),
            "domain_conf":  round(dom_conf, 4),
            "radial_depth": round(radial, 4),
        }

        if teacher_coords is not None and seq_id in teacher_acc_idx:
            tidx  = teacher_acc_idx[seq_id]
            t_vec = teacher_coords[tidx]
            cos   = float(F.cosine_similarity(
                torch.from_numpy(tang_mean).unsqueeze(0),
                torch.from_numpy(t_vec).unsqueeze(0),
            ).item())
            row["cos_to_teacher"] = round(cos, 4)

        if args.coords:
            row["coords"] = tang_mean.tolist()

        n_seqs += 1
        if args.output_format == "tsv":
            print("\t".join(str(row.get(h, "")) for h in hdr))
            sys.stdout.flush()
        else:
            results.append(row)

    elapsed = time.time() - t0
    if args.output_format == "json":
        print(json.dumps(results, indent=2))
    print(f"\n# {n_seqs} sequences in {elapsed:.1f}s ({n_seqs/elapsed:.0f} seq/s)",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def cmd_info(args):
    ckpt     = torch.load(args.model, map_location="cpu", weights_only=False)
    k        = ckpt.get("k_values", [4, 5, 6])
    n_params = sum(v.numel() for v in ckpt["model_state_dict"].values())
    size_mb  = sum(v.numel() * v.element_size()
                   for v in ckpt["model_state_dict"].values()) / 1e6
    m = ckpt.get("val_metrics", {})
    print(f"KESTREL model : {args.model}")
    print(f"  Architecture: {ckpt['input_dim']} → {ckpt['hidden_dim']} × {ckpt.get('n_layers',4)} → {ckpt['embed_dim']}")
    print(f"  Parameters  : {n_params:,}  ({size_mb:.1f} MB FP32 / {size_mb/2:.1f} MB FP16)")
    print(f"  k-values    : {k}")
    print(f"  Max sub-wins: {ckpt.get('max_sub_windows', 4)}")
    print(f"  Min win bp  : {ckpt.get('min_win_bp', 100)}")
    print(f"  κ (kappa)   : {ckpt['kappa']}")
    print(f"  Epoch       : {ckpt.get('epoch', '?')}")
    print(f"  Val cos mean: {m.get('cos_sim_mean', '?'):.4f}")
    print(f"  Val cos med : {m.get('cos_sim_median', '?'):.4f}")
    print(f"  Val dom acc : {m.get('domain_acc', '?'):.1%}")


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _progress_hook(count, block_size, total_size):
    if total_size <= 0:
        return
    pct  = min(count * block_size / total_size * 100, 100)
    done = int(pct / 2)
    bar  = "#" * done + "." * (50 - done)
    print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)


def cmd_download(args):
    model_key = args.model_name.lower()
    if model_key not in _MODELS:
        sys.exit(f"Unknown model '{model_key}'. Choose from: {', '.join(_MODELS)}")

    info = _MODELS[model_key]

    if _ZENODO_RECORD == "PENDING":
        print("Zenodo deposit is not yet published.")
        print("Once the DOI is live, update _ZENODO_RECORD in kestrel_bio/cli.py")
        print(f"Expected URL: {info['url']}")
        sys.exit(1)

    dest_dir = Path(args.dest) if args.dest else DEFAULT_MODEL_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / info["filename"]

    if dest.exists() and not args.force:
        print(f"Already downloaded: {dest}")
        print("Use --force to re-download.")
        return

    print(f"Downloading KESTREL {model_key} ({info['size_mb']} MB)...")
    print(f"  Source : {info['url']}")
    print(f"  Dest   : {dest}")

    urllib.request.urlretrieve(info["url"], dest, reporthook=_progress_hook)
    print()   # newline after progress bar

    if info["sha256"] != "PENDING":
        print("  Verifying checksum...", end=" ", flush=True)
        actual = _sha256(dest)
        if actual != info["sha256"]:
            dest.unlink()
            sys.exit(f"Checksum mismatch!\n  expected: {info['sha256']}\n  got:      {actual}")
        print("OK")

    print(f"\nModel saved to {dest}")
    print(f"\nUsage:")
    print(f"  kestrel classify --model {dest} input.fasta")
    print(f"  kestrel info     --model {dest}")


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="kestrel",
        description="KESTREL — fast k-mer phylogenetic classifier (BiosphereAtlas companion)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- classify --
    p = sub.add_parser("classify", help="Classify sequences from FASTA/FASTQ")
    p.add_argument("input",               help="FASTA/FASTQ file (- for stdin, .gz ok)")
    p.add_argument("--model",  "-m",      required=True, help="Path to .pt checkpoint")
    p.add_argument("--teacher",           help="teacher_coords.npz for coherence scoring")
    p.add_argument("--ensemble", "-e",    type=int, default=1,
                   help="Number of random sub-fragments to ensemble (default: 1 = full seq)")
    p.add_argument("--frag-bp",           type=int, default=509,
                   help="Sub-fragment length in bp for ensemble mode (default: 509)")
    p.add_argument("--format", "-f",      dest="output_format",
                   choices=["tsv", "json"], default="tsv")
    p.add_argument("--coords",            action="store_true",
                   help="Include 129-d coordinate vector in output")
    p.add_argument("--cpu",               action="store_true", help="Force CPU inference")

    # -- info --
    p = sub.add_parser("info", help="Print model metadata")
    p.add_argument("--model", "-m", required=True)

    # -- download --
    p = sub.add_parser("download", help="Download model weights from Zenodo")
    p.add_argument("model_name",        choices=list(_MODELS),
                   metavar="MODEL",     help=f"Model to download: {', '.join(_MODELS)}")
    p.add_argument("--dest",            help=f"Destination directory (default: {DEFAULT_MODEL_DIR})")
    p.add_argument("--force",           action="store_true", help="Re-download if already present")

    args = parser.parse_args()
    {"classify": cmd_classify, "info": cmd_info, "download": cmd_download}[args.command](args)


if __name__ == "__main__":
    main()

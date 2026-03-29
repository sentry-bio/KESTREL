#!/usr/bin/env python3
"""
zenodo/upload.py — Deposit KESTREL model weights to Zenodo
===========================================================
Run this on the inference server where the .pt files live.

Prerequisites:
  pip install requests
  export ZENODO_TOKEN=<your personal access token from zenodo.org/account/settings/applications>

Usage:
  python zenodo/upload.py --sandbox          # test run (sandbox.zenodo.org)
  python zenodo/upload.py                    # real deposit (zenodo.org)

After publishing:
  1. Copy the DOI into kestrel_bio/cli.py (_ZENODO_RECORD)
  2. Copy the DOI into kestrel_bio/pyproject.toml (Zenodo URL)
  3. Run: python zenodo/upload.py --verify <DOI>  to check sha256 matches
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

MODEL_FILES = {
    "kestrel_nano_best.pt": "/home/rohit/kestrel_nano/kestrel_nano_best.pt",
    "kestrel_v5_best.pt":   "/home/rohit/kestrel_v5/kestrel_v5_best.pt",
}

METADATA_PATH = Path(__file__).parent / "metadata.json"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def upload(sandbox: bool, token: str):
    try:
        import requests
    except ImportError:
        sys.exit("pip install requests")

    base = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create empty deposit
    print(f"Creating deposit on {base} ...")
    r = requests.post(f"{base}/api/deposit/depositions",
                      json={}, headers=headers)
    r.raise_for_status()
    dep   = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print(f"  Deposit ID: {dep_id}")
    print(f"  Edit URL:   {dep['links']['html']}")

    # 2. Upload files
    checksums = {}
    for name, local_path in MODEL_FILES.items():
        if not Path(local_path).exists():
            print(f"  SKIP (not found): {local_path}")
            continue
        size_mb = Path(local_path).stat().st_size / 1e6
        print(f"  Uploading {name} ({size_mb:.0f} MB) ...", end=" ", flush=True)
        cksum = sha256(local_path)
        checksums[name] = cksum
        with open(local_path, "rb") as fh:
            r = requests.put(f"{bucket}/{name}", data=fh, headers=headers)
        r.raise_for_status()
        print("OK")
        print(f"    sha256: {cksum}")

    # 3. Set metadata
    print("  Setting metadata ...", end=" ", flush=True)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    r = requests.put(f"{base}/api/deposit/depositions/{dep_id}",
                     json={"metadata": meta}, headers=headers)
    r.raise_for_status()
    print("OK")

    # 4. Print checksums for cli.py
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Review and publish the deposit at:")
    print(f"   {dep['links']['html']}")
    print("\n2. Once published, copy the DOI record ID into kestrel_bio/cli.py:")
    print("   _ZENODO_RECORD = \"<record_id>\"  # e.g. 1234567")
    print("\n3. Update sha256 checksums in kestrel_bio/cli.py:")
    for name, cksum in checksums.items():
        key = "nano" if "nano" in name else "full"
        print(f'   _MODELS["{key}"]["sha256"] = "{cksum}"')
    print("\n4. Tag and push: git tag v1.0.0 && git push --tags")
    print("5. Build and upload to PyPI:")
    print("   cd kestrel_bio && python -m build && twine upload dist/*")

    return dep_id, checksums


def verify(doi: str, token: str):
    """Download each file from Zenodo and verify sha256 matches upload."""
    try:
        import requests
    except ImportError:
        sys.exit("pip install requests")

    record_id = doi.split(".")[-1]
    r = requests.get(f"https://zenodo.org/api/records/{record_id}")
    r.raise_for_status()
    files = r.json()["files"]
    print(f"Record has {len(files)} file(s):")
    for f in files:
        print(f"  {f['key']}  {f['checksum']}")


def main():
    parser = argparse.ArgumentParser(description="Deposit KESTREL weights to Zenodo")
    parser.add_argument("--sandbox", action="store_true",
                        help="Use sandbox.zenodo.org (no real deposit)")
    parser.add_argument("--verify",  metavar="DOI",
                        help="Verify an already-published deposit")
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set ZENODO_TOKEN environment variable first.\n"
                 "Get a token at: https://zenodo.org/account/settings/applications")

    if args.verify:
        verify(args.verify, token)
    else:
        upload(sandbox=args.sandbox, token=token)


if __name__ == "__main__":
    main()

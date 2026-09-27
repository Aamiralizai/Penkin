#!/usr/bin/env python3
"""Fetch the raw Affymetrix CEL files for GSE9825 and GSE12632 from NCBI GEO.

The files are included in the repository; this script re-downloads them or
verifies the copies present. They drive an optional QC and
normalisation-sensitivity step. The primary transcript-to-capacity analysis
uses the deposited series matrix, so the published values do not depend on
them. Run the workflow with ``--skip-raw-geo`` to omit the step.

Usage
-----
    python scripts/fetch_geo_raw.py            # download and verify
    python scripts/fetch_geo_raw.py --verify   # verify existing files only

About 70 MB is downloaded. Every file is checked against
``penkin/data/geo_raw/CHECKSUMS.txt``, which records the SHA-256 of the exact
files used for the published analysis, so a silent upstream change cannot pass
unnoticed.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "penkin" / "data" / "geo_raw"
MANIFEST = DEST / "CHECKSUMS.txt"

SERIES = {
    "GSE9825": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE9nnn/GSE9825/suppl/GSE9825_RAW.tar",
    "GSE12632": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE12nnn/GSE12632/suppl/GSE12632_RAW.tar",
}


def expected():
    if not MANIFEST.exists():
        sys.exit(f"Missing manifest: {MANIFEST.relative_to(ROOT)}")
    out = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            out[rel.strip()] = digest
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(quiet=False):
    exp = expected()
    missing, bad, ok = [], [], 0
    for rel, digest in sorted(exp.items()):
        p = DEST / rel
        if not p.exists():
            missing.append(rel)
        elif sha256(p) != digest:
            bad.append(rel)
        else:
            ok += 1
    if not quiet:
        print(f"{ok}/{len(exp)} files present and matching")
        for rel in missing:
            print(f"  missing  {rel}")
        for rel in bad:
            print(f"  MISMATCH {rel}")
    return missing, bad


def download():
    exp = expected()
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in SERIES.items():
        target = DEST / f"{name}_RAW"
        needed = [r for r in exp if r.startswith(f"{name}_RAW/") and not (DEST / r).exists()]
        if not needed:
            print(f"{name}: already complete, skipping")
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"{name}: downloading {url}")
        try:
            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
                with urllib.request.urlopen(url, timeout=120) as resp:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        tmp.write(chunk)
                tmp_path = Path(tmp.name)
        except (urllib.error.URLError, TimeoutError) as exc:
            sys.exit(
                f"\nDownload failed for {name}: {exc}\n"
                f"Fetch {url} manually, extract into {target.relative_to(ROOT)}, "
                f"then run: python scripts/fetch_geo_raw.py --verify"
            )
        print(f"{name}: extracting")
        with tarfile.open(tmp_path) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    member.name = Path(member.name).name  # flatten, ignore paths
                    tar.extract(member, target, filter="data")
        tmp_path.unlink()

    missing, bad = verify()
    if missing or bad:
        print("\nSome files are missing or do not match the published checksums.")
        return 1
    print("\nAll raw GEO files present and verified.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="check existing files without downloading")
    args = ap.parse_args()
    if args.verify:
        missing, bad = verify()
        sys.exit(1 if (missing or bad) else 0)
    sys.exit(download())


if __name__ == "__main__":
    main()

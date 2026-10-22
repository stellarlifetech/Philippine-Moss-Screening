#!/usr/bin/env python3
"""Reproduction check: rerun the full analysis from data/ into a temporary
folder and confirm that every published table in results/ is reproduced
(numerically, to 1e-9) and that the SHA-256 checksums of the inputs match.

Usage:  python src/verify_outputs.py            (about 10 minutes)
"""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ok = True
    sums = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split(maxsplit=1)
        sums[name.strip()] = h
    for name, h in sums.items():
        if name.startswith("data/") and sha256(ROOT / name) != h:
            print("INPUT CHANGED:", name); ok = False
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / "src/moss_screening.py"),
                        "--data", str(ROOT / "data"), "--out", tmp], check=True,
                       stdout=subprocess.DEVNULL)
        for ref in sorted((ROOT / "results").glob("*.csv")):
            new = Path(tmp) / ref.name
            if not new.exists():
                print("MISSING:", ref.name); ok = False; continue
            a, b = pd.read_csv(ref), pd.read_csv(new)
            num = a.select_dtypes("number").columns
            same = (list(a.columns) == list(b.columns) and a.shape == b.shape
                    and np.allclose(a[num].fillna(-9).values, b[num].fillna(-9).values, atol=1e-9)
                    and (a.drop(columns=num).astype(str).values == b.drop(columns=num).astype(str).values).all())
            print(("OK      " if same else "DIFFERS ") + ref.name)
            ok &= bool(same)
    print("ALL OUTPUTS REPRODUCED" if ok else "REPRODUCTION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Compare two generated datasets to qualify a MATLAB release.

Purpose
-------
The A00 release gate is a VALIDATED ALLOWLIST: a campaign generated entirely on
one validated release is homogeneous, whichever release that is. To add a
release, its output must be shown equivalent to an already-validated one. This
script is step (2) of that qualification (step (1) is the MATLAB smoke suite,
run on the candidate release).

Usage
-----
On each machine, generate the SAME micro dataset::

    # on an unvalidated release, set the qualification flag first:
    #   Windows :  set A00_RELEASE_QUALIFICATION=1
    #   bash    :  export A00_RELEASE_QUALIFICATION=1
    python make_micro_smoke.py

then copy one folder next to the other and run::

    python compare_generation_releases.py <dirA> <dirB>

Verdicts
--------
BIT-IDENTICAL          every state file hashes equal -> release qualifies.
NUMERICALLY-EQUIVALENT labels identical; signals differ only within tolerance
                       (default 1e-10 relative). Qualifies ONLY if the verdict
                       is explicitly accepted and recorded in
                       docs/framework_rationale.md.
MATERIALLY-DIFFERENT   labels differ, or signals exceed tolerance -> the two
                       releases must NOT be mixed in one campaign.

Nothing is written; both datasets are opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys

import numpy as np
from scipy.io import loadmat

_STATE = re.compile(r"^\d{4}\.mat$")
# Signal fields written per passage by D01 (RAW/Option-B format).
_SIGNAL_FIELDS = ("AcelVert", "AcelVertVag", "PitchPrimVag",
                  "acel_under", "DimAcel", "DimSpace")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_files(d: Path) -> list[str]:
    return sorted(f.name for f in d.iterdir() if _STATE.match(f.name))


def manifest(d: Path) -> dict:
    p = d / "case_info.mat"
    if not p.exists():
        return {}
    ci = loadmat(p, simplify_cells=True).get("case_info", {})
    return ci if isinstance(ci, dict) else {}


def scalar(v):
    a = np.ravel(np.asarray(v, dtype=object))
    return a[0] if a.size else None


def max_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a-b| / max(1, |a|, |b|) over finite entries; inf on shape mismatch."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    denom = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return float(np.max(np.abs(a - b) / denom))


def compare_labels(A: Path, B: Path) -> tuple[bool, list[str]]:
    """Damage-state labels must be EXACTLY equal: they define the dataset."""
    notes: list[str] = []
    pa, pb = A / "damage_states.mat", B / "damage_states.mat"
    if not (pa.exists() and pb.exists()):
        return False, ["damage_states.mat missing on one side"]
    ma = loadmat(pa, simplify_cells=True)
    mb = loadmat(pb, simplify_cells=True)
    ok = True
    for key in sorted(set(ma) | set(mb)):
        if key.startswith("__"):
            continue
        if key not in ma or key not in mb:
            ok = False
            notes.append(f"{key}: present on only one side")
            continue
        va, vb = np.asarray(ma[key], dtype=object), np.asarray(mb[key], dtype=object)
        if va.shape != vb.shape:
            ok = False
            notes.append(f"{key}: shape {va.shape} vs {vb.shape}")
        elif not np.array_equal(va, vb):
            ok = False
            notes.append(f"{key}: values differ")
    return ok, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a", type=Path)
    ap.add_argument("dir_b", type=Path)
    ap.add_argument("--tol", type=float, default=1e-10,
                    help="max relative signal difference for NUMERICALLY-EQUIVALENT")
    args = ap.parse_args()
    A, B = args.dir_a, args.dir_b
    for d in (A, B):
        if not d.is_dir():
            print(f"not a directory: {d}")
            return 2

    print("GENERATION-RELEASE COMPARISON")
    print(f"  A: {A}")
    print(f"  B: {B}\n")

    mA, mB = manifest(A), manifest(B)
    relA = scalar(mA.get("matlab_release", "?"))
    relB = scalar(mB.get("matlab_release", "?"))
    print(f"  MATLAB release      A={relA}  B={relB}")
    for key in ("gen_schema", "generation_behavior_version", "gen_fingerprint"):
        va, vb = scalar(mA.get(key, "?")), scalar(mB.get(key, "?"))
        flag = "OK " if va == vb else "*** DIFFERS ***"
        print(f"  {flag} {key}: {va} | {vb}")
    if scalar(mA.get("gen_fingerprint")) != scalar(mB.get("gen_fingerprint")):
        print("\n  Fingerprints differ -> the two runs used different CONFIG, not just\n"
              "  a different release. Re-generate with identical settings first.")
        return 2

    fa, fb = state_files(A), state_files(B)
    if fa != fb:
        print(f"\n  MATERIALLY-DIFFERENT: state file sets differ "
              f"({len(fa)} vs {len(fb)} files)")
        return 1

    labels_ok, notes = compare_labels(A, B)
    print(f"\n  damage-state labels identical: {labels_ok}")
    for n in notes:
        print(f"      {n}")

    identical, differing = [], []
    for name in fa:
        if sha256(A / name) == sha256(B / name):
            identical.append(name)
        else:
            differing.append(name)
    print(f"  byte-identical state files: {len(identical)}/{len(fa)}")

    worst = 0.0
    worst_where = ""
    if differing:
        print(f"\n  {len(differing)} file(s) differ byte-wise; comparing numerically")
        for name in differing:
            da = loadmat(A / name, simplify_cells=True)
            db = loadmat(B / name, simplify_cells=True)
            for field in _SIGNAL_FIELDS:
                if field not in da or field not in db:
                    continue
                va, vb = da[field], db[field]
                cells = isinstance(va, (list, np.ndarray)) and getattr(va, "dtype", None) == object
                pairs = zip(np.ravel(va), np.ravel(vb)) if cells else [(va, vb)]
                for x, y in pairs:
                    d = max_rel_diff(np.asarray(x, dtype=float),
                                     np.asarray(y, dtype=float))
                    if d > worst:
                        worst, worst_where = d, f"{name}:{field}"
        print(f"  worst relative signal difference: {worst:.3e}  ({worst_where})")

    print()
    if not labels_ok:
        print("VERDICT: MATERIALLY-DIFFERENT (labels differ) — do NOT mix these releases.")
        return 1
    if not differing:
        print("VERDICT: BIT-IDENTICAL — the candidate release qualifies.")
        return 0
    if worst <= args.tol:
        print(f"VERDICT: NUMERICALLY-EQUIVALENT (worst {worst:.3e} <= tol {args.tol:.1e})")
        print("         Qualifies ONLY if this verdict is explicitly accepted and")
        print("         recorded in docs/framework_rationale.md.")
        return 0
    print(f"VERDICT: MATERIALLY-DIFFERENT (worst {worst:.3e} > tol {args.tol:.1e})")
    print("         Generate the whole campaign on ONE of these releases.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

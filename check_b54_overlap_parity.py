"""Executable MATLAB/Python parity test for overlapping ballast patches.

Run:  python check_b54_overlap_parity.py
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np
from scipy.io import loadmat

from TTBI_2D.b54_model_matrices import _track_vectors


fails = 0


def check(name: str, cond: bool) -> None:
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails += 1


def fixture(patches) -> tuple:
    sleeper = NS(Tnum=9, num_app=2, num_onbeam=5, num_aft=2, spacing=1.0)
    track = NS(
        Sleeper=sleeper,
        Pad=NS(Prop=NS(k=11.0, c=12.0)),
        Ballast=NS(Prop=NS(k=21.0, c=22.0)),
        BallastOnBeam=NS(Prop=NS(k=31.0, c=32.0)),
    )
    model = NS(Mesh=NS(XLoc=NS(sleepers=np.arange(9, dtype=float))))
    calc = NS(Cte=NS(tol=1e-9))
    damage = NS(track={
        "x_bridge_local": 10.0,
        "ballast_patches": patches,
    })
    return track, model, calc, damage


print("B54 GOVERNING-PATCH PARITY CHECKS")

patches = np.array([
    [9.0, 13.0, 0.8, 2.0],
    [11.0, 15.0, 1.8, 0.6],
])
expected_k = np.array([1, .8, .8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.])
expected_c = np.array([1, 2., 2., .6, .6, .6, .6, .6, 1.])

V = _track_vectors(*fixture(patches))
check("Python stiffness follows largest |log eta_k|",
      np.array_equal(V.mult_bal_k, expected_k))
check("Python governing patch supplies paired damping",
      np.array_equal(V.mult_bal_c, expected_c))
V_rev = _track_vectors(*fixture(patches[::-1]))
check("Python result is row-order invariant",
      np.array_equal(V_rev.mult_bal_k, V.mult_bal_k)
      and np.array_equal(V_rev.mult_bal_c, V.mult_bal_c))

# Invoke the actual MATLAB production helper and compare its saved numerical
# vector to Python's result. No repository data/results/bundles are touched.
matlab = shutil.which("matlab")
matlab_ran = matlab is not None
if matlab is None:
    # Audit r5: a silent skip used to still print "ALL PASS", so a dispatch-PC
    # preflight log could claim cross-language parity that never ran. The
    # summary line below now states the degraded scope explicitly.
    print("  [SKIP] MATLAB executable not on PATH — the MATLAB half did NOT run")
else:
    repo = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="b54-parity-") as td:
        out_path = Path(td, "matlab_vectors.mat")
        env = os.environ.copy()
        env["B54_PARITY_OUT"] = str(out_path)
        proc = subprocess.run(
            [matlab, "-batch", "smoke_b54_overlap_parity"],
            cwd=repo / "scour_MATLAB",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        if proc.returncode != 0:
            print(proc.stdout)
        check("MATLAB overlap smoke exits successfully", proc.returncode == 0)
        check("MATLAB parity fixture emitted", out_path.exists())
        if out_path.exists():
            m = loadmat(out_path)
            mk = np.ravel(m["k_mult"])
            mc = np.ravel(m["c_mult"])
            check("MATLAB stiffness vector equals Python",
                  np.array_equal(mk, V.mult_bal_k))
            check("MATLAB damping vector equals Python",
                  np.array_equal(mc, V.mult_bal_c))

print()
if fails:
    print(f"B54 OVERLAP PARITY: {fails} CHECK(S) FAILED")
elif matlab_ran:
    print("B54 OVERLAP PARITY: ALL PASS")
else:
    print("B54 OVERLAP PARITY: PYTHON-ONLY PASS "
          "(MATLAB cross-language half SKIPPED — not verified on this machine)")
sys.exit(1 if fails else 0)

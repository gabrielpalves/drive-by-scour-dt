"""
check_raw_parity.py — Python half of the Option-B parity smoke.

Verifies that core/dataset._raw_to_space_crop reproduces MATLAB's interp1+crop
EXACTLY on real generated data. Run AFTER scour_MATLAB/smoke_raw_parity.m has
written matlab_ref_parity.mat into the dataset folder.

    python check_raw_parity.py "scour_MATLAB/Results/<case_name>"

PASS = max|MATLAB - Python| == 0 (or < 1e-12 float noise).
"""
import sys, os
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.dataset import _raw_to_space_crop

if len(sys.argv) < 2:
    sys.exit(__doc__)
folder = sys.argv[1]

ref_f = os.path.join(folder, "matlab_ref_parity.mat")
if not os.path.exists(ref_f):
    sys.exit(f"No {ref_f} — run smoke_raw_parity.m in MATLAB first.")

data = sio.loadmat(os.path.join(folder, "0001.mat"))["data"][0, 0]
R = sio.loadmat(ref_f)
ref = R["ref"][0, 0]
npass = int(np.ravel(R["npass"])[0])

if "DimSpace" not in (data.dtype.names or ()):
    sys.exit("0001.mat is LEGACY (pre-interpolated) — nothing to check.")

groups = ["AcelPrimVag", "PitchPrimVag", "AcelRodaPrimVag"]
worst = 0.0
for p in range(npass):
    dim_acel = data["DimAcel"][0, p]
    dim_space = data["DimSpace"][0, p]
    cs = data["crop_start"][0, p]
    ce = data["crop_end"][0, p]
    for g in groups:
        raw = data[g][0, p]                      # (rows, DimAcel)
        m_ref = ref[g][0, p]                     # (rows, crop_len) from MATLAB
        py = np.vstack([_raw_to_space_crop(raw[r, :], dim_acel, dim_space, cs, ce)
                        for r in range(raw.shape[0])])
        if py.shape != m_ref.shape:
            sys.exit(f"SHAPE MISMATCH {g} p{p}: python {py.shape} vs matlab {m_ref.shape}")
        d = float(np.max(np.abs(py - m_ref)))
        if not np.isfinite(d):
            sys.exit(
                f"NON-FINITE PARITY DIFFERENCE {g} p{p} — investigate "
                "before generating"
            )
        worst = max(worst, d)
        print(f"  passage {p+1:2d}  {g:16s} shape {py.shape}  max|diff| = {d:.3e}")

print(f"\nWORST max|MATLAB - Python| = {worst:.3e}")
if worst < 1e-12:
    print("PARITY PASS")
else:
    print("PARITY FAIL — investigate before generating")
    sys.exit(1)

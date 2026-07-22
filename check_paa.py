"""
check_paa.py — unit checks for the TRUE Keogh PAA (audit r3, 2026-07-22).

The pre-fix `_apply_paa` was a linear-interpolation resampler (point
subsampling, no averaging). These checks pin the corrected semantics:
equal-width window MEANS, exact fractional windows for non-divisible
lengths, global-mean preservation, and the audit's regression example.

Run:  python check_paa.py     (exit 0 = ALL PASS)
"""
from __future__ import annotations

import sys

import numpy as np

from core.preprocessing import TTBIPreprocessor

fails = 0


def check(name: str, cond: bool) -> None:
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails += 1


def paa(x, n):
    """Run the class transform on a 1-D signal, return 1-D output."""
    p = TTBIPreprocessor(method='paa', n_segments=n)
    return p._apply_paa(np.asarray(x, dtype=np.float32)[None, None, :])[0, 0]


def paa_reference(x, n):
    """Brute-force reference: per-sample overlap weights with each window.

    Sample k occupies [k, k+1); window j is [j*L/n, (j+1)*L/n). The window
    mean weights each sample by its overlap length / window width.
    """
    x = np.asarray(x, dtype=np.float64)
    L = len(x)
    w = L / n
    out = np.zeros(n)
    for j in range(n):
        lo, hi = j * w, (j + 1) * w
        acc = 0.0
        for k in range(L):
            ov = max(0.0, min(hi, k + 1) - max(lo, k))
            acc += x[k] * ov
        out[j] = acc / w
    return out


print("TRUE-PAA CHECKS")

# 1. The audit's regression example: subsampling gave [0,4]; PAA gives [0,2].
check("audit example [0,0,0,4] -> [0,2]",
      np.allclose(paa([0, 0, 0, 4], 2), [0.0, 2.0]))

# 2. Constant signal -> constant, divisible and non-divisible lengths.
for L, n in [(512, 512), (5856, 512), (10, 3), (7, 5)]:
    check(f"constant preserved (L={L}, n={n})",
          np.allclose(paa(np.full(L, 3.25), n), 3.25, atol=1e-6))

# 3. Divisible ramp: window means are the means of consecutive blocks.
x = np.arange(12, dtype=float)
check("divisible ramp = block means",
      np.allclose(paa(x, 4), [1.0, 4.0, 7.0, 10.0], atol=1e-6))

# 4. Impulse: total mass preserved (sum(out)*w == sum(x)); support spans at
#    most the two windows the fractional boundary can split it across.
L, n = 10, 3
x = np.zeros(L); x[3] = 1.0        # sample [3,4) straddles the 10/3 edge
y = paa(x, n)
check("impulse mass preserved", np.isclose(y.sum() * (L / n), 1.0, atol=1e-6))
check("impulse support <= 2 windows", int(np.count_nonzero(y)) <= 2)

# 5. Non-divisible cases match the brute-force overlap reference exactly.
rng = np.random.default_rng(42)
for L, n in [(10, 3), (11, 4), (5856, 512), (101, 7)]:
    x = rng.standard_normal(L)
    check(f"matches overlap reference (L={L}, n={n})",
          np.allclose(paa(x, n), paa_reference(x, n), atol=1e-5))

# 6. Global mean preserved exactly (integral property), random signals.
for L, n in [(5856, 512), (997, 128)]:
    x = rng.standard_normal(L)
    check(f"global mean preserved (L={L}, n={n})",
          np.isclose(paa(x, n).mean(), x.mean(), atol=1e-6))

# 7. Batch shape/dtype contract + L == n passthrough.
X = rng.standard_normal((5, 2, 100)).astype(np.float32)
Y = TTBIPreprocessor(method='paa', n_segments=25)._apply_paa(X)
check("batch shape (5,2,25)", Y.shape == (5, 2, 25))
check("float32 dtype", Y.dtype == np.float32)
Z = TTBIPreprocessor(method='paa', n_segments=100)._apply_paa(X)
check("L==n passthrough unchanged", np.allclose(Z, X))

# 8. Chunk-boundary consistency: >512 samples goes through 2 chunks; the
#    result must equal a per-row transform.
Xb = rng.standard_normal((700, 1, 40)).astype(np.float32)
Yb = TTBIPreprocessor(method='paa', n_segments=8)._apply_paa(Xb)
rowwise = np.stack([paa(Xb[i, 0], 8) for i in (0, 511, 512, 699)])
check("chunked == row-wise at chunk boundary",
      np.allclose(Yb[[0, 511, 512, 699], 0, :], rowwise, atol=1e-6))

print()
print("PAA: ALL PASS" if fails == 0 else f"PAA: {fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)

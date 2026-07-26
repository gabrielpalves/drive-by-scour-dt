"""Subset/order-invariance checks for load-time sensor noise.

Run:  python check_sensor_noise_pairing.py
"""
from __future__ import annotations

import sys

import numpy as np

from core.dataset import (_inject_sensor_noise, NOISE_RNG_SEED,
                          PREPROC_PROTOCOL)


fails = 0


def check(name: str, cond: bool) -> None:
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails += 1


print("SENSOR-NOISE PAIRING CHECKS")

rng = np.random.default_rng(20260723)
# Strictly nonzero, DOF-distinct signals make an accidental no-op or channel
# swap visible. Shape/order matches the full cache loader.
raw = (rng.normal(size=(7, 8, 31)).astype(np.float32)
       + np.arange(1, 9, dtype=np.float32)[None, :, None])
raw_before = raw.copy()
all_dofs = list(range(8))
reversed_dofs = list(reversed(all_dofs))

modes = {
    "all_mult": set(all_dofs),
    "legacy_wheel": {3, 4},
    "sprung_mult": set(all_dofs) - {3, 4},
}
outputs = {}
check("PCG64 algorithm is protocol-pinned",
      PREPROC_PROTOCOL.get("noise_bit_generator") == "numpy.random.PCG64")

for mode, active in modes.items():
    sn = {"mode": mode, "desvio": 0.05}
    full = _inject_sensor_noise(raw, all_dofs, sn)
    rev = _inject_sensor_noise(raw[:, reversed_dofs, :], reversed_dofs, sn)
    outputs[mode] = full
    check(f"{mode}: float32 output", full.dtype == np.float32)
    check(f"{mode}: caller input unchanged", np.array_equal(raw, raw_before))
    check(f"{mode}: deterministic repeat",
          np.array_equal(full, _inject_sensor_noise(raw, all_dofs, sn)))

    for d in all_dofs:
        single = _inject_sensor_noise(raw[:, [d], :], [d], sn)[:, 0, :]
        pair_dofs = [d, (d + 3) % 8]
        if pair_dofs[0] == pair_dofs[1]:
            pair_dofs[1] = (d + 1) % 8
        pair = _inject_sensor_noise(raw[:, pair_dofs, :], pair_dofs, sn)[:, 0, :]
        rev_local = reversed_dofs.index(d)
        check(f"{mode}: DOF {d} identical alone/pair/full/reversed",
              np.array_equal(single, pair)
              and np.array_equal(single, full[:, d, :])
              and np.array_equal(single, rev[:, rev_local, :]))
        changed = not np.array_equal(single, raw[:, d, :])
        check(f"{mode}: DOF {d} active-mask semantics",
              changed == (d in active))

# A global DOF uses the same keyed random field across modes whenever both
# modes activate it. Only the mask changes, never the realization.
check("wheel DOF 3 draw shared by legacy_wheel and all_mult",
      np.array_equal(outputs["legacy_wheel"][:, 3, :],
                     outputs["all_mult"][:, 3, :]))
check("sprung DOF 0 draw shared by sprung_mult and all_mult",
      np.array_equal(outputs["sprung_mult"][:, 0, :],
                     outputs["all_mult"][:, 0, :]))

manual_rng = np.random.Generator(np.random.PCG64([NOISE_RNG_SEED, 0]))
manual0 = (raw[:, 0, :] + 0.05 * raw[:, 0, :]
           * manual_rng.standard_normal(raw[:, 0, :].shape).astype(np.float32))
check("DOF draw is exactly the protocol-pinned PCG64 stream",
      np.array_equal(outputs["all_mult"][:, 0, :], manual0))

try:
    _inject_sensor_noise(raw[:, [0], :], [0], {"mode": "unknown"})
except ValueError:
    check("unknown mode rejected", True)
else:
    check("unknown mode rejected", False)

print()
print("SENSOR NOISE: ALL PASS" if fails == 0
      else f"SENSOR NOISE: {fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)

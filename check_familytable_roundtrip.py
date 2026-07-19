"""check_familytable_roundtrip.py - Python side of the Feature-A contract smoke.

Run AFTER `smoke_familytable` in MATLAB (which writes
scour_MATLAB/Results/_smoke_familytable/damage_states.mat with REAL MATLAB
`save`). This proves core.dataset.read_state_table parses the GENUINE MATLAB
cellstr/logical encoding — the other check scripts build their fixtures with
scipy.io.savemat, whose cell encoding differs from MATLAB's, so without this
smoke the real A00 output format was never actually exercised.

Prints ALL PASS / a FAIL and exits nonzero on failure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.dataset import read_state_table, _stratum_keys  # noqa: E402

SMOKE_DIR = os.path.join('scour_MATLAB', 'Results', '_smoke_familytable')
EXPECTED_FAMILY = (['target_healthy'] * 2 + ['scour_only'] * 2
                   + ['bearing_only'] + ['nuisance_only'] + ['joint'] * 2)
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        fails += 1


if not os.path.exists(os.path.join(SMOKE_DIR, 'damage_states.mat')):
    print(f"  [SKIP] {SMOKE_DIR} not found — run smoke_familytable in MATLAB first.")
    sys.exit(0)

table = read_state_table(SMOKE_DIR)
check("REAL MATLAB cellstr family parses", table['family'] == EXPECTED_FAMILY,
      str(table['family']))
check("AnchorTarget parses as ints",
      table['anchor_target'].tolist() == [0, 0, 2, 3, 1, 0, 0, 0])
check("AnchorLevel parses as ints",
      table['anchor_level'].tolist() == [0, 0, 1, 2, 1, 0, 0, 0])
check("CrackOn parses as bools (real MATLAB logical)",
      table['crack_on'].tolist() == [False, False, False, False, False, True, True, False])
check("damage matrices parse with values intact",
      abs(table['damage_states'][2, 1] - 0.15) < 1e-12
      and abs(table['bearing_fixity'][4, 0] - 0.2375) < 1e-12)
# Stratum keys must be constructible from the real encoding too.
keys = _stratum_keys(table, 0.60)
check("stratum keys build from the real table",
      keys[0] == 'target_healthy' and keys[2] == 'scour_only|target2'
      and keys[4] == 'bearing_only|target1' and keys[5] == 'nuisance_only'
      and keys[6].startswith('joint|crack'), str(keys))

print()
print("FAMILY-TABLE ROUNDTRIP: ALL PASS" if fails == 0 else
      f"FAMILY-TABLE ROUNDTRIP: {fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)

# Audit r5 handoff — independent verification of the R4 commits

> **HISTORICAL HANDOFF — SUPERSEDED BY R11.** This preserves the R5 review
> record. Open items described below have since been redesigned under R11; this
> file does not authorize generation, ablation, bundle dispatch or paper claims.

Audience: the external auditor (Codex), round 5.
Baseline audited: `f805fbe`, `a91f2f2`, `c0ead72` (i.e. `6f61fa0..c0ead72`,
+5870/−523 across 36 files), plus `docs/audit_r4_results.md`.

## 1. Verdict

**R4 is accepted, with one substantive defect found and fixed, one minor
weakness fixed, and one undocumented breaking change to write up.**

The implementation quality is high and the self-report is accurate: every
claim in `audit_r4_results.md` that I tested held up. Two structural
properties I specifically tried to break and could not:

- **Cross-language B54 parity is real, not a shared-bug trap.** The natural
  failure mode of extracting `B54_TrackVectors.m` and testing it from
  `check_b54_overlap_parity.py` is that both sides share the same defect.
  They do not: the Python check carries an *independent hardcoded oracle*
  (`expected_k`/`expected_c`), and I confirmed by mutation that breaking the
  **MATLAB** helper alone (governing rule → last-patch-wins) turns the check
  RED via the row-order-invariance assertion, with the Python mirror intact.
- **The test-once firewall is enforced at runtime, not just by convention.**
  Finalist CV receives `development_idx` only, and
  `comprehensive_ablation_multidamage.py:1647` raises if
  `development_idx ∩ outer_idx` is non-empty.

## 2. How this round was verified

Green suites were treated as *unproven* until each guard was shown to fail.
I ran all 12 Python checks (all PASS) and the MATLAB battery, then ran a
**mutation harness**: for each audit-r3/r4 fix, re-inject the original defect
byte-for-byte, run the guard that should catch it, and require a non-zero
exit; files restored from original bytes in a `finally`, with a post-condition
assert that the tree is byte-identical.

| Re-injected defect | Guard | Result |
|---|---|---|
| PAA → point subsampling | `check_paa.py` | CAUGHT |
| Noise RNG keyed by subset index, not global DOF | `check_sensor_noise_pairing.py` | CAUGHT |
| Multi-head loss weights → uniform | `check_weighted_head_mse.py` | CAUGHT |
| **Objective → all-head aggregate** | *(none)* | **MISSED** |
| B54 Python mirror → multiplicative stacking | `check_b54_overlap_parity.py` | CAUGHT |
| B54 **MATLAB** helper → last-patch-wins | `check_b54_overlap_parity.py` | CAUGHT |
| Hierarchical bootstrap → no state resampling | `check_statistical_inference.py` | CAUGHT |

## 3. Findings

### F1 — BLIND SPOT (substantive): the pre-registered primary estimand was unguarded. FIXED.

Reverting `core/task.py::objective_value` to `return metrics["mse"]` — i.e.
letting model selection see the bearing heads again, silently undoing the r3
pre-registration — **passed the entire suite**, all 12 checks including
`check_protocol_hash.py`. I verified this exhaustively (mutation applied once,
whole battery run: every check green).

Why it slipped through both of us: `TRAIN_PROTOCOL["objective"]` *describes*
the estimand in prose, but nothing bound the description to the
implementation. Contrast the two mechanisms that do work — `batch_size` and
`patience` are **read from** `TRAIN_PROTOCOL` by the trainer (single source of
truth, cannot drift), and the loss is asserted **behaviourally** by
`check_weighted_head_mse.py`. The objective had neither. This is precisely the
declared-vs-actual gap the R7.1 "drive the sampler FROM the protocol" design
closed for the search space, never extended to the objective.

Fix: five behavioural assertions appended to `check_campaign_controls.py` —
scour-primary when bearing heads exist; the aggregate is *not* returned on
bearing rungs; correct aggregate fallback without bearing heads; a scour
improvement outranks a bearing improvement; and the `TRAIN_PROTOCOL` prose
still declares the scour-primary estimand (so re-registering it must move the
protocol hash). Mutation harness re-run: **6/6 caught, 0 missed.**

### F2 — MINOR: parity check could silently degrade to half its claim. FIXED.

`check_b54_overlap_parity.py` used `shutil.which("matlab")` and, when absent,
printed `[SKIP]` and then **`ALL PASS`**. On a dispatch PC whose MATLAB is not
on PATH, the preflight log would assert cross-language parity that never ran.
Fixed: the summary now prints `PYTHON-ONLY PASS (MATLAB cross-language half
SKIPPED — not verified on this machine)`. Verified both branches by running
with PATH stripped. Exit semantics unchanged.

### F3 — MINOR (documentation): an undocumented breaking change in the DT layer.

`digital_twin/assets.py` now routes `DigitalAsset.__init__` through
`verify_standalone_dt_package`, which **raises** if metadata lacks
`champion_weights_sha256`, `scaler_sha256`, `scaler_filename`,
`protocol_hash`, `protocol_descriptor`. Every pre-R4 DT package therefore
becomes unloadable.

I judge the change **correct and in-mandate** (it is the deployment-provenance
item), and it is harmless in practice: pre-mass-fix packages are invalid
anyway, and the prototype-twin figures use a synthetic observation model, not
`DigitalAsset`. But `audit_r4_results.md` does not record it as breaking, and
the ground rules asked for out-of-scope effects to be written up. Please add
it to the R4 record. Note also the DT layer has never had the campaign's audit
treatment — a DT hardening pass remains queued, and this change touches it.

Confirmed non-issues while checking F3: `digital_twin` imports cleanly (the
`b43_model_geometry` ImportError is the pre-existing TTBI_2D bare-import
convention, reproduced identically on the pre-R4 tree, not a regression).

## 4. Agreements with the R4 record

- The `bundle_*.zip` files present in the snapshot are stale — **do not
  dispatch them**. Rebuild from the approved commit only.
- `nuisance_only` stays exploratory, no CI, no binary-detection claim; the
  design floor of 50 states for ~10 outer-test states is the right target for
  the next regeneration.
- No sensitivity/specificity/POD/minimum-detectable-damage claims without a
  development-locked threshold.
- `docs/paper1_methodology.md` is correctly banner-marked NON-AUTHORITATIVE.
- The publication-safe claim as written in §"Publication-safe claim" is one I
  can defend; keep the s16/s23 "exploratory deployment selection" label.

## 5. Open for r5

1. **Audit F1's fix** — five new checks in `check_campaign_controls.py`. In
   particular: is a behavioural assertion the right closure, or should
   `objective_value` be *derived from* `TRAIN_PROTOCOL` (the batch_size/patience
   pattern) so the estimand cannot diverge from its declaration by construction?
   I lean toward the latter as the stronger fix; I chose the behavioural guard
   because it does not move the protocol hash. Your call — argue it.
2. **Apply the same mutation discipline to your own R4 modules.** I mutated the
   *fixes*; the new machinery deserves it too. Highest-value targets:
   `repeated_stratified_group_folds` (fold exclusivity, stratum balance),
   `hierarchical_state_seed_bootstrap` (seed-median placement inside the
   replicate), the paired-contrast alignment, and the artifact/environment
   verifiers. If a guard cannot be made to fail, it is not a guard.
3. **Reconcile the duplicated fingerprint mechanism.** A00 now carries *both*
   my unconditional `gen_rule_ver` and your conditional `track_eov_impl`. Both
   exist because each of us tried to preserve valid data. **The user has since
   decided to discard all existing data and regenerate every rung from
   scratch**, so that constraint is gone. My proposal: bump `gen_schema` to r9
   (the standing rule — "anything pre-r9 is orphaned by construction"), keep
   exactly ONE behaviour-version key for future rule-only changes, and delete
   the other. State which you prefer and why.
4. **Benchmark before dispatch** (your gate 4) — one representative Optuna
   study and one finalist-CV refit, so the 1,344–1,359-study estimate has a
   wall-clock attached. This now gates a 3-PC, all-ten-rung campaign.

## 6. Ground rules (unchanged)

Feature-byte changes bump `CACHE_SCHEMA_TAG`; behaviour changes get a named
protocol entry; every new mechanism ships a `check_*.py`; MATLAB stays
`checkcode`-clean and parfor-safe; do not touch `data/`, `results/`, or the
bundle zips; disagreements with the decisions of record in
`docs/audit_r4_handoff.md` §2 are findings, not edits.

### Green-state reference at this handoff

12 Python checks ALL PASS; MATLAB `smoke_audit`, `smoke_b54_overlap_parity`,
`smoke_contact_closure`, `smoke_stage3`, `smoke_geometry` ALL PASS;
mutation harness 6/6 caught, 0 missed; working tree byte-identical after every
mutation run.

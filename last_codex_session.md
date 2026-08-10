Verdict: Claude made a good mathematical start and documented the new campaign thoughtfully, but the repository is not runnable under the new design. No data generation or PC dispatch should begin yet.

## What is correct

The wheelset physics is correct:

\[
\ddot z_w=u_{tt}+2vu_{xt}+v^2u_{xx}+\ddot h_w .
\]

The implementation in [D01_DataProcessing.m](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/scour_MATLAB/D01_DataProcessing.m:64) has the right terms, and the added `-m·hdd_path` contribution in [B66_ContactForce.m](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/scour_MATLAB/B66_ContactForce.m:67) has the correct sign relative to the coupled solver in [B65_DynamicCalcCoupledFaster.m](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/scour_MATLAB/B65_DynamicCalcCoupledFaster.m:208).

Claude also correctly retained `AcelRodaPrimVag` as the legacy virtual rail-field diagnostic and added a separate physical wheelset field.

## Immediate blockers

1. Fresh MATLAB generation currently always aborts.

[execute_generation_state.m](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/scour_MATLAB/+ttbi/execute_generation_state.m:184) adds `AcelWheelsetPrimVag` and `channel_schema_id`, then calls the exact payload validator. But [state_payload_fields.m](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/scour_MATLAB/+ttbi/state_payload_fields.m:7) lacks both fields, so [validate_resumed_state_payload.m](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/scour_MATLAB/+ttbi/validate_resumed_state_payload.m:17) rejects every newly computed state before saving.

The Python side also rejects the new fields and still maps channels 3–4 to `AcelRodaPrimVag`: [generation_state_contract.py](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/core/generation_state_contract.py:35) and [dataset.py](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/core/dataset.py:390).

The schema decision should be:

- `case_info`/manifest is authoritative.
- The loader resolves the schema once from that manifest.
- Every state file must contain exactly the same schema ID.
- Mixed schemas are rejected.
- Do not infer dataset truth from the first state file or branch independently per file.

2. Contract versions were not bumped.

The code still declares R11 and `generation-rules-v7` in [build_generation_identity.m](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/scour_MATLAB/+ttbi/build_generation_identity.m:9).

This work requires both:

- `gen_schema` bump: the saved payload contract changed.
- `generation_behavior_version` bump: contact-force reconstruction and generated responses changed.

The channel schema must also enter the generation fingerprint, case manifest, cache provenance, protocol hash, result manifests, and plotting metadata.

3. The wheelset mask is not exactly solver-consistent.

The coupled solver activates the profile-inertia term only where `elexj > 0`. D01 and B66 currently use only `x_path >= 0`, leaving post-exit `hdd_path` active. That particularly matters when wheel-defect terms continue outside the rail domain.

Use the active-element mask, preferably through one shared helper used by both D01 and B66. That would also remove the duplicated chain-rule expression.

4. “Axle-box quantity” is too strong.

The defensible name is:

> Idealized model-predicted constrained-wheelset vertical acceleration, used as an axle-box response proxy.

The model omits actual mounting dynamics, contact compliance, sensor bandwidth/filtering and much of the real unsprung assembly.

5. The tests do not qualify the new implementation.

I ran two relevant checks:

- `check_channel_semantics.py` fails immediately because it still pins an old comment phrase and does not check `hdd_path` or `physical8_v1`.
- `check_generation_contract.py` reports all 186 mutations caught despite fresh generation being guaranteed to fail its own validator. That exposes a real integration-test blind spot.

Required new tests include a manufactured four-term acceleration case, a nonzero-`hdd_path` B66 force-balance case, save/resume/load integration, wrong/missing/mixed schema rejection, and a mutation that switches channels 3–4 back to the legacy rail field.

The Python TTB-2D mirror also remains stale: [b66_contact_force.py](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/TTBI_2D/b66_contact_force.py:47) still omits `hdd_path`.

## The campaign protocol is documented, not implemented

The plan in [paper1_campaign_plan.md](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/docs/paper1_campaign_plan.md:204) largely captures our discussion, but the executable policy remains:

- four PAA-only arms;
- three HPO seeds;
- full-eight-channel HPO;
- old L60/L99 anchor stages;
- 24 studies × 100 trials, rather than the new 100 studies.

See [hyperparameter_policy.py](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/core/hyperparameter_policy.py:51). It explicitly forces every non-anchor subset—including the selected pair—to a one-trial frozen run at line 1369. Therefore, the 2,000-trial final-pair HPO cannot currently run.

Also:

- The 16 RAW/PAA cells are absent.
- Multi-rate pooling still flattens sequence-length-dependent features in [models.py](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/core/models.py:65), confounding RAW versus PAA.
- Robustness remains 30 initialization seeds on one split in [robustness.py](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/training/robustness.py:39).
- Existing checkers actively pin the old protocol, so “all green” would currently qualify the wrong campaign.

There is also one methodological ambiguity we should correct from our own earlier discussion: the `5 winners × 3 grouped splits × 2 initialization seeds` matrix is development adjudication if it selects the frozen HPO vector. It cannot simultaneously be called unbiased post-selection stability. After selecting the vector, final stability needs a separate predeclared post-freeze refit set evaluated on the sealed outer test.

## Answers to Claude’s five planning questions

### 1. Fixed profile

A single fixed phase is not a classical statistical confound because it never varies with the label. It does, however, make the conclusions conditional on one excitation realization. Because signals are mapped to space, roughness features align at exactly the same locations in every passage, allowing a model to exploit profile-specific damage interactions—particularly for local EI-loss cases.

Fernandes reports one generated FRA Class-4 profile, so the choice is literature-comparable, but that paper does not establish phase robustness. :codex-file-citation{path="/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/papers/Fernandes_2025-multi-damage-classification-railway-bridges-drive-by.pdf" purpose="source"}

Recommendation: keep the fixed profile for the primary campaign, but predeclare a small frozen-model distribution-shift test using 3–5 alternative Class-4 phase seeds on balanced healthy/scour/bearing/crack states.

### 2. Model-form classifications

| Item | Recommendation |
|---|---|
| 6/15/30 m rail clearance | **(a)** Micro-convergence decides production before generation. |
| One-seat/two-rail scaling | **(b)** Keep inherited baseline, compare consistent 1×/2× alternatives before final freeze; do not silently double. |
| 0.545→0.600 m spacing | **(b)** Run a spacing-consistent recalculation sensitivity using Zhai’s equations. |
| Missing `Kw/Cw`, on-bridge condensation | **(b)** Freeze as an explicit inherited simplified topology. If a wheelset channel wins, make topology sensitivity mandatory before publication. |
| Rayleigh refinement | **(a)** Recalibrate per grid for production; fixed-M0 coefficients are the sensitivity. |
| Rail 0.1% target | **(b)** Retain as author-chosen and run a small damping sensitivity. |

Zhai explicitly states its properties per rail seat, uses 0.545 m spacing, reports +12% ballast-acceleration overprediction without shear, and concludes that adjacent-ballast shear is necessary for track-dynamic analysis. :codex-file-citation{path="/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/papers/Zhai_2004_modelling_experiment_railway_ballast_vibrations.pdf" purpose="source"}

The TTB-2D paper supports describing the implementation as a simplified 2D lumped track model, but it does not validate every scaling and bridge-condensation choice. :codex-file-citation{path="/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/papers/Cantero_2D_TTBI.pdf" purpose="source"}

Any sensitivity code needed for cross-model comparisons should be present before commit A; otherwise adding it later changes the source root.

### 3. State counts

`L99-S/M = 475` is consistent with three scour targets.

`F40-M = 450` is not. Under the existing five-family formula with one central scour target:

\[
50+25+50+50+250=425.
\]

F40-M should get its own one-pier matrix, not reuse the two-pier L60 matrix. Ideally it should also contain an explicitly matched subset of F40-S states/EOV seeds so the effect of adding bearing/crack mechanisms can be evaluated cleanly.

### 4. Transport gate

The proposed `4 pipelines × 3 blocks × 5 seeds = 60` fits is reasonable, but it tests hyperparameter transport if weights are retrained per block—not frozen-model transport.

“Re-HPO only on collapse” needs a numerical development-only trigger fixed beforehand. Any rescue HPO should give every retained pipeline the same budget and be reported separately from the frozen-hyperparameter result.

### 5. Version bump

Removing stages alone requires a behavior-version bump, not necessarily a payload-schema bump. But the combined replan changes stages, geometry/state designs, contact reconstruction, and saved channel fields. Therefore this release needs coordinated bumps of both `gen_schema` and `generation_behavior_version`, plus cache/study tags.

## Two final planning corrections

The branch strategy contradicts itself. [ISSUES_FOUND.md](/C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/paper1/ISSUES_FOUND.md:2871) says strip deferred mechanisms before commit A, while the campaign plan says keep them disabled on `main` and branch future work from A. I recommend the latter: no strip, four bridge-only stage configurations, and a future experimental branch from commit A.

Finally, the “10.8 GPU-days floor” is a baseline extrapolation, not a lower bound: it comes from one two-channel PAA-512 case on a laptop RTX 4070, while the target GPUs and RAW cells differ. Benchmark representative RAW/PAA cells on the desktops before scheduling. Also keep the full comparative 16-cell HPO on the matched 5060 Ti machines; assigning only selected cells to the RTX 2060 would correlate hardware with pipeline.

Overall: accept Claude’s handoff as a solid design/start, but mark it **NOT RUNNABLE / NOT DISPATCH-READY**. The next implementation priority is completing and testing `physical8_v1`; then resolve the HPO adjudication wording and F40-M state design before rewriting the campaign contracts.

---

## Current Codex handoff — 2026-08-10

The audit above is historical. Its implementation blockers have been closed.
The controlling current record is the final `Codex implementation closure` and
`Codex run-start closure` at the end of `paper1/ISSUES_FOUND.md`, together with
`docs/paper1_campaign_plan.md`.

The present candidate has four production blocks, `physical8_v1`, complete
training adapters, F25-R/F25-X production, final MATLAB model-form evidence,
and an executable external authorization chain. The last audit found and closed
two operational gaps: a fresh clean-A main capacity publisher now exists, and
the benchmark establishes registered deterministic state before binding that
receipt. F25 now executes two registered pair-envelope plus two conservative
full-eight RAW capacity cases, uses exact locked/source/runtime receipt
validation, and ships experiment-qualified paired-workspace evidence with
externally anchored ZIP verification.

Source identities for the amended candidate are:

- MATLAB generator: `aa187204cb3f89e24cb8bc894034044bad38b0358f5e0cd586338f84a8418efb`
  (344 files; unchanged, so final03 evidence remains valid).
- Python runtime: `3ca4040fa289901569e8e73b9eb875e53220b9a829b4a84839be2743952440f1`
  (150 files).

Do not start retained runs merely because local source qualification is green.
The main campaign still needs genuine target RTX 5060 capacity/benchmark
evidence, 12 qualification-pair receipts and inventory across three MATLAB
hosts, the 420-case contact authorization, and the external dispatch manifest
before report-only B and the six Paper-1 ZIPs. Build the two F25 ZIPs directly
from clean A, but do not start F25 jobs until their retained hashes/commit pass
`--verify-pair` and the genuine four-case preflight passes on the RTX 2060.

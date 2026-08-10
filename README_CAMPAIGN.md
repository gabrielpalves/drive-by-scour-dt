# Paper 1 production campaign

> **Status (2026-08-09): DISPATCH BLOCKED.**
>
> The four-block implementation and model-form decisions are complete. Production
> generation, training, bundle publication, and dispatch remain blocked until a
> clean commit A passes the external benchmark, host-qualification, contact, and
> authorization gates below. A report-only commit B is the final authorization
> step; prose outside the fixed report header cannot authorize a run.

This is the concise operator guide. The controlling scientific specification is
[`docs/paper1_campaign_plan.md`](docs/paper1_campaign_plan.md). The only dispatch
verdict is [`docs/audit_r5_results.md`](docs/audit_r5_results.md), whose filename
is retained for compatibility. The methodology and framework-rationale files are
historical design records, not execution instructions.

Here, "registered" means prospectively fixed in versioned, hash-identified
repository source. It does not mean externally preregistered.

## Production contract

| Block | Physics | States | Passages/state | Solves | Dataset |
|---|---|---:|---:|---:|---|
| `F40-S` | 40 m scour-only | 305 | 50 | 15,250 | `F40-S_L40_st305` |
| `F40-M` | 40 m scour + bearing fixity + local EI loss | 425 | 50 | 21,250 | `F40-M_L40_st425` |
| `L99-S` | 99.6 m scour-only | 475 | 50 | 23,750 | `L99-S_L99.6_st475` |
| `L99-M` | 99.6 m scour + bearing fixity + local EI loss | 475 | 50 | 23,750 | `L99-M_L99.6_st475` |
| **Total** | | | | **84,000** | |

`F40-S` is the dense 0--60% support-stiffness-loss grid: 61 severities
times five replicas. The other blocks use the fixed five-family state design.
All partitions are semantic-state grouped; passages from one state never cross
train, inner-validation, and sealed-test partitions.

Every production passage is explicitly bound to:

- response schema `physical8_v1`: three vehicle-body/bogie vertical
  accelerations, two total constrained-wheelset vertical-acceleration proxies,
  and three vehicle-body/bogie pitch rates;
- a separate `acc_under` virtual rail-field diagnostic, which is not substituted
  for either wheelset channel;
- `rail_end_clearance_m = 6` under decision
  `paper1-rail-domain-clearance-c06-v1`;
- one fixed FRA class-4 profile realization, operational speed/temperature/
  vehicle variability enabled, and track-damage and wheel-OOR mechanisms
  disabled by configuration;
- raw, un-interpolated, noise-free MATLAB histories. RAW/PAA selection,
  observation noise, scaling, and model input construction occur in Python.

The two multi-damage blocks model nominal bearing fixity and a local uniform
element-EI reduction. Scour remains support-stiffness loss, not scour-hole depth;
the EI-loss label is a numerical crack surrogate, not a field-calibrated crack
measurement.

## Training contract

The registered comparison is the 16-cell matrix

`{RAW, PAA} x {position off/on} x {LSTM off/on} x {multi-rate off/on}`.

The complete pre-outcome inventory contains at most 160 listed 100-trial HPO
studies (16,000 trial slots) and 1,740 listed refit jobs. Authenticated aliases
may reduce executed work but may not remove jobs from the inventory or duplicate
a canonical fit. `F40-S` selects the physical channel pair; the selected-pair
pipelines are then tuned independently within every block. Outer-test data stay
sealed until the corresponding block-freeze artifact is authenticated.

Hardware allocation is by registered seed parity, not by architecture,
representation, channel, or outcome. The two training hosts must therefore
remain matched within every scientific comparison block.

## Exact six-bundle publication set

| Bundle | Purpose |
|---|---|
| `bundle_f40s_generate.zip` | `F40-S` generation |
| `bundle_f40m_generate.zip` | `F40-M` generation |
| `bundle_l99s_generate.zip` | `L99-S` generation |
| `bundle_l99m_generate.zip` | `L99-M` generation |
| `bundle_train_labA.zip` | balanced Lab-A training-job share |
| `bundle_train_labB.zip` | complementary balanced Lab-B training-job share |

The four generation manifests bind their exact stage contracts. The two training
manifests are disjoint and their authenticated union is the complete job grid.
Partial publication is forbidden.

## External authorization gates

Complete these in order. Evidence produced before the final source bytes or on
another commit does not qualify.

1. Run the full source-sensitive MATLAB/Python, mutation, provenance, protocol,
   generation, and training checks. Refresh the source-bound model-form evidence,
   deliberately disposition every untracked path, and create a clean commit A.
   Recompute and retain the MATLAB generator and Python runtime source roots.
2. On the benchmark RTX 5060 host at clean A, create an existing external
   receipt directory, then run the fresh-only publisher (with `PYTHONPATH` and
   `PYTHONHOME` absent and the locked environment active):

   ```powershell
   Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
   Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
   $env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
   # If CUDA_VISIBLE_DEVICES is needed, set it once and keep it unchanged for
   # both the capacity and benchmark commands.
   ```

   `.\.venv-campaign-py313\Scripts\python.exe -B capacity_preflight_compute.py --receipt-dir "<canonical-absolute-external-directory>"`

   The command runs all 16 registered worst-case CUDA probes after establishing
   the registered deterministic mode. It refuses an in-repository directory or
   an existing content-addressed target. Retain the printed receipt path and
   pass it unchanged to the benchmark on the same otherwise-idle physical GPU,
   Python environment, and shell environment:

   `.\.venv-campaign-py313\Scripts\python.exe -B benchmark_paper1_compute.py --capacity-receipt "<printed-capacity-receipt.json>" --output-dir "<new-canonical-absolute-external-run-directory>"`

   Its dispatch-gating v2 evidence must come from one fresh, uninterrupted
   100-trial `F40-S`
   `RAW_POS1_LSTM1_MR1` study on physical channel 1, using the full
   305-state x 50-passage x 5,831-sample non-scientific fixture. Any OOM,
   pre-existing/partial study, retry, replacement, or nonterminal/failed trial
   makes that run nonqualifying.
3. On every intended MATLAB generation PC, freshly generate and execute
   qualification micros for all four blocks from A. Retain each host receipt,
   compare every unordered host pair for every block, and validate the complete
   qualification-receipt inventory. Host labels are trusted-operator
   self-attestation, not cryptographic hardware identity.
4. On one already-qualified reference host, run the source-bound contact and
   time-step closure over all 420 micro passages: 93 `F40-S`, 93 `F40-M`,
   117 `L99-S`, and 117 `L99-M` cases. Each case is solved at requested
   1, 0.5, and 0.25 ms. Independently verify the complete result and create the
   external contact-authorization receipt. This is bounded engineering closure
   for the bilateral solver, not validation of separation/re-contact physics.
5. While HEAD is still clean A, run `dispatch_authorization.py create` with the
   absolute external benchmark, qualification graph/inventory, contact result,
   and contact receipt. The create-once authorization manifest and underlying
   evidence remain outside the repository.
6. Independently review the evidence, then create report-only commit B by
   changing only `docs/audit_r5_results.md`. Its fixed header must identify A and
   the exact external authorization-manifest SHA-256 and use the exact
   `PAPER-1 DISPATCH AUTHORIZED` status required by the builder.
7. From clean B, build the complete set in one operation:

   `.\.venv-campaign-py313\Scripts\python.exe -B build_stage_bundles.py --dispatch-authorization-manifest "<absolute-manifest.json>"`

   The builder revalidates A/B lineage and all external evidence before writing
   any ZIP. Verify every archive against `bundle_sha256.txt`.

Qualification, capacity, benchmark, and contact work are non-scientific gate
runs. They do not authorize starting production data generation or model
selection early.

## Authorized run procedure

After the six ZIPs exist:

1. Extract each ZIP into a fresh workspace. Never overlay old data, caches,
   studies, or results.
2. Run the bundle preflights under the exact qualified MATLAB/Python/CUDA
   environments. A skip is not a pass.
3. For a generation bundle, run the preset `scour_MATLAB/A00_Run.m` unchanged.
   The included `generation_bundle_manifest.json` is authoritative. When
   `0001.mat` is complete, run MATLAB raw parity and then the Python raw-parity
   checker sequentially; stop generation if either fails.
4. For a training bundle, set `TTBI_TRAINING_JOB_MANIFEST` to the absolute
   `training_job_manifest.json` path. Execute only assigned job IDs through
   `comprehensive_ablation_multidamage.py --execute-job <job-id>`. Do not move,
   omit, or run a job on the other host.
5. Preserve every manifest, receipt, source/protocol hash, Optuna database,
   freeze artifact, model, scaler, and result sidecar. Never repair a failed
   scientific run by editing evidence.

The generated `README_BUNDLE.md` is the immediate per-host instruction sheet.
If it disagrees with this guide, stop: source or bundle lineage has drifted.

## Claim boundary

No current Paper 1 performance result, champion, sensor-pair recommendation, or
deployment asset exists until the authorized campaign completes. Conclusions
will be conditional on the registered simulator, finite state design, operating
distribution, architecture/search space, and physical response-channel proxies.
They will not establish field validity, physical sensor packaging/placement, or
a universal best architecture.

The separate Fernandes reconstruction/extension (`F25-R`/`F25-X`) uses its own
experiment ID, manifests, roots, and bundles. It is not part of this six-bundle
Paper 1 publication set.

## Isolated F25 block

Build the two F25 archives while HEAD is the clean commit A that will remain
their scientific lineage:

`.\.venv-campaign-py313\Scripts\python.exe -B build_f25_bundles.py --check-only`

`.\.venv-campaign-py313\Scripts\python.exe -B build_f25_bundles.py`

Retain the printed SHA-256 values. On the RTX 2060, use the included builder's
`--verify-pair` mode with both retained archive hashes and the retained clean-A
commit, then co-extract both archives only into one new empty workspace. Their
common source files are byte-identical; their
experiment-qualified plans, manifests, and READMEs remain distinct. Follow the
generated `README_F25-R.md`/`README_F25-X.md`: run the genuine four-case CUDA
capacity gate (two registered pair-envelope cases plus two conservative
full-eight non-job stresses), generate the shared dataset once, complete all
eight F25-R jobs, then execute all 156 F25-X jobs on the same GPU/numeric stack.
No individual sensor, pair, or architecture may move to another GPU.

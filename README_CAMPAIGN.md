# Paper 1 production campaign

> **Status (2026-08-26): source under final audit; portable dispatch flow ready.**
>
> The four-block implementation and model-form decisions are complete. After the
> source-sensitive checks pass, publish all six ZIPs from one clean, immutable
> source commit. Each destination PC qualifies its own required capabilities and
> runs the local physics/numerical smokes before retained work. MATLAB, Python,
> package, CUDA, and GPU identities are recorded as provenance, not matched as an
> eligibility condition.

This is the concise operator guide. The controlling scientific specification is
[`docs/paper1_campaign_plan.md`](docs/paper1_campaign_plan.md). Older audit and
authorization reports are historical records and do not control the current
portable bundle flow. The methodology and framework-rationale files are likewise
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
- one generated FRA class-4 profile realization (phase seed `20260728`) shared
  by every production state and passage; operational speed/temperature/vehicle
  variability is enabled, while track-damage and wheel-OOR mechanisms remain
  disabled;
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
studies (16,000 trial slots) and 1,440 listed refit jobs. Authenticated aliases
may reduce executed work but may not remove jobs from the inventory or duplicate
a canonical fit. `F40-S` selects one pair among the six sprung, measurable
vehicle responses; the two constrained-wheelset acceleration proxies remain
diagnostic and are not learning inputs. The selected-pair pipelines are then
tuned independently within every block. Outer-test data stay sealed until the
corresponding block-freeze artifact is authenticated.

Hardware allocation is by registered seed parity, not by architecture,
representation, channel, or outcome. `LabA` and `LabB` are logical, balanced job
partitions; they may run on different capable GPUs and numerical stacks. Exact
runtime identity is retained in each result's provenance, while scientific
comparability comes from the fixed protocol, balanced allocation, code hashes,
and authenticated artifacts.

That 1,600-job primary grid is the entire executable scope of these six ZIPs.
Modern-TCN and TSLANet challengers currently exist only as audited
contract/model definitions: there is no challenger executor or challenger job
manifest in this dispatch set, so no challenger run or result is claimable from
it. A later, separately audited challenger package may use only the
authenticated F40-S selected sensor pair; it may not reopen channel selection.

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

## Portable publication gate

Complete these steps in order:

1. Run the full source-sensitive MATLAB/Python, mutation, provenance, protocol,
   generation, contact/time-step, and training check suite. Deliberately
   disposition every untracked path and create one clean campaign commit.
2. From that clean commit, build the complete six-ZIP set in one operation:

   `python -B build_stage_bundles.py --check-only`

   `python -B build_stage_bundles.py`

   Verify every archive against `bundle_sha256.txt`. The builder reads the
   scientific payload from the clean commit, so all six bundles share the same
   code lineage and source manifests.
3. Copy the ZIPs and their SHA-256 file to the assigned PCs. Verify the archive
   digest before extraction and extract each bundle into a fresh workspace;
   never overlay old data, caches, studies, or results. For Python work, install
   the direct dependencies from `requirements-portable.txt` using versions and
   a CUDA-enabled PyTorch build compatible with that PC. The fully pinned
   `requirements-campaign-py313-cu128.txt` is only an optional known-good
   fallback. The extracted `paper1_bundle_identity.json` then binds the verified
   ZIP to its source commit label, source manifest, executable source roots, and
   generation/training manifest. A `.git` directory is not included or required.
4. On every generation PC, run the included capability check and local
   physics/numerical smokes, including healthy and damaged micro cases. A
   MATLAB release comparison may be retained as an optional diagnostic, but
   equality with another PC or with the known-good reference release is not a
   production gate.
5. On every training PC, run its own genuine CUDA capacity preflight for the
   registered worst-case two-channel workload. Production may start there only
   after the local forward/loss/backward/optimizer probe passes without OOM and
   with the required headroom. A timing benchmark is useful for scheduling but
   is optional and cannot authorize or veto scientific work.

   `python -B capacity_preflight_compute.py --all-stages --receipt-dir "$env:TTBI_EXECUTION_RECEIPT_DIR"`

   This creates four distinct receipts, one for each independent execution
   block (`F40-S`, `F40-M`, `L99-S`, and `L99-M`).

The known-good MATLAB and Python environment descriptors remain useful setup
references. Different MATLAB releases, Python/package versions, CUDA versions,
and GPU models are accepted when the required capabilities and local smokes pass.
Their exact identities are still saved in provenance. Source/protocol hashes,
scientific manifests, and resume identity checks remain fail-closed.

## Run procedure

After local qualification:

1. For a generation bundle, A00 must use its own directory as MATLAB's current
   working directory; it rejects every other directory. In MATLAB, replace the
   placeholder and run exactly:

   ```matlab
   cd('<ABSOLUTE_EXTRACTED_BUNDLE>\scour_MATLAB')
   A00_Run
   ```

   `A00_Run.m` remains unchanged and byte-identical in all six ZIPs; the
   authenticated `generation_bundle_manifest.json` selects the stage and exact
   count tuple. A00 writes the live generation folder inside the extracted
   workspace at `scour_MATLAB/Results/<case_name>`. Do not redirect that
   in-progress folder to `TTBI_RESULTS_ROOT` or `TTBI_DATA_ROOT`.

   When `0001.mat` is complete, run MATLAB raw parity from `scour_MATLAB`:

   ```matlab
   smoke_raw_parity('Results/<case_name>')
   ```

   Then, from the extracted bundle root, run the Python half exactly:

   ```powershell
   python check_raw_parity.py 'scour_MATLAB/Results/<case_name>'
   ```

   Equivalently, the two command templates are
   `smoke_raw_parity('<folder>')` and
   `python check_raw_parity.py '<folder>'`. Stop generation if either fails.
2. Do not start training from `scour_MATLAB/Results/<case_name>`. Generation is
   complete only when `_GENERATION_COMPLETE`, `file_digests.mat`, all numbered
   state files, and both parity checks pass. Copy each completed folder without
   editing its contents into an initially absent registered destination:

   | generation stage | required training directory |
   |---|---|
   | `F40-S` | `<shared-root>/data/F40-S_L40_st305` |
   | `F40-M` | `<shared-root>/data/F40-M_L40_st425` |
   | `L99-S` | `<shared-root>/data/L99-S_L99.6_st475` |
   | `L99-M` | `<shared-root>/data/L99-M_L99.6_st475` |

   This PowerShell helper refuses an overlay and verifies a sorted SHA-256
   inventory after the byte-preserving copy:

   ```powershell
   function Copy-TtbiDataset(
       [string]$CompletedCaseFolder,
       [string]$DataRoot,
       [string]$RegisteredDataset
   ) {
       $source = (Resolve-Path -LiteralPath $CompletedCaseFolder).Path.TrimEnd('\')
       $data = (Resolve-Path -LiteralPath $DataRoot).Path.TrimEnd('\')
       if ((Split-Path -Leaf $data) -ne 'data') {
           throw 'TTBI_DATA_ROOT must have basename data'
       }
       $destination = Join-Path $data $RegisteredDataset
       $sourceInventory = "$destination.source.sha256"
       $copyInventory = "$destination.copied.sha256"
       if ((Test-Path -LiteralPath $destination) -or
           (Test-Path -LiteralPath $sourceInventory) -or
           (Test-Path -LiteralPath $copyInventory)) {
           throw "Dataset destination/evidence already exists: $destination"
       }
       $before = @(Get-ChildItem -LiteralPath $source -Recurse -File -Force |
           Sort-Object { $_.FullName.Substring($source.Length + 1) } |
           ForEach-Object {
               $relative = $_.FullName.Substring($source.Length + 1)
               $sha = (Get-FileHash -Algorithm SHA256 `
                   -LiteralPath $_.FullName).Hash.ToLowerInvariant()
               "$sha  $relative"
           })
       Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
       $target = (Resolve-Path -LiteralPath $destination).Path.TrimEnd('\')
       $after = @(Get-ChildItem -LiteralPath $target -Recurse -File -Force |
           Sort-Object { $_.FullName.Substring($target.Length + 1) } |
           ForEach-Object {
               $relative = $_.FullName.Substring($target.Length + 1)
               $sha = (Get-FileHash -Algorithm SHA256 `
                   -LiteralPath $_.FullName).Hash.ToLowerInvariant()
               "$sha  $relative"
           })
       if (Compare-Object $before $after) {
           throw "Dataset copy failed byte-inventory verification"
       }
       $before | Set-Content -LiteralPath $sourceInventory -Encoding ascii
       $after | Set-Content -LiteralPath $copyInventory -Encoding ascii
   }
   ```

   Invoke it once per completed bundle (replace each printed `<case_name>`):

   ```powershell
   Copy-TtbiDataset '<F40-S ZIP>\scour_MATLAB\Results\<case_name>' `
       'D:\ttbi-paper1\data' 'F40-S_L40_st305'
   Copy-TtbiDataset '<F40-M ZIP>\scour_MATLAB\Results\<case_name>' `
       'D:\ttbi-paper1\data' 'F40-M_L40_st425'
   Copy-TtbiDataset '<L99-S ZIP>\scour_MATLAB\Results\<case_name>' `
       'D:\ttbi-paper1\data' 'L99-S_L99.6_st475'
   Copy-TtbiDataset '<L99-M ZIP>\scour_MATLAB\Results\<case_name>' `
       'D:\ttbi-paper1\data' 'L99-M_L99.6_st475'
   ```

   Retain both inventories. Before any job, the Python loader independently
   authenticates the completion marker, `file_digests.mat`, every listed file,
   stage/dataset contract, generator source root, and state provenance. A copy
   or rename does not bypass these checks; the first mismatch stops before a
   cache, Optuna study, or result is created.
3. In each extracted training bundle, set all execution roots explicitly. The
   `data` directory must already exist; the other roots are durable and must not
   be inside the disposable extracted source workspace. Both partitions use the
   same prospective run tag:

   ```powershell
   $env:TTBI_DATA_ROOT = 'D:\ttbi-paper1\data'
   $env:TTBI_RESULTS_ROOT = 'D:\ttbi-paper1\results-labA' # labB on the other PC
   $env:TTBI_CACHE_ROOT = 'D:\ttbi-paper1\cache-labA'
   $env:TTBI_STUDY_ROOT = 'D:\ttbi-paper1\studies-labA'
   $env:TTBI_EXECUTION_RECEIPT_DIR = 'D:\ttbi-paper1\receipts-labA'
   $env:TTBI_CAMPAIGN_RUN_TAG = 'paper1-production-001'
   $env:TTBI_TRAINING_JOB_MANIFEST = `
       (Resolve-Path '.\training_job_manifest.json').Path
   python -B comprehensive_ablation_multidamage.py --validate-manifest
   python -B capacity_preflight_compute.py `
       --all-stages `
       --receipt-dir "$env:TTBI_EXECUTION_RECEIPT_DIR"
   ```

   Use the corresponding `labB` roots on that partition. Root paths may differ
   between PCs; the run tag, source/protocol identities, and redistributed
   artifact SHA-256 values may not.
4. Execute only the current phase's locally assigned jobs. This PowerShell
   helper reads the authenticated local manifest; it does not invent or move a
   job between LabA and LabB:

   ```powershell
   function Invoke-TtbiPhase([string]$Phase, [string]$Stage = '') {
       $manifest = Get-Content -Raw -LiteralPath `
           $env:TTBI_TRAINING_JOB_MANIFEST | ConvertFrom-Json
       $jobs = @($manifest.jobs | Where-Object {
           $_.phase -eq $Phase -and (!$Stage -or $_.stage -eq $Stage)
       })
       if ($jobs.Count -eq 0) { throw "No assigned jobs for $Phase $Stage" }
       foreach ($job in $jobs) {
           python -B comprehensive_ablation_multidamage.py `
               --execute-job $job.job_id
           if ($LASTEXITCODE -ne 0) { throw "Failed job $($job.job_id)" }
       }
   }
   ```

5. Treat every transition below as a two-lab barrier. LabA and LabB may run
   jobs concurrently within the unlocked phase, but the publisher must see the
   authenticated union of all result packages and, where used, all per-job
   Optuna SQLite files. There are two supported layouts:

   - point both partitions at shared durable `TTBI_RESULTS_ROOT` and
   `TTBI_STUDY_ROOT` paths with reliable filesystem semantics; job IDs and
     SQLite filenames are disjoint;
   - keep local roots and consolidate both trees into one publisher workspace
   after each phase. The merge must preserve relative paths and bytes, accept
     a collision only when SHA-256 is identical, and verify every copied file.

   After consolidation, make that same authenticated union available to both
   partitions (shared path or the same verified merge back into each durable
   root) before any dependent job starts. A publisher-only copy is insufficient:
   downstream executors also authenticate upstream result and SQLite evidence.

   A byte-verifying PowerShell merge for the second layout is:

   ```powershell
   function Merge-TtbiTree(
       [string]$SourceRoot,
       [string]$TargetRoot,
       [string]$EvidencePrefix
   ) {
       $source = (Resolve-Path -LiteralPath $SourceRoot).Path.TrimEnd('\')
       New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
       $target = (Resolve-Path -LiteralPath $TargetRoot).Path.TrimEnd('\')
       $sourceInventoryPath = "$EvidencePrefix.source.sha256"
       $destinationInventoryPath = "$EvidencePrefix.destination.sha256"
       if ((Test-Path -LiteralPath $sourceInventoryPath) -or
           (Test-Path -LiteralPath $destinationInventoryPath)) {
           throw "Merge evidence already exists: $EvidencePrefix"
       }
       $evidenceParent = Split-Path -Parent $EvidencePrefix
       New-Item -ItemType Directory -Force -Path $evidenceParent | Out-Null
       $sourceInventory = @()
       $destinationInventory = @()
       Get-ChildItem -LiteralPath $source -Recurse -File -Force |
           Sort-Object { $_.FullName.Substring($source.Length + 1) } |
           ForEach-Object {
           $relative = $_.FullName.Substring($source.Length + 1)
           $destination = Join-Path $target $relative
           New-Item -ItemType Directory -Force `
               -Path (Split-Path -Parent $destination) | Out-Null
           $expected = (Get-FileHash -Algorithm SHA256 `
               -LiteralPath $_.FullName).Hash.ToLowerInvariant()
           if (Test-Path -LiteralPath $destination) {
               $present = (Get-FileHash -Algorithm SHA256 `
                   -LiteralPath $destination).Hash.ToLowerInvariant()
               if ($present -ne $expected) {
                   throw "Nonidentical merge collision: $relative"
               }
           } else {
               Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
           }
           $copied = (Get-FileHash -Algorithm SHA256 `
               -LiteralPath $destination).Hash.ToLowerInvariant()
           if ($copied -ne $expected) { throw "Copy drift: $relative" }
           $sourceInventory += "$expected  $relative"
           $destinationInventory += "$copied  $relative"
       }
       if (Compare-Object $sourceInventory $destinationInventory) {
           throw "Merge inventory differs after verified copy"
       }
       $sourceInventory | Set-Content `
           -LiteralPath $sourceInventoryPath -Encoding ascii
       $destinationInventory | Set-Content `
           -LiteralPath $destinationInventoryPath -Encoding ascii
       Write-Output "WROTE $sourceInventoryPath"
       Write-Output "WROTE $destinationInventoryPath"
   }
   # Run all four calls at each barrier, then point the publisher at these roots:
   Merge-TtbiTree 'D:\labA\results' 'D:\publisher\results' `
       'D:\publisher\evidence\barrier-N-labA-results'
   Merge-TtbiTree 'D:\labB\results' 'D:\publisher\results' `
       'D:\publisher\evidence\barrier-N-labB-results'
   Merge-TtbiTree 'D:\labA\studies' 'D:\publisher\studies' `
       'D:\publisher\evidence\barrier-N-labA-studies'
   Merge-TtbiTree 'D:\labB\studies' 'D:\publisher\studies' `
       'D:\publisher\evidence\barrier-N-labB-studies'
   $env:TTBI_RESULTS_ROOT = 'D:\publisher\results'
   $env:TTBI_STUDY_ROOT = 'D:\publisher\studies'
   ```

   Replace `barrier-N` with a fresh barrier identifier. The helper refuses to
   replace old evidence, writes and prints the paired sorted source/destination
   SHA-256 inventory paths, and verifies them before returning. Retain both
   files with the campaign evidence. Do not merge caches, edit SQLite databases,
   or use an overlay copy that silently replaces a differing file.

6. Run the F40-S selection barriers in this exact order on both partitions and
   publish once from the consolidated roots:

   ```powershell
   # Both labs: factorial HPO, then STOP and consolidate results + studies.
   Invoke-TtbiPhase 'f40s_factorial_hpo'

   # Make the authenticated factorial union available at both labs before
   # either lab starts the dependent OOF jobs. Then run both labs and STOP
   # again to consolidate the 480 development results.
   Invoke-TtbiPhase 'f40s_development_adjudication'

   # Publisher only, after all 480 development jobs authenticate:
   $artifactRoot = 'D:\ttbi-paper1\artifacts'
   New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
   $adjudication = Join-Path $artifactRoot 'f40s_adjudication.json'
   python -B comprehensive_ablation_multidamage.py `
       --publish-adjudication $adjudication
   $adjudicationFileSha = (Get-FileHash -Algorithm SHA256 `
       -LiteralPath $adjudication).Hash.ToLowerInvariant()
   $adjudicationArtifactSha = `
       (Get-Content -Raw -LiteralPath $adjudication | ConvertFrom-Json).artifact_sha256

   # Copy this exact file to each lab, verify the SHA, then set both variables
   # at each lab before its channel-screen jobs:
   $env:TTBI_PAPER1_ADJUDICATION_ARTIFACT = $adjudication
   $env:TTBI_PAPER1_ADJUDICATION_ARTIFACT_SHA256 = `
       $adjudicationArtifactSha
   Invoke-TtbiPhase 'f40s_frozen_hyperparameter_channel_screen'

   # STOP, consolidate both labs again; publisher writes tensor + selection:
   $channel = Join-Path $artifactRoot 'f40s_channel_selection.json'
   $selection = Join-Path $artifactRoot 'f40s_selection.json'
   $env:TTBI_PAPER1_CHANNEL_SELECTION_ARTIFACT = $channel
   $env:TTBI_PAPER1_SELECTION_ARTIFACT = $selection
   python -B comprehensive_ablation_multidamage.py `
       --publish-channel-selection $channel
   $selectionFileSha = (Get-FileHash -Algorithm SHA256 `
       -LiteralPath $selection).Hash.ToLowerInvariant()
   $selectionArtifactSha = `
       (Get-Content -Raw -LiteralPath $selection | ConvertFrom-Json).artifact_sha256
   ```

   Copy the adjudication, channel tensor, and compact selection plus their
   independently recorded SHA-256 values back to both labs. On each lab, set
   `TTBI_PAPER1_SELECTION_ARTIFACT` to its local absolute copy and
   `TTBI_PAPER1_SELECTION_ARTIFACT_SHA256` to `$selectionArtifactSha` before
   continuing. Use `$selectionFileSha` to verify transfer bytes; the artifact
   SHA is the independent content identity required by the loader.

7. With the selection authenticated on both labs, run selected-pair HPO. A
   stage can be frozen only after both labs' five-restart result/SQLite union
   for every unique resolved pipeline in that stage is consolidated:

   ```powershell
   Invoke-TtbiPhase 'f40s_selected_pair_hpo' 'F40-S'
   Invoke-TtbiPhase 'block_selected_pair_hpo' 'F40-M'
   Invoke-TtbiPhase 'block_selected_pair_hpo' 'L99-S'
   Invoke-TtbiPhase 'block_selected_pair_hpo' 'L99-M'

   # Publisher only; repeat after the corresponding stage barrier:
   foreach ($stage in @('F40-S','F40-M','L99-S','L99-M')) {
       $freeze = Join-Path $artifactRoot ("freeze-$stage.json")
       python -B comprehensive_ablation_multidamage.py `
           --publish-block-freeze $stage $freeze
       if ($LASTEXITCODE -ne 0) { throw "Freeze failed: $stage" }
       $fileSha = (Get-FileHash -Algorithm SHA256 `
           -LiteralPath $freeze).Hash.ToLowerInvariant()
       $artifactSha = `
           (Get-Content -Raw -LiteralPath $freeze | ConvertFrom-Json).artifact_sha256
       "$stage file=$fileSha artifact=$artifactSha"
   }
   ```

   Redistribute all four freezes. Before a stage's sealed jobs, point
   `TTBI_PAPER1_BLOCK_FREEZE_ARTIFACT` at the local absolute copy and set
   `TTBI_PAPER1_BLOCK_FREEZE_ARTIFACT_SHA256` from its JSON `artifact_sha256`
   field. Verify the separate file SHA after every transfer. Run
   `post_freeze_sealed_test_stability` stage by stage on both labs, consolidating
   after each stage. For example:

   ```powershell
   $stage = 'F40-S'
   $freeze = "D:\ttbi-paper1\artifacts\freeze-$stage.json"
   $env:TTBI_PAPER1_BLOCK_FREEZE_ARTIFACT = $freeze
   $env:TTBI_PAPER1_BLOCK_FREEZE_ARTIFACT_SHA256 = `
       (Get-Content -Raw -LiteralPath $freeze | ConvertFrom-Json).artifact_sha256
   Invoke-TtbiPhase 'post_freeze_sealed_test_stability' $stage

   # Repeat the block above with each stage-local freeze. Secondary transfer
   # always cites the authenticated F40-S freeze, even for downstream stages:
   $f40sFreeze = 'D:\ttbi-paper1\artifacts\freeze-F40-S.json'
   $env:TTBI_PAPER1_BLOCK_FREEZE_ARTIFACT = $f40sFreeze
   $env:TTBI_PAPER1_BLOCK_FREEZE_ARTIFACT_SHA256 = `
       (Get-Content -Raw -LiteralPath $f40sFreeze | ConvertFrom-Json).artifact_sha256
   Invoke-TtbiPhase 'secondary_frozen_hyperparameter_transfer'
   ```

   Secondary transfer is report-only for the downstream stages.
8. Preserve every manifest, receipt, source/protocol hash, Optuna database,
   freeze artifact, model, scaler, and result sidecar. Resume only when the
   scientific job identity and authenticated artifacts agree. Never repair a
   failed scientific run by editing evidence.

The generated `README_BUNDLE.md` is the immediate per-host instruction sheet.
If it disagrees with this guide, stop: source or bundle lineage has drifted.

## Claim boundary

No current Paper 1 performance result, champion, sensor-pair recommendation, or
deployment asset exists until the complete campaign finishes. Conclusions
will be conditional on the registered simulator, finite state design, operating
distribution, architecture/search space, and physical response-channel proxies.
They will not establish field validity, physical sensor packaging/placement, or
a universal best architecture.

The separate Fernandes reconstruction/extension (`F25-R`/`F25-X`) uses its own
experiment ID, manifests, roots, and bundles. It is not part of this six-bundle
Paper 1 publication set.

## Isolated F25 block

Build the two F25 archives from the same clean source revision that will remain
their scientific lineage:

`python -B build_f25_bundles.py --check-only`

`python -B build_f25_bundles.py`

Retain the printed SHA-256 values. Verify the pair with both retained archive
hashes and the retained source commit, then co-extract both archives into a new
empty workspace on every participating PC. Their common source files are
byte-identical; their experiment-qualified plans, manifests, and READMEs remain
distinct. Follow the generated `README_F25-R.md`/`README_F25-X.md`: every
training PC runs the genuine two-case CUDA capacity gate (the worst registered
RAW pair envelopes at kernels 2/5); one coordinator generates the shared fixed
Type-2 dataset; the eight F25-R and 99 F25-X jobs may then be distributed by the
documented dependency rounds. At every round barrier, use the included
append-only byte-verifying merge in both directions. Distinct jobs may use
different capable PCs and numerical stacks, while an in-progress job remains on
one host so its resume identity stays exact.

# R12 intake handoff — independent audit of Claude's 2026-08-01 return

> **INTAKE ONLY — CLAUDE'S CHANGES HAVE NOT BEEN ACCEPTED BY CODEX.**
>
> Do not create commit A, stage files, publish an OSF/Zenodo deposit, start the
> 100-trial benchmark, qualify hosts, or dispatch bundles from this state.
> The next Codex session must verify the edits and sources directly, append its
> verdict to `paper1/ISSUES_FOUND.md`, and must not edit `.tex` files.

## 1. Purpose and reading order

This file packages the transition between the completed Codex R11 review and
Claude's subsequent manuscript/source corrections. It is a resumption aid, not
a scientific verdict or authorization record.

Read these files completely, in this order:

1. `paper1/ISSUES_FOUND.md`:
   - Codex's accepted baseline review starts at line 1246;
   - Claude's new response starts at line 1501 and currently ends at line 1799;
   - pay special attention to Claude's “Self-audit round 2” at line 1657.
2. `docs/audit_r11_handoff.md` — the last independently accepted pre-Claude
   state and queue.
3. This intake handoff.
4. `docs/protocol_deposit.md` — a new specification, **not a deposit**.
5. `docs/paper1_methodology.md`, `docs/track_eov_sampling_spec.md`, and
   `docs/framework_rationale.md`.
6. Every changed manuscript section, `paper1/references.bib`, and
   `paper1/MISSING_PRIMARY_SOURCES.md`.

Do not trust the claims “fixed”, “closed”, “clean”, “independent”, “68/68”, or
“rebuilds” until the current files and primary artifacts reproduce them.

## 2. Exact intake snapshot

- Captured: 2026-08-01, after Claude's response and before this handoff file.
- HEAD: `865728f801c83a642b06a223f2a22b33f2b429b7`.
- Expanded status before adding this packet: 340 paths = 73 tracked changes +
  267 untracked paths.
- SHA-256 of the exact LF-joined output of
  `git status --porcelain=v1 --untracked-files=all` (with a final LF), before
  adding this packet:
  `f56cf4a3d0263ea0b7a290c66e1d01fe7519a10068bc54e821b3151184497f20`.
- Expected after adding this packet: 341 paths = 73 tracked changes + 268
  untracked paths.
- Post-packet status-list SHA-256:
  `e8acd4b2cd507b4aeb27042fe46e03abacab5e201dc0c7809539377b7c35aa0e`.

The repository source snapshot was recomputed rather than copied from Claude's
report:

- reviewed manifest: 429 entries;
- generator root: 297 files,
  `c34ba6d6ab166b79b2b9e6e6e45fd5ef29d952f5aa43b2a755ba8e2dd9820b3f`;
- Python-runtime root: 124 files,
  `a5d3815db9fdc85dc8ca77626938de07ebcb2a6d362d1f7b79ebb0b625a2c35c`.

These roots are byte-identical to the accepted R11 working-tree roots. This is
plausible because the new edits are manuscript/Markdown/BibTeX files outside
the two hashed roots. It does **not** make the manuscript corrections valid and
does not make the dirty tree commit-bound.

Process note: while this packet was validated, a Claude-launched
`matlab -batch run_full_suite` was active, but its parent command had changed
directory to the separate `TLT-Optimizers` workspace and redirected output to
that workspace's `run_full_suite_after_catalog.txt`. It is not a process or
residue of `drive-by-scour-dt` and was not interrupted. Future process audits
must inspect command lines/working context rather than count every MATLAB child
globally.

The required commit-A union remains expected-red: 443 bundle/manuscript files,
182 regular tracked blobs, and 261 absent from HEAD. Claude reports all 261 are
present on disk but untracked. Reproduce that set construction; do not infer it
from the total Git status count, which also contains non-required artifacts.

## 3. Pre-Claude versus Claude-return manuscript hashes

The pre-Claude column is the hash set independently preserved by Codex after
confirming that its own review had not edited `.tex`. The return column was read
after Claude's new response. `main.tex` is unchanged; the eight section files
changed and require review.

| File | Pre-Claude SHA-256 | Claude-return SHA-256 |
|---|---|---|
| `paper1/main.tex` | `e47bdd6a059147fd0655ffdae6ff5082088f3045ecab0ba6a89abd441d305017` | `e47bdd6a059147fd0655ffdae6ff5082088f3045ecab0ba6a89abd441d305017` |
| `paper1/sections/abstract.tex` | `b314fd06a4ce6eef3e8c5105230d289282a888f4936e3357c79b7d0d765a4c18` | `44ec7c2b3fe1e9e48892f434e8f7eed41483f4b2dfdf24dc95684e72b9a2c426` |
| `paper1/sections/conclusion.tex` | `fa375cd8704846f5e8f990d59ce01347b4651c3b8208094eadda9e0f3383b2d8` | `28600b264b5606f727e9f9f4f89fef4ef8d830ce63b87ad242d251494de919a9` |
| `paper1/sections/data_processing.tex` | `94886e1343240c1ffc6d7f82b741c54680358344de7705a1e4df5cda5892d0cd` | `a3b291b5c769feed895c8b104335d2512e54ba0be67ac79418a8754ac76c43e3` |
| `paper1/sections/framework.tex` | `acedf9e7d98223cb290dfe11b091faf5984bfc84b6649cb64e89b2fd73f8242e` | `47d8b0b191b34b816d0b1c4e4467a6e90e6dc04d90449ac6fa890a21d683a095` |
| `paper1/sections/introduction.tex` | `bbef3a8950a6c1b9208c87a17282bee5fbebe446d377def7b22abe8e6f04a635` | `7423bbbfdbece8bccd1061ed0f304a4d53d1378275db203f2fbda83bddab78f5` |
| `paper1/sections/limitations.tex` | `4a80d39dbfb967b6905244999b3f19eb97f12cd21ccdf224feeef3361817d0da` | `92aaa85f734d5b775bc2415a8b957bd4f7a9634d503af0057a8f96458b553af8` |
| `paper1/sections/numerical_simulation.tex` | `5eb2ac4604175a8dff13f5556458e57e3dea1c090129c71f3de0f9b2c96d429e` | `c74f480238a6372af51527031d4df31ea975274307c3517070c871533a675ae6` |
| `paper1/sections/results.tex` | `5e4c976617f59db843d86149c86b4e4c5b38acbd2cddbcbaa567dfcfeea3a624` | `fc87354d939b806924b088aa42b7cb92f28760d6d27e427dd081074c745ea13f` |

Other important Claude-return hashes:

| File | SHA-256 |
|---|---|
| `paper1/references.bib` | `e3a18100dd63fd83bbcf26367b97ba9f709422105dc92067a39e3b0bf20826cb` |
| `paper1/MISSING_PRIMARY_SOURCES.md` | `4e2d30f4c21fa131cd0f4fa490fd818e8702fa8b0777470e3830b880f666a0b3` |
| `docs/paper1_methodology.md` | `2dc9cf107f764c0b1b152d01db2f4f7b63d9c4f61344304e6c7d6373a7dd1850` |
| `docs/track_eov_sampling_spec.md` | `4a2a1c438c0e43e394f645ef97e6c9553bafbd21e651c12307f33f4a081782e6` |
| `docs/framework_rationale.md` | `9d7c84363b28912ae9077ed2a9bebd276531ea62301e2140af7423034080d8c7` |
| `docs/protocol_deposit.md` | `da9deb8e83c4c7f9a201dac6d4163744bf35ab5c5e34299fdeb7c3957fed5697` |
| `paper1/ISSUES_FOUND.md` | `e39472dc2413cbeb700c11b5b963a20cea5600a224cf82b5f490277356966506` |

Use these as intake identities, not as approved scientific roots. If any hash
has changed when the next session starts, first determine whether Claude or the
author intentionally made another edit and expand the audit scope.

## 4. What Claude claims to have changed — all unverified

### P1-S1: simulator-law evidence boundaries

Claude reports individual author-chosen/proxy labels for ballast, unsupported
sleepers, pad failure, crack activation, OOR, sampling-window quantities, the
deck-temperature law, vehicle/suspension CoVs, operational LHS bounds, pad
clipping, the near-abutment window, and the crack-location clamp. It also
reports caveats derived from Esmaeili, Wangtawesap, Augustin/Kitahara, FRA,
Guo, Shi, Iwnicki, and RIVAS.

Verify every number, scope condition, and primary-source attribution. In
particular, check the TDA-only 54.7-to-46.6 kN/mm values, the below-15-cm water
condition, relayed versus original unsupported-sleeper observations, FRA class-5
conditioning/open lower bin, Guo's study-specific 2.4 m smoothing, Shi's exact
0/1/3/5 design at 80 km/h, and polygonisation versus generic OOR amplitudes.

### P1-S2: “conservative” and bounding language

Claude reports removing inferential “conservative” wording and two unearned
“bounds error” claims, while retaining the descriptive seven-edge
tail-adjusted-envelope boundary. Search mechanically and then read each residue
semantically; words such as `bound`, `reduce`, `interval`, `robust`, and
`conservative` may be correct in one context and overclaims in another.

### P1-S3: semantic bibliography/source closure

Claude reports 68 cited keys, 68 definitions, zero missing/unused/duplicates,
13 locally backed entries added, and six entries removed. Reproduce the graph
with an independent parser and inspect every new BibTeX entry against its local
artifact. Verify author names, chapter/article DOI scope, dates, titles,
container names, and whether each citation actually supports the adjacent
claim.

One explicit judgement call remains: `garg1984dynamics` was removed because no
local textbook artifact exists, and the FRA profile constants are now
attributed to the directly inspected TTB-2D generator. Decide whether this is a
transparent implementation-provenance statement or an unacceptable loss of
canonical scientific attribution.

Claude also reports closing the Cantero/TTB-2D, VEqMon2D, Zhai-track-property,
and Fernandes-PAA placeholders and adding TTB-2D provenance to the manuscript.
Verify the exact upstream commit/licence claims and keep upstream versus
repository-local scientific mechanisms sharply separated.

### P1-S4: immutable registration is still open

`docs/protocol_deposit.md` is only a proposed deposit specification. No real
OSF/Zenodo locator or date exists, and the manuscript correctly still says no
registry deposit has been made. Do not publish anything during the audit.

The document claims that a rerunnable builder and a 12-file zip were created in
a “session scratchpad”, with deposit root
`15a37b1d991d035a193421ee6610d6199e85b1ecd571928ae5b2ffa3156224ea`.
No `PROTOCOL_IDENTITY.txt`, protocol-deposit zip, or builder was found by a
workspace search when this intake was packaged. Treat reproducibility and
location of that claimed bundle as an open audit question. A transient
scratchpad tool that the next session cannot recover is not a durable
reproducible build procedure.

Before author action, verify the 12-file membership, per-file hashes, root-hash
algorithm, exclusion rationale, environment identities, “no production data”
statement, locator insertion sites, and whether the same tree can be rebuilt
without a hidden session artifact.

### P1-R1: tracked snapshot is still open

Claude reports the same 443/182/261 required-file result and says this becomes a
mechanical `git add` after registration. Reproduce it. Do not stage or commit
until the scientific corrections and external locator are independently
accepted. The installed MATLAB is still R2025b Update 6 while the frozen
post-A qualification requires exact Update 5; do not retarget the lock.

## 5. Unreviewed artifacts requiring classification

The following appeared during Claude's work and were not present in the
accepted 334-path R11 status snapshot:

- repository-root `was` — zero bytes;
- repository-root `not` — zero bytes;
- repository-root `deposited,` — zero bytes;
- `paper1/out.txt` — 119,494 bytes;
- `docs/protocol_deposit.md` — intentional-looking new specification;
- this intake packet itself, once added.

The three zero-byte names strongly resemble an accidental shell tokenization of
the phrase “was not deposited,” but that is an inference. Inspect provenance
and contents first. Do not silently delete or include them in commit A.
Determine whether `paper1/out.txt` is a transient build log and whether it is
fully superseded by reproducible LaTeX output before proposing cleanup.

Claude also reports three deliberately open P3 manuscript issues: table
citation/number order, `sec:noise` attached to an unnumbered paragraph, and the
`tab:seeds` float splitting determinism/capacity prose. Review them, but do not
edit `.tex`; record their actual severity and hand them back to Claude/author.

## 6. Next-session audit protocol

1. Confirm HEAD, expanded status, the hashes in section 3, and tree quiescence.
2. Read all required files in section 1 completely. Inspect actual diffs; do
   not reconstruct edits from Claude's prose.
3. Classify the four suspicious artifacts in section 5 without deleting them.
4. Reproduce the BibTeX/citation graph and compile the manuscript from a clean
   build directory. Claude claims 42 pages, clean biber, and no undefined
   citations/references; verify logs and the rendered PDF, not just exit code.
5. Re-open the relevant primary PDFs and check every source-dependent numerical
   statement and scope qualifier. Prefer direct extraction/page inspection over
   secondary summaries.
6. Audit `docs/protocol_deposit.md` as a protocol and supply-chain artifact.
   Determine what must change before the author may perform the external
   deposit. Do not perform the account-bound deposit.
7. Recompute `repository_source_snapshot()` and the 443/182/261 required-file
   union. `check_campaign_controls.py` should still have exactly one expected
   red: regular tracked blobs.
8. Run lightweight text/BibTeX/build checks first. The long source-sensitive
   technical suites need not be repeated merely because `.tex`/Markdown/BibTeX
   changed outside the two hashed source roots. If any code, manifest, or source
   root differs, expand the suite and run every live-source mutation harness
   exclusively and serially.
9. Append a new independent verdict under the existing `## Codex review` in
   `paper1/ISSUES_FOUND.md`. Do not edit `.tex`.
10. If the text and deposit specification are acceptable, state precise author
    actions for creating the immutable deposit. Commit A remains prohibited
    until the real locator/date are inserted, the exact post-locator tree is
    re-audited, all 443 required files are tracked, and the tree is clean.

## 7. Ready-to-paste prompt for the next Codex session

```text
Continue the independent R12 scientific audit of the repository. Read
docs/audit_r12_intake_handoff.md, paper1/ISSUES_FOUND.md in full (especially
Claude's response beginning at line 1501 and its self-audit round 2), and
docs/audit_r11_handoff.md. Then inspect every actual Claude edit in the
manuscript, bibliography, evidence-boundary documents, and
docs/protocol_deposit.md. Do not trust Claude's reported 68/68 citation graph,
source interpretations, LaTeX rebuild, or deposit hash without reproducing
them. Re-open the primary PDFs for every material numerical/source claim.

Do not edit any .tex file. Record the new verdict under the existing “Codex
review” section of paper1/ISSUES_FOUND.md. Do not stage, commit, deposit,
benchmark, qualify hosts, or dispatch. Classify the suspicious untracked files
was, not, deposited,, and paper1/out.txt without deleting them. Confirm whether
the claimed protocol-deposit builder/zip actually exists and is reproducible.
Run mutation harnesses only serially. If all manuscript/source P1s are closed,
give the exact author steps for the immutable deposit and the post-deposit
re-audit; commit A still waits for the real locator/date and a clean tracked
443-file snapshot.
```

## 8. Stop condition for this intake

This packet intentionally does not judge Claude's corrections. The next session
starts at **verification of the Claude-return tree**, not at commit A and not at
the downstream benchmark/qualification queue.

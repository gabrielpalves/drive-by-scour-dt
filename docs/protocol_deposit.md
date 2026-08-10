# Immutable protocol deposit — specification and hand-off (P1-S4)

> **Status: NOT PURSUED — author decision, 2026-08-01.** No deposit will be
> made, and P1-S4 is closed by *removing the claim that required it* rather
> than by satisfying it. The manuscript no longer invokes preregistration:
> `paper1/sections/introduction.tex` previously read "in the spirit of
> preregistered designs~\cite{nosek2018preregistration}, although no external
> registry deposit was made", and now says only that the specification was
> fixed in versioned, hash-identified source before any production data
> existed, with "registered" explicitly defined as that and as denoting no
> external registry deposit. The `nosek2018preregistration` entry is removed
> from the bibliography (67 keys, 67 entries, still closed).
>
> **Rationale.** Preregistration is a norm in psychology, clinical medicine
> and parts of ecology; it is essentially absent from structural engineering
> and SHM venues, where the equivalent expectation is a data/code availability
> statement plus an archived artifact with a DOI at publication. That archive
> is already planned. P1-S4 arose because the paper reached for
> preregistration culture in one sentence; with that sentence gone, the
> obligation goes with it, and the claim that remains — prospective
> specification in hash-identified source — is one the repository can actually
> evidence through its own source roots, protocol descriptors, and gates.
>
> **This document is retained, not deleted.** The specification below is
> correct and the deposit bundle was built and verified before the decision
> (root hash `15a37b1d991d035a193421ee6610d6199e85b1ecd571928ae5b2ffa3156224ea`
> over 12 files). If a venue or co-author later asks for a registered
> protocol, everything needed is here: re-run the builder against the
> then-current tree, deposit, and insert the locator at the sites in §5.
> Nothing about the campaign depends on it.

> **Everything below this line is HISTORICAL.** It records why the deposit
> was once treated as a pre-A blocker and how it would have been performed.
> None of it is an instruction: the deposit is not required, must not be
> performed as part of the pre-A queue, and the "insertion text" of §5 quotes
> manuscript sentences that no longer exist. Codex R12 (2026-08-01) accepted
> the closure-by-withdrawal and confirmed no external deposit is necessary
> for the revised claim.

## 1. Why this was treated as a blocker (historical)

The manuscript at that time invoked preregistration culture and used
"registered" without a local definition. The R11 audit
(`docs/audit_r11_handoff.md` §P1-S4) rejected the following as substitutes for
registration:

- methodology files that are source-locked inside a **mutable** working tree;
- a **promised future** data-archive DOI;
- repository commit hashes, which the author can rewrite.

Under that reading, only an external deposit that the author cannot silently
alter, carrying a resolvable locator and a deposit date, would convert
"source-locked" into "registered", and it would have to happen before the
campaign generated production data. *The current manuscript makes no such
claim — "registered" is defined locally as versioned, hash-identified source
with no external deposit — so this rationale no longer binds anything.*

## 2. Precondition that would apply if a deposit were ever made (historical)

A deposit is immutable and a superseded protocol is worse than none, so a
deposit would only ever make sense once all of the following were true:

1. P1-S1 is closed: every exact simulator law in
   `paper1/sections/numerical_simulation.tex` is individually classified as an
   author-chosen prior, an engineering proxy, or a source-supported value with
   its scope caveat.
2. P1-S2 is closed: no generic inferential "conservative" language remains.
3. P1-S3 is closed: the citation graph is semantically closed — every cited key
   has an identifiable source artifact, and every essential audited source is
   wired to the claim it supports.
4. `docs/paper1_methodology.md` and `docs/paper1_outline.md` agree with the
   `.tex` on every registered value, statistic, and decision rule.

## 3. What is deposited

The deposit is the **protocol**, not the data and not the full source tree. It
must be sufficient for a reader to check, after the fact, that nothing in the
design or the analysis plan changed once results existed.

### 3.1 Required documents

| Item | Path | Role |
|---|---|---|
| Registered methodology | `docs/paper1_methodology.md` | the design of record |
| Registered outline / analysis plan | `docs/paper1_outline.md` | pre-specified analyses and reporting rules |
| Evidence boundary for the priors | `docs/track_eov_sampling_spec.md` | which values are author-chosen versus source-supported |
| Primary-source audit | `paper1/MISSING_PRIMARY_SOURCES.md` | per-source verdicts behind the boundary |
| Campaign contract | `core/campaign_contract.py` | the machine-checked schema tag and rung/state contract |
| Protocol descriptor module | `core/protocol.py` | the hashing rules and the two-hash design |
| Statistical inference module | `core/statistical_inference.py` | the exact registered resampling procedure |
| Registered manuscript sections | `paper1/sections/numerical_simulation.tex`, `paper1/sections/data_processing.tex`, `paper1/sections/framework.tex`, `paper1/sections/limitations.tex` | the prose form of the same specification |
| Bibliography | `paper1/references.bib` | the evidence graph as of deposit |

Results sections are **excluded** on purpose: at deposit time every result slot
is a `\pending{}` marker and depositing an empty results section adds nothing.

### 3.2 Required identity block

A plain-text `PROTOCOL_IDENTITY.txt` at the deposit root, containing:

- the SHA-256 of every deposited file, one per line, sorted by path;
- a root hash over that sorted list;
- the reviewed generator source root and Python-runtime source root SHA-256
  values from `core.source_provenance.repository_source_snapshot()`;
- the frozen environment identifiers: MATLAB `25.2.0.3177638` (R2025b Update 5)
  and the Python/CUDA descriptor in `environment/campaign-py313-cu128.json`;
- a one-line statement that **no production campaign data existed at deposit
  time**, with the reason (`docs/audit_r5_results.md`: dispatch blocked).

**`protocol_core_hash` is deliberately not deposited** (this corrects an
earlier draft of this document). It is emitted per rung by the driver at run
start and incorporates per-rung inputs — the `sensor_noise` preset and the
architecture set — so no single value is well defined before the campaign
runs. The **Python-runtime source root is the invariant that binds them all**:
every core descriptor embeds it as
`code.python_runtime_source_root_sha256`, so checking that a post-campaign
`protocol_descriptor.json` carries the deposited value proves the analysis code
was the registered one. Depositing the root is therefore stronger, not weaker,
than depositing one rung's core hash.

Generate the hashes from the same working tree that becomes commit A. If a
single byte of a listed file changes afterwards, the deposit is stale and the
locator must not be cited as covering it.

### 3.3 The bundle is already built

A re-runnable builder assembles the deposit directory, writes
`PROTOCOL_IDENTITY.txt`, and zips the result. **Re-run it after any change to a
deposited file**, then upload the fresh zip — the whole value of the deposit is
that its hashes match the tree that becomes commit A.

It lives in the session scratchpad rather than in the repository on purpose:
`check_source_provenance.py` rejects any unmanifested `.py` file inside a
reviewed directory, so adding a builder at the repository root would fail that
gate. Keep it outside the tree, or add it to `bundle_source_files.txt` first.

This document is deliberately **excluded** from the deposit: it records the
deposit's own root hash, so including it would make that hash
self-referential and impossible to state.

State as built on 2026-08-01 (12 files, matching §3.1 exactly):

| Quantity | Value |
|---|---|
| deposit root hash | `15a37b1d991d035a193421ee6610d6199e85b1ecd571928ae5b2ffa3156224ea` |
| generator source root | `c34ba6d6ab166b79b2b9e6e6e45fd5ef29d952f5aa43b2a755ba8e2dd9820b3f` (297 files) |
| Python-runtime source root | `a5d3815db9fdc85dc8ca77626938de07ebcb2a6d362d1f7b79ebb0b625a2c35c` (124 files) |

Both source roots equal the values recorded in `docs/audit_r11_handoff.md`,
confirming that the manuscript and documentation corrections did not disturb
the hashed source boundary.

## 4. Deposit steps (historical — do not perform)

These steps would create an account-bound public record and would have to be
performed by the author, not by an assistant. They are recorded for the
contingency of §0 only; nothing in the pre-A queue calls for them.

1. Assemble the files of §3.1 in a directory and add `PROTOCOL_IDENTITY.txt`.
2. Create the deposit on **OSF** (`https://osf.io`) as a *Registration* — not a
   plain project — so the snapshot is frozen and time-stamped, or on **Zenodo**
   as a versioned record if a DOI is preferred over an OSF registration GUID.
   OSF Registrations are immutable and carry a registration date; Zenodo
   records are immutable per version and carry a DOI.
3. Choose the embargo policy deliberately. An embargoed OSF registration still
   fixes the date and the content hash while keeping the text private until the
   paper is public; this is usually the right choice before submission.
4. Record the returned **locator** (OSF GUID/URL or Zenodo DOI) and the
   **deposit date** exactly as the platform prints them.

## 5. Where a locator would have been inserted (historical — obsolete text)

The "Current text" column below quotes manuscript sentences that **no longer
exist** (the preregistration sentence was deleted on 2026-08-01), so this
table can no longer be executed as written. If a deposit were ever revived,
the insertion sites would need to be re-derived from the then-current text.

| File | Current text | Replace with |
|---|---|---|
| `paper1/sections/introduction.tex` (the "prospectively specified and source-locked" passage) | "…in the spirit of preregistered designs~\cite{nosek2018preregistration}, although no external registry deposit was made." | the same sentence ending in a deposited-protocol clause carrying the real locator and date, and a new `references.bib` entry for the deposit itself |
| `docs/paper1_methodology.md` §1 | no deposit statement | one line naming the locator, the deposit date, and the deposited protocol root hash |
| `docs/protocol_deposit.md` (this file) | "Status: NOT DEPOSITED" | the locator, date, root hash, and the tree state the deposit covers |

Do **not** promote "registered" to "preregistered" anywhere else. The shorthand
definition in the introduction stays; the deposit upgrades its evidentiary
standing, not the paper's vocabulary.

## 6. Re-audit after insertion (historical)

If a deposit were ever made, the audit would require re-checking *the exact
tree that was deposited*, so after inserting the locator:

1. Re-run `check_protocol_hash.py` and confirm the protocol hash is unchanged
   by the insertion (the locator lives in prose and in the bibliography, not in
   any hashed descriptor). If it does change, investigate before proceeding.
2. Re-verify each deposited file's SHA-256 against `PROTOCOL_IDENTITY.txt`.
   Only the three files of §5 are permitted to differ, and only by the locator
   insertion.
3. Recompile `paper1/main.tex` (pdflatex → biber → pdflatex ×2) and confirm zero
   undefined citations and references.
4. Then, and only then, proceed to P1-R1: make all 443 required paths regular
   tracked blobs, obtain a clean tree, and create commit A.

## 7. What the deposit does not buy

It does not make the priors empirical, it does not make the resampling
envelope an inferential guarantee, and it does not qualify this host's MATLAB
stack. Those remain P1-S1, P1-S2, and the post-A Update 5 qualification
blocker respectively.

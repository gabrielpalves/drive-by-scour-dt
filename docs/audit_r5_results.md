# Paper-1 dispatch authorization (legacy filename)

**Status: PAPER-1 DISPATCH BLOCKED.**

**Tested source commit:** PENDING
**Dispatch authorization manifest SHA-256:** PENDING

The filename `audit_r5_results.md` is retained because the fail-closed bundle
builder and authorization lineage bind this path. This pending report does not
authorize production generation, training, bundle publication, or dispatch.

## Contract under review

The candidate Paper 1 campaign has four production blocks:

| Block | States | Passages/state | Response schema | Rail-end clearance |
|---|---:|---:|---|---:|
| `F40-S` | 305 | 50 | `physical8_v1` | 6 m |
| `F40-M` | 425 | 50 | `physical8_v1` | 6 m |
| `L99-S` | 475 | 50 | `physical8_v1` | 6 m |
| `L99-M` | 475 | 50 | `physical8_v1` | 6 m |

The clearance is bound to decision
`paper1-rail-domain-clearance-c06-v1`. The publishable output is exactly four
generation bundles plus two balanced training bundles. Historical datasets,
benchmarks, caches, studies, champions, results, and ZIPs do not qualify.

## Evidence required before report-only commit B

All entries below must bind the same clean commit A and be independently
revalidated before this report changes status.

| Evidence | Pending requirement |
|---|---|
| Source | clean 40-character commit A; final MATLAB-generator and Python-runtime source roots; complete integrated and mutation checks |
| Model form | final source-bound clearance, track-parameter, and fixed-Rayleigh evidence ledger |
| Capacity and benchmark | genuine `ttbi-paper1-compute-benchmark-v2` run from `benchmark_paper1_compute.py`, with its exact CUDA capacity receipt and zero forbidden trial states/events |
| MATLAB hosts | all four qualification stages on every intended host; complete pairwise receipt graph and aggregate inventory |
| Contact closure | independently verified 420-case four-stage contact/time-step result and external authorization receipt |
| Dispatch authorization | canonical create-once manifest from `dispatch_authorization.py create`, stored outside the repository and binding every item above |
| Independent review | confirmation that the evidence, source identities, A/B lineage, and exact six-bundle inventory agree |

Any source change after evidence collection reopens the source-sensitive gates.
An interrupted or partially resumed benchmark is diagnostic only. Qualification
host labels remain trusted-operator self-attestation; they do not prove physical
machine identity.

## Report-only B procedure

After every evidence row passes:

1. Keep the tested commit A unchanged and clean.
2. Replace the blocked status on line 3 with the builder's exact authorized
   status string: `**Status: PAPER-1 DISPATCH AUTHORIZED.**`
3. Replace `PENDING` on line 5 with the exact lowercase 40-character commit A
   enclosed in backticks.
4. Replace `PENDING` on line 6 with the exact lowercase 64-character SHA-256 of
   the external dispatch-authorization manifest enclosed in backticks.
5. Record the independently reviewed evidence below this section without adding
   another status, tested-source, or authorization-manifest header.
6. Create commit B with this file as the only A-to-B change. The benchmark and
   all qualifying evidence must have run against A, not B.
7. Invoke `build_stage_bundles.py` with the absolute external authorization
   manifest. The builder must mechanically revalidate the manifest and evidence;
   this report alone is never sufficient.

## Reviewed evidence record for commit B

- Clean tested commit A: **PENDING**
- MATLAB generator source root: **PENDING**
- Python runtime source root: **PENDING**
- Model-form evidence root: **PENDING**
- Paper 1 benchmark evidence root: **PENDING**
- CUDA capacity receipt SHA-256: **PENDING**
- Qualification inventory SHA-256: **PENDING**
- Contact-closure result/receipt SHA-256: **PENDING**
- External dispatch-authorization manifest path and SHA-256: **PENDING**
- Independent reviewer and review date: **PENDING**

Until those fields are completed and the fixed header is authorized, the exact
six bundles must not be built or dispatched.

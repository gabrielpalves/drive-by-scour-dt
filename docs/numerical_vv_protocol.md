# Numerical verification and validation protocol

**Status:** prospective execution protocol, 2026-08-03. This document does
not claim that the work below has already been completed. It defines the
evidence that must exist before numerical TTBI results are described as
verified, and the additional evidence that would be needed before a physical
model-validation claim is made.

**Scope:** the modified TTB-2D MATLAB solver, its bridge/track/wheel damage
mechanisms, and the response quantities used by the SHM study. This protocol
does not qualify the ML model, prove field transfer, or replace the frozen
campaign dispatch gates.

The implemented mechanisms, parameters, and first affected functions are
catalogued in [damage_model_reference.md](damage_model_reference.md). The
current reviewer-facing work plan is
[shm_reviewer_readiness_plan.md](shm_reviewer_readiness_plan.md). Those files
define what is being studied; this file defines how its numerical
implementation and solution are to be challenged.

## 1. Claim boundary

These terms are not interchangeable.

| Evidence class | Question answered | Evidence in this protocol | What it does **not** establish |
|---|---|---|---|
| **Code verification** | Does the code implement the intended equations and mappings correctly? | Independent element/assembly calculations, boundary-condition fixtures, exact mechanism deltas, conservation/equilibrium checks, and comparison with an immutable upstream-code oracle. | Accuracy of the equations for a real damaged railway bridge. |
| **Numerical solution verification** | Is discretization/integration error acceptably small for those equations? | Spatial-mesh and time-step refinement, waveform convergence, modal matching, generalized Richardson/GCI estimates where applicable, and contact-admissibility checks. | Physical validity of bilateral contact, damage laws, or simulated operational variability. |
| **Model validation** | Do predictions agree with independent physical observations at the quantity and scope claimed? | Requires a separately specified experimental or field benchmark with measured inputs, uncertainty, calibration/validation separation, and held-out response quantities. No internal comparison in this document substitutes for it. | Field performance outside the validation domain. |
| **Future field validation** | Does the complete method work on instrumented in-service train/track/bridge passages and known support/damage states? | Future held-out field campaign, with independently measured train, speed, temperature, geometry, track condition, and reference bridge/damage information. | It is not complete in the present repository. |

Calibration is parameter selection, not validation. Reproducing upstream code
is code/lineage verification, not model validation. Matching a qualitative
literature trend is a plausibility check unless independent measured data and
its uncertainty are actually compared at matching scope. Synthetic ML test
accuracy is not TTBI model validation.

## 2. Evidence already present, and its correct interpretation

1. [`smoke_damage_toggles.m`](../scour_MATLAB/smoke_damage_toggles.m)
   exercises bridge scour, bearing, and crack toggles and selected modal and
   damping contracts. [`smoke_damage_mechanism_contracts.m`](../scour_MATLAB/smoke_damage_mechanism_contracts.m)
   exercises track-matrix and wheel-path contracts.
   [`smoke_b54_overlap_parity.m`](../scour_MATLAB/smoke_b54_overlap_parity.m)
   and [`check_b54_overlap_parity.py`](../check_b54_overlap_parity.py) check
   overlap behavior. These are regression and mechanism-contract evidence;
   they are not physical validation.

2. [`execute_generation_state.m`](../scour_MATLAB/+ttbi/execute_generation_state.m)
   and [`validate_state_metadata.m`](../scour_MATLAB/+ttbi/validate_state_metadata.m)
   enforce a broad 0.2--15 Hz first-mode gate and narrower target-healthy
   geometry diagnostics (3--6 Hz for the short bridge and 2--4 Hz for the
   long bridge). These catch gross mass/stiffness regressions. They are not
   experimental modal validation and shall never be labelled as such.

3. [`contact_closure_study.m`](../scour_MATLAB/contact_closure_study.m),
   [`contact_closure_gate.m`](../scour_MATLAB/contact_closure_gate.m), and
   [`contact_gate_accept_report.m`](../scour_MATLAB/contact_gate_accept_report.m)
   implement a substantial time-step and bilateral-contact admissibility
   study. Its current frozen criteria come from
   [`contact_gate_policy_definition.m`](../scour_MATLAB/contact_gate_policy_definition.m).
   This is numerical solution/admissibility evidence for the implemented
   bilateral model. It is not validation of physical wheel separation or
   re-contact.

4. [`README.md`](../README.md) and
   [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) identify the exact
   TTB-2D upstream commit and local import lineage. Source hashes and
   [`check_source_provenance.py`](../check_source_provenance.py) support
   provenance/reproducibility. Provenance alone is neither numerical
   verification nor model validation.

The independent assembly/BC oracle is now implemented. The remaining work is
controlled coupled spatial-refinement evidence, added bridge-output
time-refinement evidence, a numerical reproduction of the held upstream
example, and complete paired one-mechanism response studies.

## 3. Rules common to every V&V run

Every run shall be made from a clean, hash-identified source commit. It shall
record MATLAB version, architecture, operating system, relevant toolbox
versions, host identity, exact invoked command, wall-clock start/end, and
SHA-256 hashes of every input and output. A run with a source or input mismatch
is invalid, not merely a warning.

The case manifest shall store the realized values, not just requested values:
mesh spacing, node/support coordinates, actual `linspace` time interval,
sample count, train/sleeper/bridge geometry, speed, temperature, vehicle and
suspension draws, rail-profile realization and phase, wheel OOR descriptor,
and every damage descriptor.

For paired comparisons, clone a saved baseline descriptor and edit only the
named mechanism. Recreating a case from a different `StateUID` is insufficient
if doing so advances or changes any named random stream. The harness must
assert equality of all non-intervention descriptors before solving. The two
geometries use separate frozen baselines; no random draw is shared across
geometries unless the manifest says so explicitly.

No acceptance threshold may be raised after viewing a qualifying result. A
pilot used to choose a threshold must be marked non-qualifying, retained, and
followed by a source-locked threshold ledger and a fresh qualifying run.

## 4. Code verification

### 4.1 Existing commands

From MATLAB, with `<repo>` replaced by the absolute repository path:

```matlab
cd("<repo>/scour_MATLAB");
smoke_damage_toggles;
smoke_damage_mechanism_contracts;
smoke_b54_overlap_parity;
smoke_geometry;
smoke_stage3;
```

From PowerShell at repository root:

```powershell
python check_b54_overlap_parity.py
python check_source_provenance.py
```

These commands shall remain prerequisites, but none is to be reported as
physical validation.

### 4.2 Implemented independent element and assembly harness

[`vv_euler_bernoulli_reference.m`](../scour_MATLAB/vv_euler_bernoulli_reference.m)
derives element matrices by exact integration of cubic Hermite polynomials and
assembles them with Boolean transformations; it does **not** call or copy
[`B03_BeamMatrices.m`](../scour_MATLAB/B03_BeamMatrices.m).
[`smoke_structural_oracle.m`](../scour_MATLAB/smoke_structural_oracle.m)
currently performs the following checks.

1. For at least two non-round values of `E`, `I`, `rho`, `A`, and element
   length, independently evaluate the closed-form two-node Euler--Bernoulli
   consistent mass and stiffness matrices used in `B03_BeamMatrices.m`.
   Compare every entry, symmetry, units/scaling under doubled length, the two
   element rigid-body modes, and strain/kinetic energy for fixed nontrivial
   trial vectors.

2. Hand-assemble two- and three-element dense reference beams. Check global
   DOF connectivity, shared-node accumulation, matrix symmetry, total
   translational mass, rigid-body nullity before supports, and positive
   definiteness of the free-DOF mass matrix. A reference assembled by copying
   the production loop is not independent.

3. Apply a static midspan point load to a uniform simply supported beam and
   compare reactions and midspan deflection with
   `R_A = R_B = P/2` and `delta_mid = P L^3/(48 E I)`. Compare bridge-only
   frequencies with
   `f_n = n^2*pi/(2*L^2)*sqrt(E*I/(rho*A))`. Report error versus mesh size;
   the finite-element result is not expected to be exact on a coarse mesh.

4. Check discrete static equilibrium (`sum reactions = applied load`) and
   virtual-work/energy equality independently. For a dynamic fixture, compute
   the residual `M*a + C*v + K*u - f` at saved steps and report its norm after
   scaling each term. The static checks are implemented; the coupled dynamic
   residual extractor remains missing because the necessary states are not yet
   retained at the required boundary.

### 4.3 Implemented boundary-condition oracle and remaining scope

The same smoke compares [`B02_BoundaryConditions.m`](../scour_MATLAB/B02_BoundaryConditions.m)
and `B03_BeamMatrices.m` with an independent free-DOF elimination oracle.

1. Use both production lengths (60.0 and 99.6 m), every supported geometry,
   and small synthetic two- and three-span fixtures. Assert unique support-node
   assignment and, for every positive-spring production support, coordinate
   error no larger than the source-locked cumulative-roundoff tolerance
   `max(256,2*n_bridge_elements)*eps(max(bridge_length_m,1))`. The historical
   nearest-node mapping may be characterized against the half-element bound
   only as an explicitly rejected diagnostic; it is not an admissible support
   tolerance. Also assert the intended constrained vertical/rotational DOFs
   and the expected number of rigid modes.

2. At scour losses 0, 0.30, and 0.60, assert that only the selected vertical
   support spring changes and that it equals `(1-d)*3.44e8 N/m`. Verify the
   undamaged, 30%, and 60% matrix differences entry by entry.

3. At zero and nonzero nominal bearing fixity, assert that rotational springs
   enter only the intended abutment rotational diagonals; intermediate
   supports remain unchanged. Check the descriptor-to-stiffness conversion
   before `B02_BoundaryConditions.m` as well as the final matrix entry.

4. The production code replaces fixed-DOF rows/columns with equal artificial
   diagonals in both `M` and `K`. Compare all retained elastic eigenpairs with
   a reference that deletes constrained DOFs exactly. Explicitly identify and
   exclude any artificial constraint eigenpairs. A retained-mode discrepancy
   is a verification failure, not a tolerance issue.

5. Match modes by the mass-normalized modal assurance criterion (MAC), not
   solely by sorted ordinal frequency. Recheck the two target Rayleigh damping
   ratios from the independently matched modes.

The executable oracle now covers element entries and scaling, two- and
three-element assembly, free/elastic/rigid BCs, 0/30/60% scour, abutment
rotational springs, static reactions/deflection/energy, exact-elimination
eigenpairs, mass-metric MAC, Rayleigh targets, and isolated B54 structural
embedding. It rejects seven plausible implementation mutations. This is strong
code-verification evidence, not physical validation, and it does not close the
missing coupled dynamic residual or immutable-upstream-array comparison.

It also quantified the production mesh correction for the three-span 60 m
bridge. Relative to the historical 0.30 m grid, which realized the internal
supports at 20.1 and 39.9 m, the support-aligned 0.20 m grid changed the first
five bridge frequencies by up to about 0.75% and increased the magnitude of a
fixed diagnostic point-load deflection by about 2.16%. The old snapping was
therefore not scientifically negligible merely because it was deterministic.

The adopted bridge \(E,I,\rho A\), and 3% damping set traces to the Fernandes
two-by-20 m example. Reusing that set for L60 and especially the four-by-24.9 m
L99.6 geometry is an explicit idealized geometry/scale stress transfer, not
calibration of those longer bridge configurations. The verification below can
show that the transferred mathematical model is solved consistently; it cannot
turn that property transfer into field validation.

## 5. Spatial discretization verification

### 5.1 Mesh levels and invariants

[`A04_Options.m`](../scour_MATLAB/A04_Options.m) now calls
[`bridge_mesh_elements_per_sleeper.m`](../scour_MATLAB/bridge_mesh_elements_per_sleeper.m)
to choose a support-aligned bridge density while retaining a two-element rail
mesh at production. [`B02_BoundaryConditions.m`](../scour_MATLAB/B02_BoundaryConditions.m)
rejects a positive-spring support that is not on a bridge node. The executable
geometry-specific refinement sequence is:

| Geometry | Level | Bridge elements/bay (`h`) | Rail elements/bay (`h`) | Role |
|---|---|---|---|---|
| L60, 3 spans | M0 | 3 (0.200 m) | 2 (0.300 m) | Current production |
| L60, 3 spans | M1 | 6 (0.100 m) | 4 (0.150 m) | Primary finer comparison |
| L60, 3 spans | M2 | 12 (0.050 m) | 8 (0.075 m) | Second primary finer comparison |
| L60, 3 spans | M3 | 24 (0.025 m) | 16 (0.0375 m) | Conditional resolution |
| L99.6, 4 spans | M0 | 2 (0.300 m) | 2 (0.300 m) | Current production |
| L99.6, 4 spans | M1 | 4 (0.150 m) | 4 (0.150 m) | Primary finer comparison |
| L99.6, 4 spans | M2 | 8 (0.075 m) | 8 (0.075 m) | Second primary finer comparison |
| L99.6, 4 spans | M3 | 16 (0.0375 m) | 16 (0.0375 m) | Conditional resolution |

The physical bridge length, support locations, sleeper spacing, vehicle,
damage coordinates, approach length, and requested time step must remain
identical. Record bridge and rail requested/realized spacing separately and
retain every support coordinate and floating-point offset. Reject a run if mesh
rounding changes physical bridge or support geometry. The former universal
0.600/0.300/0.150/0.075 m sequence is retained only as a rejected diagnostic:
it snaps at least one internal support for one or both geometries.

Run one-factor rail-only and bridge-only refinement as diagnostics if the
coupled result is sensitive, but qualify the production solution only from
the joint refinement above. Lumped sleeper/ballast DOFs and sleeper spacing
are not refined.

The executable on-bridge ballast-mass inventory is now mesh-invariant. Zhai et
al. (2004), Eq. (5) and Table 1, support a 531.4 kg discrete *independent*
ballast mass at each rail support point. B54's inherited bridge topology is
different: it has no on-bridge ballast DOF and condenses one retained 531.4 kg
value directly onto each deck DOF under an assigned sleeper. That deck
attachment is an inherited TTB-2D model-form choice, not a consequence of
Zhai's equation. Assigning full lumps to both bridge endpoints is a second,
author-chosen domain partition. Under those declared conventions, the total is
53,671.4 kg for L60 (101 points) and 88,743.8 kg for L99.6 (167 points),
invariant at every registered bridge mesh level. The coupled preflight isolates
the bridge mass block and rejects any missing, duplicated, off-node, or
mesh-density-scaled lump. It verifies the declared inventory, not the physical
validity of the condensation.

Topology is a separate P1 model-validity question. Zhai's complete ballast
model retains independent ballast motion and couples adjacent masses through
shear stiffness and damping \(K_w,C_w\). The paper reports a 12% ballast-
acceleration overprediction for its no-shear comparison and concludes that
the shear coupling is necessary for track-dynamics analysis. The repository
omits \(K_w,C_w\) on the approach, bridge, and exit. Before a physical-
validation claim, prospectively compare the retained condensed/no-shear model
with a source-faithful independent-ballast/shear arm, or provide and verify a
bridge-specific condensation that explains the omitted DOFs and shear branch.
An upstream reproduction establishes lineage only; it cannot close this
model-form transfer by itself.

A distinct parameter-scope question remains unresolved. Zhai's Table 1 labels
the track quantities *per rail seat*. The inherited planar property function
doubles rail mass/inertia and the reported half-sleeper mass, but leaves pad,
ballast, and sub-ballast mass/stiffness/damping values at the tabulated values.
That mixed convention must not be silently re-scaled: the intended one-seat
line model versus equivalent two-rail model requires an upstream benchmark or
a prospective two-arm sensitivity decision before physical validation claims.

Spacing is a second, independent source-scope transfer. Zhai's parameter set
uses 0.545 m rail-support spacing, while the generator uses 0.600 m and retains
the source's \(M_b\), \(K_b\), and \(K_f\) values without re-evaluating their
spacing-dependent expressions. The tabulated damping values are not claimed to
be spacing-derived; their open transfer is the per-rail-seat issue above. This is
a proxy-informed hybrid baseline, not a spacing-consistent reproduction of
Zhai. Qualification requires either an upstream benchmark that establishes the
intended convention or a prospective 0.545/0.600 m parameter-consistency
sensitivity; nominal values must not be recomputed post hoc after seeing
response results.

Mesh-level damping treatment must also be frozen before the coupled study.
The production path recomputes bridge Rayleigh \(\alpha,\beta\) from the first
two elastic modes on every mesh and then uses those bridge frequencies as the
rail reference frequencies. Thus refinement changes both \(K(h)\) and
\(C(h)\). The primary arm shall use recalibrated-per-grid coefficients because
that is current production behavior; a secondary arm shall hold the M0 healthy
coefficients fixed across grids. Every level must retain \(\alpha,\beta\), the
two reference frequencies, achieved modal damping ratios, and a hash/summary
of the assembled bridge and rail damping blocks. This choice may not be
switched after viewing convergence results.

### 5.2 Required cases

For each 60.0 and 99.6 m geometry, run:

1. the bridge-only healthy analytical/static/modal fixtures from Section 4;
2. a coupled healthy case at 70 km/h and 3 degC, 80 km/h and 18 degC, and
   90 km/h and 33 degC, using one frozen zero-deviation vehicle realization
   and a fixed rail-profile/OOR descriptor;
3. one separate coupled stress case for **each** active mechanism in Section 8,
   with all other descriptors equal to the healthy case; both registered dry-
   ballast sign arms; and one finite combined spatial stress case.

This third item is not executable until an exact finite descriptor table is
added to the source-locked case manifest. The phrase "maximum registered
level" is insufficient for Poisson counts, Bernoulli pad failures, damage
locations, crack width/localization, or polygon order/phase. Before any
qualifying run, freeze every row's complete realized descriptor: count,
coordinates/lattice indices, lengths, multipliers or severity, overlap
resolution, target support/element, wheel, polygon order/amplitude/phase, and
all unchanged EOV controls. The combined row must be chosen prospectively from
finite registered quantiles or explicit engineering values, not from the
worst outcome observed in a pilot. Until that table exists, the current
foundation remains nonqualifying by design.

These cases prevent a mesh that is adequate for scour from being assumed
adequate for a shorter track patch or wheel-path excitation without evidence.

The mesh study shall use the finest accepted time step from Section 6 so that
time error cannot masquerade as mesh error.

The crack implementation needs special treatment. A `lc <= 0` crack selects a
single element, so blindly refining the mesh changes the physical width of
the damaged region. For a spatial-convergence experiment, map a frozen
physical damage interval and integrated EI loss onto every mesh. If the
production scientific model intentionally remains a one-element crack, run a
separate localization/mesh sensitivity and report that dependency as a
model-definition limitation; do not disguise it as ordinary discretization
convergence.

### 5.3 Mesh QoIs

At every level save and compare:

- first five bridge elastic frequencies and mass-normalized MAC for matched
  modes;
- static midspan deflection, both support reactions, force balance, and strain
  energy;
- bridge midspan displacement and acceleration from
  [`B49_BeamDeformation.m`](../scour_MATLAB/B49_BeamDeformation.m) and
  [`B53_BeamAcceleration.m`](../scour_MATLAB/B53_BeamAcceleration.m);
- maximum absolute bending moment and shear, their coordinates/times, and
  bridge-entry/exit values from
  [`B31_BeamBM.m`](../scour_MATLAB/B31_BeamBM.m) and
  [`B33_BeamShear.m`](../scour_MATLAB/B33_BeamShear.m);
- all four signed wheel contact-force histories, signed peak, positive
  tensile-demand peak, on-track tensile fraction, and peak location;
- RMS, absolute peak, normalized waveform error, and correlation for the eight
  vehicle channels already used by `contact_closure_study.m`;
- fixed-parameter PSD band energy for bridge-related responses over the
  declared 0.2--15 Hz analysis band, and response at the analytically predicted
  polygon forcing harmonic when polygonization is active. Store the PSD
  estimator, window, overlap, sample rate, and binning; none may change by
  mesh level.

Interpolate time histories onto the same 0.01 m distance grid and common fixed
window used by `contact_closure_study.m`. Never compare arrays merely by sample
index when actual time grids differ.

### 5.4 Mesh convergence metrics

For scalar `Q`, report consecutive relative change

`e_21 = abs(Q_h/2 - Q_h) / max(abs(Q_h/2), Q_floor)`,

where `Q_floor` is fixed in the tolerance ledger before qualification. With
three monotone, mode-consistent grids, report observed order, Richardson
extrapolate, and fine-grid GCI using the actual spacing ratios. For a ratio of
two, the conventional calculations are

`p = log(abs((Q_h-Q_h/2)/(Q_h/2-Q_h/4)))/log(2)`

and

`GCI_fine = 1.25*abs(Q_h/2-Q_h/4)/(abs(Q_h/4)*(2^p-1))`.

The factor 1.25 is an explicit author-chosen convention aligned with the
current contact policy, not attributed here to a held source. The observed
order ceiling is source-locked at `p <= 10`; callers may tighten but cannot
raise it. GCI is invalid when the differences change sign, the denominator is
numerically zero, modes switch, or `p` is non-finite, nonpositive, or greater
than 10. In those cases run M3, show all levels, and use a conservative
finest-pair bound without calling it GCI.

For a waveform `y`, report reference-RMS-normalized RMSE,
reference-peak-normalized maximum error, correlation, peak amplitude error,
and peak-position error. A single scalar peak cannot replace waveform
convergence.

The repository now contains a nonqualifying foundation harness:
`numerical_vv_protocol_definition`, `numerical_vv_support_alignment`,
`numerical_vv_bridge_fixture`, scalar/waveform metric helpers, a coupled-mesh
preflight, and `numerical_vv_micro_run`. It records geometry-specific bridge
and rail fields, retained descriptor preimages and hashes, actual B11 time
grids, and hashed flat artifacts. It does not execute the coupled refinement
study and cannot authorize a scientific qualification claim.

Both MATLAB and Python package validators deliberately reject every
qualification request and every run kind other than `nonqualifying_micro`.
This fail-closed restriction was added after adversarial testing showed that a
self-authored placeholder package could otherwise declare itself passing. A
future qualifying verifier must independently recompute scientific evidence
and bind the complete source/case inventory; the current integrity receipt is
not that verifier.

## 6. Time-step refinement and bilateral-contact admissibility

### 6.1 Existing contact-closure workflow

[`B11_TimeSpaceDiscretization.m`](../scour_MATLAB/B11_TimeSpaceDiscretization.m)
derives a requested resolution through `max_accurate_frq`, then realizes an
actual interval after integer interval counting and `linspace`. Therefore all
convergence calculations must use the recorded actual interval, never the
requested label alone.

After fresh source-locked `F40-S`, `F40-M`, `L99-S`, and `L99-M`
qualification micros exist on the designated reference host, the exact MATLAB
command is:

```matlab
contact_closure_gate( ...
    "<absolute-F40-S-micro>", ...
    "<absolute-F40-M-micro>", ...
    "<absolute-L99-S-micro>", ...
    "<absolute-L99-M-micro>", ...
    "<absolute-new-closure-output>", ...
    "SourceCommit", "<commit-A-SHA>");
```

Then, from repository root:

```powershell
python check_contact_closure_gate.py "<absolute-closure-output>" `
  --source-commit <commit-A-SHA> `
  --receipt "<absolute-new-contact-closure-receipt.json>"
```

This is the workflow specified in
[`README_CAMPAIGN.md`](../README_CAMPAIGN.md). It evaluates all 420 frozen cases
at requested 1.0, 0.5, and 0.25 ms and authenticates their actual intervals.
For a non-qualifying exploratory case, the existing lower-level command is:

```matlab
contact_closure_study("<absolute-dataset>", state_index, passage_index, ...
    "DtMs", [1 0.5 0.25], ...
    "OutputDir", "<absolute-new-output>");
```

The eight `physical8_v1` response channels are carbody vertical acceleration,
both bogie vertical accelerations, the first two idealized constrained-wheelset
vertical-acceleration proxies from `AcelWheelsetPrimVag`, carbody pitch rate,
and both bogie pitch rates. The wheelset proxies are model-predicted responses,
not claims of measured axle-box channels. The four Eulerian `acc_under` values
in `AcelRodaPrimVag` remain mandatory solver diagnostics and are excluded from
the deployed response schema. Current comparison metrics are
reference-RMS-normalized RMSE, reference-peak-normalized maximum error,
correlation, RMS, and absolute peak. Contact metrics are signed peak, positive
tensile-demand peak, tensile path fraction, classification stability, and
convergence/GCI information.

The frozen contact policy uses coarse-to-fine waveform gates of
`0.05/0.10/0.995` (NRMSE/normalized maximum error/correlation), medium-to-fine
gates of `0.02/0.05/0.999`, and final 24 kN positive tensile-demand peak and
0.002 on-track tensile-fraction limits. These are prospectively locked,
author-chosen engineering closure criteria. They are not literature-derived
validation thresholds and must not be raised after observing a run.

### 6.2 Required extension for solution verification

The existing closure study is strong for vehicle waveforms and contact, but it
does not by itself qualify all bridge QoIs used in the scientific analysis.
A future test-only extractor/comparator must add, for the same exact runs:

- bridge midspan displacement and acceleration histories;
- maximum absolute bridge bending moment and shear and their time/location;
- each of the four signed contact-force histories, rather than only aggregate
  contact summaries;
- modal-coordinate or full-state dynamic equilibrium residual where saved
  states permit it;
- fixed-parameter 0.2--15 Hz bridge-response PSD energy and polygon-harmonic
  response for relevant cases.

Apply the scalar and waveform metrics from Section 5.4. If any three-level
sequence is non-monotone, switches event/classification, or gives invalid
observed order, add a requested 0.125 ms level for that case and report all
four actual intervals. Do not discard a coarse level to manufacture monotonic
convergence.

[`B66_ContactForce.m`](../scour_MATLAB/B66_ContactForce.m) reconstructs force
for a bilateral direct-contact solver. A positive tensile-demand artifact is
not proof that a wheel physically separated; the solver cannot simulate
separation/re-contact. Passing this section supports only bounded numerical
tension and solution stability for the implemented bilateral equations.

### 6.3 Finite rail-domain convergence

The current production configuration adds ten sleeper bays beyond the nominal
track construction, i.e. only 6 m at 0.600 m spacing, and uses no absorbing
rail boundary. Zhai et al. report good numerical convergence when the moving
wheelset remains at least 15 m from a finite rail-beam end. That statement is
not automatic validation of this different coupled bridge model, but it makes
the present 6 m boundary margin a required numerical sensitivity.

Run source-locked arms with realized minimum wheel-to-end clearances of 6, 15,
and 30 m while holding the bridge, sleeper spacing, vehicle, damage/EOV
descriptors, time grid, and saved bridge-distance window fixed. Record the
actual clearance of every wheel at the solve start/end; do not infer it only
from a requested sleeper count. Compare all Section 5.3 bridge/contact QoIs and
the ten full solver-response channels on their common retained window. Treat
15 versus 30 m as the primary convergence comparison; 6 m is the production
diagnostic. If the retained-window result fails the frozen waveform/scalar
criteria, enlarge the production domain or narrow the admissible claim. Do not
hide an initial boundary transient by cropping it without showing that it has
decayed before the first reported bridge event.

The executable clearance gate is now frozen in
[`rail_domain_clearance_study.m`](../scour_MATLAB/rail_domain_clearance_study.m)
and the `rail_domain_clearance` block of
[`numerical_vv_protocol_definition.m`](../scour_MATLAB/numerical_vv_protocol_definition.m).
It is deliberately narrower than the still-incomplete general coupled V&V
matrix:

- geometries are support-aligned F40 (40.0 m, two spans, 0.20 m deck / 0.30 m
  rail elements) and L99.6 (99.6 m, four spans, 0.30 m deck / rail elements);
- healthy operating points are V70/T3, V80/T18, and V90/T33, with the fixed
  five-vehicle nominal property draw, giving 18 coupled solves across the
  three clearances;
- excitation is one fixed FRA-v2 class-4, seed-20260728 realization synthesized
  on the common comparison coordinate [-30, 390] m. Each expanded-domain arm
  evaluates that realization at `x_global - domain_translation`, so FFT length
  and finite-domain origin cannot change the retained excitation;
- the retained coordinate is leading-wheel travel from solve start, on the
  exact D01-equivalent 0.01 m grid from 10.00 m through
  `10.00 + L_bridge + 18.30` m. Expanding the domain translates the deck and
  wheel path together, preserving lead-wheel-to-deck travel, solve duration,
  time grid, and crop. The omitted option remains the inherited ten-bay 6 m
  default;
- every vehicle/wheel start and end distance is saved. The requested value is
  an exact sleeper-lattice rail padding and a guaranteed minimum per-wheel
  clearance. The exit leading wheel realizes it exactly; the start-controlling
  wheel retains the source-locked 0.50 m positive surplus from rounding the
  nominal 106.3 m train length up to 106.8 m (one sleeper-lattice endpoint),
  and that realized surplus is reported rather than relabelled;
- waveform evidence comprises all ten legacy responses, all four idealized
  constrained-wheelset accelerations, bridge midspan displacement and
  acceleration, and four signed contact-force histories. Bridge/modal/extreme,
  event, and contact scalar QoIs are also saved and compared.

The two registered comparisons reuse the already-frozen contact waveform
criteria. For 6 versus 30 m (coarse diagnostic), every case/channel must satisfy
NRMSE <= 0.05, normalized maximum error <= 0.10, and correlation >= 0.995. For
15 versus 30 m (primary), the limits are 0.02, 0.05, and 0.999. Common-profile,
geometry/time/crop, per-wheel-clearance, and bilateral-contact-admissibility
checks are conjunctive. Select 6 m only if **both** comparisons pass over all
six geometry/operating groups; otherwise select 15 m if the primary comparison
passes; otherwise production remains **unresolved** and 30 m is a diagnostic,
not an automatically authorized production value.

The study writes only to a new output directory, hashes the complete reviewed
MATLAB source root and every flat artifact, binds the manifest from a completion
marker, and is checked independently by
[`check_rail_domain_clearance_study.py`](../check_rail_domain_clearance_study.py).
[`smoke_rail_domain_clearance.m`](../scour_MATLAB/smoke_rail_domain_clearance.m)
runs one F40/V70/T3/C15 solve and must remain `UNVERIFIED`; it cannot exercise
or author the selection verdict. Even a complete passing clearance package is
limited to this finite-domain model-form decision. It does not weaken the
existing refusal to claim general numerical qualification or physical
validation from `numerical_vv_micro_run`.

## 7. Immutable upstream TTB-2D reproduction target

Repository provenance fixes the upstream as
`ElsevierSoftwareX/SOFTX-D-22-00221` commit
`28d35528ac6624200a881bcd6130382b81579a01` (2022-10-11), locally imported at
commit `4530bf1238b45d442da5071b8d02559913164dab`. At import, 41 common files were
byte-identical and 16 were already adapted. Later repository-local changes
must not be attributed to the upstream author.

The repository-held article `papers/Cantero_2D_TTBI.pdf` (not redistributed in
dispatch ZIPs) describes the upstream illustrative case: one Manchester
Benchmark vehicle at 120 km/h, track
parameters attributed there to Zhai, a 30 m approach, FRA Class 6 profile, a
50 m simply supported ballasted bridge attributed there to Xia, interaction
enabled, and a 1 ms time step. It displays wheel contact forces, vertical
accelerations of all vehicle DOFs, bridge midspan acceleration, and bridge
bending moment in space and time. The paper does not provide a complete
machine-readable numerical oracle or an acceptance tolerance, so neither shall
be invented from the figures.

The required benchmark has two stages:

1. Check out the exact upstream commit in an immutable, read-only location.
   Run its original illustrative example without local source substitution.
   Export the complete input workspace and raw numerical results, not just
   screenshots. Record repository tree hash, MATLAB environment, exact example
   file hash, and hashes of all exported arrays.
2. Through a test-only compatibility wrapper, run the current
   `scour_MATLAB` solver with the exact upstream inputs and with every local
   damage, temperature law, stochastic vehicle perturbation, extra profile
   phase, and wheel OOR feature disabled unless present in the upstream case.
   Compare actual time/space grids; four wheel-contact histories; every
   available vehicle DOF history (including the vertical-response outputs
   displayed in the article); bridge midspan displacement/acceleration; and
   the bending-moment field, maximum, time, and coordinate.

Use numerical arrays as the oracle. Reproducing the article's plotted shape is
a secondary visual check only. Set comparison tolerances from (a) repeated
same-source upstream executions on the qualification environment and (b)
condition-aware floating-point/error propagation through the compatibility
mapping. Freeze those tolerances before running the modified solver. If a
mapping intentionally changes an equation, document and isolate it rather
than widening the tolerance.

This benchmark tooling is currently missing. Specifically, there is no
repository command that provisions and verifies the immutable upstream
checkout, no raw-oracle exporter, no complete Manchester/50 m compatibility
mapper, no array-level comparator, and no signed manifest/receipt joining the
two runs. Those five pieces must be added as test/V&V tooling before upstream
reproduction can be claimed. The upstream article reports its own comparison
with an independently developed model, but that statement is not inherited as
validation of this modified code or its added damage mechanisms.

## 8. Paired mechanism-trend study

The primary purpose is to verify that each descriptor changes only its
intended equations and to determine the actual response trend without
presupposing one. For each geometry, clone the healthy state's complete set of
50 registered passage descriptors. At every severity, use the same 50 speed,
temperature, vehicle, suspension, rail-profile, phase, and wheel-OOR draws.
These are common random numbers (CRN). Assert equality of all non-intervention
fields and report each paired difference; passages remain correlated repeated
observations and are not relabelled as 50 independent specimens.

Run one mechanism at a time at baseline, an interior registered level, and the
maximum registered level. If a mechanism has only two registered states, use
those two rather than inventing a midpoint.

| Mechanism | First code-level quantity that must change | Analytically required check | Dynamic/modal interpretation |
|---|---|---|---|
| Scour support loss | Selected bridge vertical support spring in `B02_BoundaryConditions.m`, then global `K` in `B03_BeamMatrices.m` | Entry equals `(1-d)*3.44e8 N/m`; all nonselected support entries and `M` are unchanged. For fixed `M`, matched bridge eigenfrequencies cannot increase under this positive-semidefinite stiffness removal. | Record, do not assume, vehicle/bridge/contact amplitude trends. |
| Nominal bearing fixity | Abutment rotational spring(s), then global bridge `K` | Only intended abutment rotational diagonals increase; intermediate supports and `M` are unchanged. Matched bridge eigenfrequencies cannot decrease when a nonnegative rotational spring is added. | No universal acceleration/RMS direction is asserted. |
| Local crack | Selected element `I/EI` before global assembly | Physical location/support are fixed; intended element stiffness decreases by the registered factor and no mass entry changes. Matched bridge frequencies cannot increase for the same physical damaged interval. | Single-element localization dependence is reported separately as required in Section 5. |
| Ballast patch | Sleeper--ballast blocks of track `K` and `C` in `B54_ModelMatrices.m` | Changed indices, patch coordinates, stiffness multiplier, and damping multiplier match the descriptor exactly; all other blocks are identical. | Retained softening or stiffening labels are compared separately. No blanket monotone response claim is made. |
| Unsupported sleepers | Selected sleeper--ballast `K` and `C` blocks in `B54_ModelMatrices.m` | Exactly the selected contiguous sleepers are multiplied by `1e-6`; neighbors and unrelated blocks are unchanged. | This is a linear near-zero-support approximation, not gap/contact validation. Dynamic direction is measured. |
| Rail-pad service change | All rail--sleeper `K` and `C` blocks in `B54_ModelMatrices.m` | The state-global service multipliers apply to every pad, with identical index coverage and the frozen stiffness/damping magnitudes. | Stiffness and damping effects are reported separately; no universal response direction is imposed. |
| Rail-pad failure | Selected rail--sleeper `K` and `C` blocks in `B54_ModelMatrices.m` | Only intended failed-pad blocks reduce by the frozen factors. | Compare with service variability as a distinct mechanism, not a severity continuum unless the descriptor explicitly defines one. |
| Wheel polygonization | Added wheel path `h`, `hd`, and `hdd` in [`B25_WheelProfiles.m`](../scour_MATLAB/B25_WheelProfiles.m), then the coupled solve in [`B65_DynamicCalcCoupledFaster.m`](../scour_MATLAB/B65_DynamicCalcCoupledFaster.m) and contact reconstruction | With order and phase fixed, added path amplitude scales linearly with registered amplitude; derivative frequency/amplitude and forcing harmonic follow speed, wheel radius, and order. Track/bridge matrices remain identical. | Contact and response peaks need not be monotone because phase and resonance matter. Check bilateral-contact admissibility. |

For scour, bearing fixity, and crack, the descriptor's first direct change is
to (K), not (M). The solved counterfactual is nevertheless not a fixed-(C)
experiment. `B24_BeamDamping.m` recalibrates Rayleigh coefficients from each
state's first two elastic bridge modes, and B00 then uses those bridge
frequencies as the rail damping references. The constant-target-damping
closure therefore changes assembled bridge and rail/model (C) globally when
structural (K) changes. Report these (C) deltas explicitly and run a
fixed-healthy-Rayleigh sensitivity before interpreting a dynamic response as a
pure stiffness effect.

The bridge target is 3%, while the separate rail target
`Track.Rail.Damping.per = 0.1%` is not present in Zhai et al. (2004). Treat
the rail value as an inherited author-chosen modeling choice. A qualifying
dynamic study must retain it prospectively and include it in the damping
sensitivity ledger; it is not source validation.

FRA rail-profile phase, speed, temperature, per-vehicle mass, and suspension
draws are EOV controls, not damage classes. They remain fixed within every CRN
pair. Wheel flats are dormant/unsupported in the active campaign and are not
silently introduced into this study.

For every pair save the exact matrix/path delta, first five matched modes and
MAC, the ten vehicle waveform metrics, four wheel contact histories, bridge
midspan displacement/acceleration, maximum moment/shear, and declared spectral
QoIs. Report individual paired differences plus median, range, and rank/order
summaries. Do not add inferential confidence intervals unless an independent
sampling/estimand protocol is separately justified.

A directional mechanism statement is permitted only when the matrix/path
direction is exact by construction or the paired response direction remains
unchanged after its numerical uncertainty interval is included. Otherwise the
result is reported as non-monotone or unresolved, not forced into the expected
SHM narrative.

The exploratory `response_signature_*` harness now provides a nine-intervention
catalog, field-by-field exogenous-control checks, exact matrix/path deltas,
mass-metric mode matching, waveform/PSD/contact metrics, and a paired solver.
One canonical V80/T18 healthy-versus-30%-scour passage has run successfully at
the aligned L60 production mesh. It changed one bridge/model stiffness entry,
left mass unchanged, changed bridge/model damping broadly through the
state-specific Rayleigh closure, lowered all five matched frequencies, and
remained within the registered contact-admissibility limits. This is evidence
for that one exploratory pair only. The other eight interventions and the
complete registered 50-passage studies remain unexecuted and unqualified.

## 9. Tolerance and convergence decision policy

Every threshold must appear in `tolerance_rationale.csv` with: check/QoI,
numeric value and units, normalization and zero floor, evidence class, basis,
source or pilot artifact hash, date frozen, source commit, and whether it is a
hard gate or reporting threshold.

Use four distinct bases:

1. **Algebraic/assembly checks:** derive a scale- and condition-aware floating
   tolerance from machine precision and the operation/reference conditioning.
   Store the calculation. Do not use a percent scientific tolerance to hide an
   incorrect index or BC.
2. **Existing contact closure:** retain the source-locked policy values in
   Section 6 and label them author-chosen engineering-admissibility criteria.
3. **Mesh/time solution error:** use actual-grid consecutive differences and a
   valid extrapolation/GCI where available. If `Q_f` has uncertainty `U_Q`, do
   not report digits finer than `U_Q`. A directional paired claim with
   `Delta = Q_damage-Q_base` requires an uncertainty bound
   `U_Delta = U_damage+U_base` that cannot include or cross zero in the claimed
   direction. A quantitative decision threshold must remain on the same side
   after both QoIs are expanded by their uncertainty bounds.
4. **Upstream reproduction:** derive the limit from immutable-oracle
   repeatability, floating-point propagation, and any explicitly isolated
   mapping transformation. The article's figure resolution is not a numerical
   tolerance.

There is deliberately no universal invented percentage for all SHM QoIs. A
QoI that lacks monotone/asymptotic refinement, a finite conservative error
bound, or a decision-robust interval is marked **unverified at the production
resolution**. The remedy is a finer grid, better mode/event matching, a
corrected implementation, or a narrower claim, not a post-result threshold
increase.

## 10. Required artifact package

Store qualifying output under an immutable external location or the ignored
`results/numerical_vv/<source-commit>/` tree. At minimum it shall contain:

- `manifest.json`: source/upstream commits, exact commands, input/output
  SHA-256 inventory, environment, and parent/child artifact graph;
- `case_table.csv` and `descriptor_hashes.csv`: requested and realized
  geometry, separate bridge/rail mesh counts and spacings, time step,
  operating/EOV values, named seeds/stream keys, retained canonical descriptor
  preimages and hashes, and damage descriptors;
- `element_assembly_checks.csv`, `bc_checks.csv`,
  `static_equilibrium_checks.csv`, and `modal_matching.csv`;
- `mesh_scalar_qoi.csv`, `mesh_waveform_metrics.csv`, and machine-readable raw
  histories needed to recompute them;
- `time_scalar_qoi.csv`, `time_waveform_metrics.csv`, contact metrics, current
  contact-closure report, and its independent Python receipt;
- `upstream_oracle_manifest.json`, raw upstream/current arrays, compatibility
  mapping, and array-comparison report;
- `mechanism_pair_manifest.csv`, exact matrix/path delta summaries, all paired
  response QoIs, and non-intervention equality checks;
- `tolerance_rationale.csv` and a final `vv_verdict.json` containing every
  required case, PASS/FAIL/UNVERIFIED status, failure reason, and the claims
  each result can and cannot support;
- plots generated only from the retained machine-readable tables/arrays,
  including mesh/time convergence, matched modal shapes, waveform overlays,
  contact histories, and paired mechanism trends.

Every table must carry case ID, geometry, mesh/time level, source commit, input
hash, and units. Missing or extra cases, overwritten artifacts, non-finite
values, coordinate mismatches, or a failed hash invalidate the package. A PDF
or screenshot without underlying arrays is never the sole evidence.

## 11. Completion and reporting rules

The numerical V&V package is complete only when all of the following are true:

1. existing regression, geometry, damage-contract, parity, and provenance
   prerequisites pass;
2. independent element, assembly, BC, static, eigenpair, damping, and dynamic
   residual checks pass;
3. the M0--M2 mesh sequence is qualified for every required QoI/case, or M3 is
   run and any remaining nonconvergence is explicitly marked unverified;
4. the complete existing 420-case contact gate passes and the added bridge
   time-step QoIs meet the decision policy;
5. the exact upstream illustrative example is reproduced by array-level
   comparison with a retained immutable oracle;
6. all active mechanisms pass exact intervention-isolation checks and their
   50-passage CRN studies are reported without unsupported monotonic claims;
7. the artifact inventory and independent receipts are complete and hashes
   resolve.

After those steps, the defensible wording is that the implementation has been
code-verified over the tested contracts and its reported numerical solutions
have quantified mesh/time uncertainty over the tested domain. That still does
not authorize the phrases "physically validated damage model," "validated
wheel separation," or "field-validated digital twin." Those require the
independent experimental and future field evidence defined in Section 1.

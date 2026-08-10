# Dry-ballast stiffness-sign sensitivity

Status: the matched-magnitude transform is retained and smoke-tested, but it is
**deferred from the current four Paper-1 blocks**. All four blocks disable track
EOV by configuration, and `A00_Run.m` rejects every nonempty
`TTBI_DRY_BALLAST_STIFFNESS_ARM` request. A future
`exp/track-train-damage` experiment must introduce a reviewed track-active
configuration before either arm can generate data. This is not a campaign rung
or a field-calibrated prior.

## Question and estimand

The retained dry-fouling scenario draws
`eta_k_base ~ U(1.2, 2.0)`, although the directly audited dry-sand experiment
indicates mild softening rather than stiffening. The sensitivity asks whether a
scientific conclusion depends on that disputed **stiffness direction**.

If the future track/train experiment runs both arms, do so only after the split
and reported metric are frozen. The primary
analysis uses the model and scaler selected and trained on the retained/default
campaign; freeze both before opening the reciprocal-arm outputs, and do not
retrain or refit preprocessing on reciprocal data. Pair records by `state_uid`
and passage index. The primary SHM estimand is the paired change in that frozen
model's state-level loss,

`Delta_i = L_i(reciprocal-softening) - L_i(retained-stiffening)`,

summarized over the same outer-test states, with uncertainty resampled at the
independent-state level. Report the all-state estimand and, secondarily, the
conditional estimand for states whose persisted `track_log` contains at least
one dry patch. A raw-response diagnostic may analogously report paired changes
in prespecified channel RMS or spectral features, but it must not replace the
frozen-model estimand after outcomes are inspected. A separately declared
retrained-per-arm analysis is secondary: it asks a different
achievable-performance question and must not be substituted for distributional
robustness of the retained/default model.

## Arms and exact pairing

- `retained-stiffening`: `eta_k = eta_k_base`, the unchanged campaign scenario.
- `reciprocal-softening`: `eta_k = 1 / eta_k_base`, hence
  `eta_k in [0.5, 0.8333...]` and
  `log(eta_k_soft) = -log(eta_k_base)`.

The reciprocal transform preserves absolute log-distance from the healthy
multiplier 1 while reversing its sign. It consumes no random draw. Therefore,
under the same arm-independent seed schedule, the two runs have exactly the
same patch count, wet/dry indicator, length and location, wet-patch stiffness,
damping, hanging-sleeper groups, pad condition/failures, cracks, profiles,
vehicles, speed, temperature, scour, and nominal bearing fixity. Only dry-patch
stiffness changes. The arm is persisted in every sensitivity `track_log`, in
`case_info.mat`, and in `generation_config_json`; it changes the generation
fingerprint. Reusing a persistent track state under the other arm is rejected.

## Current configuration boundary

There is deliberately no authorized A00 command for either arm in Paper 1.
Keep `TTBI_DRY_BALLAST_STIFFNESS_ARM` empty for all four production blocks. A
nonempty request fails before generation because `use_track_eov=false`.

The dormant implementation contract is checked without producing campaign data:

```powershell
matlab -batch "cd('C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/scour_MATLAB'); smoke_dry_ballast_sign_sensitivity"
```

Before future execution, the track/train branch must register an experiment ID,
a track-active configuration, complete matched arm manifests, and an
authenticated evaluator. The source-root hashes and all arm-independent
fingerprint inputs must then agree across arms.

## Outputs and acceptance checks

When a future track-active experiment is registered, the ordinary state payload
and channel names are retained and sensitivity outputs are isolated under:

```text
scour_MATLAB/Results_sensitivity/dry_ballast_stiffness_sign/
  retained-stiffening/<case_name>/
  reciprocal-softening/<case_name>/
```

Each arm directory contains the normal authenticated `case_info.mat`,
`damage_states.mat`, and numbered state files. Before calculating a response
or prediction contrast, verify for every paired state and passage:

1. equal `state_uid`, target values, named-stream seed IDs, operational draws,
   and all nuisance descriptors other than dry `eta_k`;
2. equal patch coordinates and `eta_c`, with wet `eta_k` exactly equal;
3. reciprocal dry values exactly satisfying
   `eta_k_soft * eta_k_retained = 1` within floating-point tolerance;
4. different generation fingerprints but equal generator source roots; and
5. explicit arm labels in `case_info` and each persisted `track_log`.

### Analysis-authorization status: deferred

The ordinary campaign loader/provenance gate is allowed to reject these files:
the arm is intentionally fingerprinted, so the reciprocal generation
fingerprint cannot equal the definitive campaign fingerprint. Do **not** remove,
weaken, monkey-patch, or manually bypass that gate. Before either arm is used in
a paper result, implement a dedicated sensitivity evaluator and immutable
receipt that:

- authenticates both completion markers, file-digest roots, source roots,
  numerical environments, and full `generation_config_json` records;
- proves that the arm field and its entailed dry `eta_k` values are the only
  allowed cross-arm differences, while all state identities, seeds, targets,
  and other latent descriptors are exactly paired;
- binds the already-frozen retained/default model, scaler, split, protocol
  hash, metric, and tested state inventory; and
- records the paired estimand, state-level uncertainty calculation, and output
  table digest.

Generation support and sampler contracts are implemented. This dedicated
comparison/evaluation tooling and receipt are still **pending**, so a full arm
run is not yet authorized for a manuscript claim even though the exact
generation commands above are executable.

Run the fast executable contract before dispatch:

```powershell
matlab -batch "cd('C:/Users/gabri/OneDrive/Desktop/Doutorado/Sandwich/DigitalTwins/drive-by-scour-dt/scour_MATLAB'); smoke_dry_ballast_sign_sensitivity"
```

## Interpretation boundary

The reciprocal arm is a deliberately symmetric perturbation on a logarithmic
multiplier scale. It is **not** an estimate of a dry-fouling population,
prevalence, posterior, or physically calibrated softening range. It does not
test the damping law, wet-fouling law, patch prevalence/length/location law, or
interactions outside this simulator. A small paired effect supports robustness
to this one sign choice within the modeled domain; a large effect means the
headline conclusion is assumption-sensitive and must be qualified. Neither
result validates either arm as field truth.

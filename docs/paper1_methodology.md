# Paper 1 — Methodology (draft prose)

Drafted 2026-07-09 from the implementation (`scour_MATLAB/`, `core/`, `training/`) and the
talk (`presentation/SHM_Markdown/main.md`). This is the §4 material for
`docs/paper1_outline.md`. **Numbers marked ⚠ need a final check against the code/config at
write-time** (units, sampling rate, exact ranges). Notation can be normalised when we adopt
the journal template.

---

## 4.1 Overview

The proposed drive-by scour-identification pipeline has four stages: (i) a physics-based
train–track–bridge interaction (TTBI) simulation generates vehicle acceleration responses
for a range of scour severities under environmental and operational variability; (ii) each
multi-channel response is standardised and compressed by Piecewise Aggregate Approximation
(PAA); (iii) a modular one-dimensional neural network maps the compressed signal to a scour
severity; and (iv) every architecture and sensor configuration is optimised by Bayesian
hyperparameter search and ranked by a repeated-seed robustness criterion. Stages (iii)–(iv)
are the object of the ablation: we vary one architectural block or one sensor at a time and
measure the effect on a fixed metric, isolating which design choices genuinely contribute.

*(Figure: pipeline schematic — physics → PAA → modular CNN → robustness ranking.)*

## 4.2 Train–track–bridge interaction model

Responses are simulated with the two-dimensional TTBI framework of Cantero (TTB-2D /
VEqMon2D) [cite Cantero 2022], which solves the fully coupled vehicle–track–bridge system in
the vertical plane by direct time integration of the coupled equations of motion.

**Bridge.** A ⚠ 40 m Euler–Bernoulli beam on three supports (two equal spans), with
modulus of elasticity E = 35 GPa, second moment of area I = 0.33 m⁴, mass per unit length
⚠ ρ (verify units), and 3% modal damping. Supports are modelled as vertical springs; the
central foundation carries the scour damage (§4.3).

**Vehicle.** A planar multi-body railway vehicle (car body, two bogies, wheelsets, with
primary and secondary suspension), using the calibrated properties of the TTB-2D
"O'Brien-calibrated" vehicle [cite]. The train comprises ⚠ five successive vehicles;
on-board accelerometers are placed on the **leading vehicle**, from which eight
degrees of freedom (DOFs) are recorded:

| # | DOF | Mass it sits on |
|---|---|---|
| 0 | Car-body vertical | suspended (secondary) |
| 1 | Front-bogie vertical | suspended (primary) |
| 2 | Rear-bogie vertical | suspended (primary) |
| 3 | Wheelset 1 vertical | unsuspended |
| 4 | Wheelset 2 vertical | unsuspended |
| 5 | Car-body pitch rate | suspended |
| 6 | Front-bogie pitch rate | suspended |
| 7 | Rear-bogie pitch rate | suspended |

The suspension acts as a mechanical filter: car-body channels are the most attenuated but
the most practical to instrument (power, protection, access); wheelset channels are the
least filtered but the noisiest. This trade-off is precisely what the sensor-economy
ablation (§5.2) quantifies.

**Track.** A layered ballasted-track model after Zhai et al. [cite Zhai 2004] — rail
(Euler–Bernoulli beam) on discrete rail-pad, sleeper, and ballast elements — couples the
wheelsets to the bridge and injects realistic high-frequency wheel–rail dynamics.

**Rail irregularity.** A single fixed measured longitudinal-level profile is used for every
passage. **Scope note (report honestly):** rail-profile irregularity is therefore *not* a
source of operational variability in this study; it is held constant so that the recorded
variability is attributable to the controlled factors of §4.4. Rail-profile degradation as
an environmental factor is addressed in the follow-up digital-twin work.

## 4.3 Scour damage model

Scour removes soil around the foundation, reducing its vertical and lateral support
stiffness and lowering the global natural frequencies [cite Prendergast & Gavin; Kamariotis
2024]. Consistent with the drive-by scour literature [cite Fernandes 2024, 2025], scour at
the central foundation is idealised as a reduction of the support's vertical stiffness,

  k_v(d) = (1 − d) · k_v0,  with healthy k_v0 = 3.44 × 10⁸ N/m,

where d ∈ [0, 60%] is the scour severity. The severity is discretised in 1% steps, yielding
**61 damage states** (0% healthy → 60%). This spans early, hard-to-detect damage (the
regime of practical interest) through severe loss of support.

## 4.4 Environmental and operational variability, and datasets

To prevent the network from learning a single idealised operating point, each of the 200
passages per damage state is simulated under randomised environmental and operational
variability (EOV):

- **Measurement noise** — additive signal noise (std ⚠ σ = 5% of the signal).
- **Temperature** — sampled in ⚠ [3, 33] °C, entering through a temperature-dependent
  modulus of elasticity E(T) (⚠ −0.3%/°C about 15 °C).
- **Vehicle-property variability** — per-vehicle randomisation of ⚠ three mechanical
  properties (mass/suspension), reflecting load and maintenance state.
- **Train speed** (second dataset only) — sampled per passage in ⚠ [70, 90] km/h.

Two datasets isolate the effect of speed variability, the hardest confounder:

- **D1** — noise + temperature + vehicle-property variability (constant nominal speed).
- **D2** — D1 **plus** per-passage train-speed variability.

Each dataset contains 61 states × 200 passages = ⚠ 12,200 labelled passages per channel.
Each passage is the leading-vehicle DOF time history over the bridge crossing (plus a short
approach/exit for free vibration), cropped to a fixed spatial window and ⚠ sampled at
~1 kHz.

## 4.5 Signal preprocessing

Each channel is standardised (zero mean, unit variance) using statistics fitted **only on
the training partition** (fixed 80/20 split, seed 42) to prevent information leakage, then
compressed by **Piecewise Aggregate Approximation (PAA)** to a fixed length of 512 segments:
the sequence is partitioned into equal windows and each is replaced by its mean. PAA acts as
a structural low-pass filter — it smooths the high-frequency, high-energy rail-corrugation
and wheel–rail noise while preserving the low-frequency deflection-basin / stiffness-loss
signature — and reduces the sequence length by roughly an order of magnitude, cutting
training cost [cite Fernandes 2025 for the PAA-as-EOV-filter precedent]. A continuous
wavelet transform (CWT, Morlet, ⚠ 64 scales) branch is included as a two-dimensional
preprocessing comparator.

## 4.6 The ablated architecture

All architectures share a modular one-dimensional backbone with three independently
toggle-able blocks, so a single implementation spans the ablation grid:

- **Convolutional backbone.** A stack of 1-D convolution–ReLU(–max-pool) layers extracts
  local spatio-temporal features from the (channels × 512) input; depth, width, and kernel
  sizes are hyperparameters (§4.7).
- **Space2Vec (optional).** A learnable spatial-position embedding (a linear plus a set of
  periodic components of the normalised along-bridge coordinate), concatenated to the input
  channels — grounding the vibration to physical position on the span.
- **LSTM (optional).** A recurrent block adding temporal context across the sequence.
- **N-HiTS multi-rate pooling (optional).** The physics-matched block: the feature sequence
  is sub-sampled at several rates and the views concatenated, so the network simultaneously
  represents fast, localised events and slow, global trends.

**The inductive-bias argument.** The drive-by signal is intrinsically two-timescale:
high-frequency transients from wheel–rail contact and track irregularity, superimposed on
the low-frequency modal sag induced by foundation stiffness loss — the component that
actually encodes scour. N-HiTS multi-rate pooling matches this structure by construction:
the fine pooling rate preserves the sharp wheel-impact features while the coarse rates
expose the global deflection trend, letting the classifier read scour from the slow
component without discarding the fast one. This is the paper's central claim, tested in §5.1.

**Head and target.** A dense head maps the pooled features to **61 ordinal outputs**. Rather
than treat the 61 states as unordered classes, we regress the (discretised) severity and
score with **mean squared error on the state index**, which preserves the physical damage
ordering: predicting 41% for a true 42% is nearly free, whereas 5% for a true 55% is heavily
penalised. This ordinal formulation is the natural bridge to the continuous multi-foundation
regression of the follow-up work.

## 4.7 Hyperparameter optimisation

For every architecture × sensor configuration, up to ⚠ 26 hyperparameters (learning rate,
weight decay, per-layer filter counts and kernel sizes, pooling flags, dense widths and
dropout, and — when active — LSTM size and N-HiTS pooling rates) are optimised with Optuna's
**multivariate Tree-structured Parzen Estimator (TPE)**. TPE natively handles the
*conditional* search space (blocks and their parameters appear or disappear with the
architecture flags), which fixed Gaussian-process surrogates do not. Each study runs ⚠ 200
trials with 25% random start-up and successive-halving pruning of unpromising trials; the
objective is the best validation MSE (§4.6) over ⚠ 50 epochs with early stopping.

## 4.8 Robustness-based model selection

A single Optuna champion is the *luckiest* trial, which over-states performance and hides
fragility. We therefore re-train each champion over **30 random seeds** and rank
configurations by the 95% upper confidence bound on the mean MSE,

  UCB₉₅ = MSĒ + 1.96 · σ_MSE / √30,

rewarding models that are *reliably* good rather than occasionally excellent. We additionally
report a **collapse-rate** — the fraction of the 30 seeds whose MSE exceeds a physical
tolerance (a model that has failed to learn the damage ordering) — which exposes
configurations, typically single weak sensors, that a point estimate would wrongly endorse
(§5.2). Median MSE with inter-quartile range and the collapse-rate are the reported summary
statistics.

---

### Verify-before-submission checklist
- ρ (bridge mass/length) value + units; sampling rate; signal crop window length.
- Exact EOV ranges: noise σ, temperature range and E(T) law, which/how-many vehicle
  properties, speed range.
- N vehicles in the train and which vehicle carries the sensors (code: `AcelPrimVag` =
  leading vehicle).
- Optuna: exact HP count, trials, epochs, pruner settings; CWT scale count.
- Confirm the "O'Brien-calibrated" vehicle citation and the Zhai track reference.

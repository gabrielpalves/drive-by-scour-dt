# Deep-research prompt — vehicle suspension damage as a drive-by EOV

> **HISTORICAL RESEARCH PROMPT.** Current R11 evaluates empirical robustness
> under the simulated nuisance distribution; it does not assume or claim
> mathematical invariance.

Paste the block below into Gemini Deep Research (or NotebookLM Deep Research).
Written 2026-07-21, motivated by a methodology review of the ablation ladder: the
train-side damage model currently covers wheel out-of-roundness only (polygonization
enabled; flats implemented but disabled pending a unilateral-contact solver). Suspension
faults are NOT modelled, yet they directly alter the suspension filtering that determines
which vehicle channels can see bridge damage — so they could plausibly shift the
sensor-placement ranking. Before adding a suspension-damage rung to the ladder
(planned: ONE isolated rung on the 3-span/60 m bridge), the fault modes, magnitudes,
prevalence, and draw policy must be literature-anchored. See
docs/framework_rationale.md and docs/track_eov_sampling_spec.md for the house style:
every randomized parameter needs a citation or a documented derivation from one.

---

**Topic: Realistic modelling and randomization of railway-vehicle suspension
degradation (dampers, springs, air springs) for simulation-based machine learning on
drive-by bridge-monitoring signals — fault modes, parameter magnitudes, fleet
prevalence, and their effect on car-body / bogie / axle-box vertical dynamics**

**Context.** We use a 2-D vertical coupled train–track–bridge interaction model
(TTB-2D, Cantero; Zhai-type track) to train neural networks that estimate bridge pier
scour (support-stiffness loss) and abutment bearing seizure from single-passage
drive-by signals. The vehicle is a 10-DOF 2-D model: car-body bounce + pitch, two
bogies bounce + pitch, wheel/axle vertical DOFs; sensors are the 8 vehicle channels
(car-body, bogie, wheel verticals + car-body/bogie pitch rates); speeds 70–90 km/h.
Healthy-fleet variability is already randomized per passage (vehicle mass and
suspension parameters within nominal ranges, speed, temperature). Wheel
out-of-roundness (polygonization) is modelled per passage; wheel flats are excluded
for solver reasons. We now want to add SUSPENSION FAULTS as a labelled-as-nuisance
EOV: the network must stay invariant to them, and we will measure whether they
corrupt the scour/bearing estimates and whether they change which sensors are
optimal. Faults belong to the TRAIN, so they travel with the vehicle (a different
fleet vehicle each passage), not with the bridge state.

**Research questions.**

1. **Fault taxonomy and field prevalence.** For in-service passenger/regional rolling
   stock: what are the documented failure modes of (i) primary vertical dampers,
   (ii) secondary vertical/lateral dampers, (iii) primary coil/rubber springs,
   (iv) secondary air springs (deflation, orifice blockage), (v) anti-yaw dampers
   (note: our model is 2-D vertical — flag which modes are unrepresentable)?
   What FRACTION of an in-service fleet typically runs with a degraded component at
   any given time (standing prevalence, not annual incidence — distinguish the two),
   and what is the typical latency between fault onset and detection/repair under
   time-based vs condition-based maintenance?

2. **Fault magnitudes used in simulation studies.** In the bogie/suspension
   fault-detection and vehicle-dynamics literature, what parameter changes represent
   each fault? Report the conventional grids (e.g., damper effectiveness reduced to
   75/50/25/0% of nominal; spring stiffness ±X%; air-spring pressure loss levels)
   with citations per number. Which magnitudes are considered "incipient" vs
   "failed", and which would trigger ride-quality alarms (EN 14363 / UIC 518
   acceptance limits) — i.e., where is the boundary between a fault that plausibly
   stays undetected in service (our nuisance regime) and one that would be caught
   immediately?

3. **Draw policy for domain randomization.** Given that suspension state travels
   with the VEHICLE (different train each passage) while our persistent nuisances
   (crack, profile, track) travel with the BRIDGE STATE: is a per-passage Bernoulli
   draw with fleet-prevalence probability the defensible policy (analogous to our
   per-passage wheel-OOR draw)? Any published precedent for how vehicle-condition
   heterogeneity across a fleet is randomized in simulation-trained drive-by or
   onboard-monitoring ML?

4. **Signal signatures.** For each fault type: which measurement locations
   (car-body, bogie frame, axle box) and frequency bands change, and by how much
   (RMS/PSD deltas)? Specifically: does a failed primary damper predominantly
   amplify bogie-frame response near the bogie bounce/pitch modes, and a failed
   secondary damper the car-body 1–3 Hz band? Quantitative simulation or measurement
   references preferred — these bands overlap the quasi-static scour deflection
   signature our sprung channels rely on, which is exactly why the interference
   question matters.

5. **Interference with infrastructure monitoring.** Any published evidence on
   vehicle-condition faults confounding drive-by / onboard INFRASTRUCTURE monitoring
   (bridge or track condition estimated from a vehicle whose own suspension is
   degraded)? Studies that jointly monitor vehicle and infrastructure, or that report
   robustness of track/bridge indicators to vehicle parameter drift, are directly on
   point. If literature is thin, say so explicitly — a documented gap is a
   contribution claim for us.

6. **Mapping to a 2-D 10-DOF model.** Best practice for representing these faults in
   a planar vertical model: multiplicative factors on per-bogie c and k (primary) and
   per-vehicle secondary c and k; front-vs-rear-bogie asymmetry (representable) vs
   left-right asymmetry and roll coupling (NOT representable in 2-D — how much does
   the literature say we lose, and is the planar simplification precedented for
   vertical-dynamics fault studies?).

**Deliverable.** A recommended randomization table: fault type → model parameter(s)
and multiplier range → standing prevalence → draw policy (per passage) → citation,
with every number either cited or explicitly flagged as an inference from a cited
anchor (state the derivation). Peer-reviewed sources and standards (EN 14363,
UIC 518, EN 13298, maintenance-practice literature) preferred; conference and
industry sources acceptable if flagged. Please distinguish numbers that are
simulation conventions (e.g., a 50%-damper grid chosen for convenience) from numbers
grounded in field/failure data.

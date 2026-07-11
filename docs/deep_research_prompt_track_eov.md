# Deep-research prompt — track-layer damage as EOV nuisances for drive-by scour detection

Paste the block below into Gemini Deep Research / NotebookLM Deep Research.
Written 2026-07-09; motivates a possible new ablation stage (track-layer EOV) —
see docs/framework_rationale.md §7 and the sensitivity-check-first plan.

---

**Topic: Track-layer damage (ballast, hanging sleepers, rail pads) in train–track–bridge
interaction models — representation, measured parameter ranges, spatial statistics, and
relevance as confounders for drive-by bridge-scour detection**

**Context.** We use a 2D vertical coupled train–track–bridge interaction (TTBI) model
(TTB-2D by D. Cantero, with a Zhai-type track: rail as a beam on discrete fastener/pad
springs, sleeper masses, ballast and sub-ballast spring–damper layers at 0.6 m sleeper
spacing; on the bridge, a ballast layer couples the track to a multi-span deck beam). We
train machine-learning models on vehicle-mounted (drive-by) sensors — bogie and car-body
vertical accelerations and pitch rates of the first vehicle — to estimate bridge pier
scour (modelled as a loss of vertical support stiffness at pier supports) and abutment
bearing damage (rotational stiffness). Deck cracks (local EI reduction, Sinha-type) and
rail-profile degradation (regenerated from FRA/ORE PSD classes) are ALREADY included as
randomized training nuisances (domain randomization / EOV augmentation). We now need to
decide whether ballast, sleeper, and rail-pad damage should also be added as randomized
training nuisances, and if so with what data-grounded parameters.

**Research questions.**

1. **Model representation.** What are the standard ways to represent, in 2D vertical
   Zhai-type TTBI models: (a) hanging / unsupported sleepers (gap nonlinearity vs
   zeroed/reduced ballast spring; typical group sizes), (b) ballast degradation and
   fouling (stiffness reduction, damping change, regional differential settlement),
   (c) rail-pad deterioration (does aging stiffen or soften pads, and by how much;
   missing/failed pads)? Canonical references (e.g. Lundqvist & Dahlberg; Zhai) plus the
   most recent work (2020–2026), including heavy-haul "interlayer damage" studies.

2. **Measured parameter ranges.** Field or laboratory data for: percentage of voided/
   hanging sleepers on operating lines and void depths; ballast stiffness loss due to
   fouling or settlement (in %); rail-pad stiffness change with age, load cycles, and
   temperature (new-vs-aged factors); typical defect/patch lengths. Any published
   statistical distributions of these quantities.

3. **Spatial statistics / dependence.** Evidence that these damages CLUSTER spatially:
   consecutive hanging-sleeper groups (how many in a group), ballast fouling patches and
   their correlation lengths, pad degradation near welds/joints/transitions. What is the
   recommended way to sample spatially correlated damage in track simulation studies —
   patch models, correlated random fields (what correlation length), Markov models?

4. **Frequency content and channel sensitivity.** Which frequency bands and which
   vehicle DOFs (unsprung wheel/axle-box vs bogie vs car body) do these local track
   damages excite? Is there evidence they can be confused with GLOBAL bridge-stiffness
   changes (pier support stiffness loss, bearing seizure) in drive-by / indirect
   monitoring, or are they spectrally/spatially separable?

5. **Confounder evidence in drive-by SHM.** Studies where track-layer damage or track
   quality degradation degrades indirect bridge-damage detection, and how they handled
   it (training-set augmentation, filtering, feature choice). Include any study that
   treats track faults as nuisance/EOV in a learning-based drive-by pipeline.

6. **Occurrence rates.** How common is each fault in practice (defects per km-year,
   % of sleepers voided on a typical line, pad replacement intervals) — to justify what
   a "realistic" nuisance distribution looks like in training data.

**Desired output.** A synthesis containing:
(i) a recommended modelling choice per damage type for a Zhai-type 2D vertical model;
(ii) a CONCRETE SAMPLING SPECIFICATION we can implement directly: number of damaged
pads / hanging-sleeper groups / ballast patches per km (or per 100 m), severity
multipliers with ranges and suggested probability distributions, location distributions,
and spatial-correlation parameters (group/patch lengths), each anchored to cited
measurements;
(iii) a short verdict on whether these damages plausibly CONFOUND pier-scour estimation
from bogie/car-body signals (as opposed to axle-box), with the key supporting references;
(iv) full citations for everything.

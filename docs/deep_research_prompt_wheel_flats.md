# Deep-research prompt — wheel flats / OOR occurrence & severity (Stage 3)

Written 2026-07-09. Purpose: replace the flagged ASSUMPTION parameters in the
Stage-3 wheel-OOR sampling (A00: oor_p_wheel=0.08, flat length U(20,60) mm) with
literature-anchored values BEFORE the Stage-3 dataset is generated.
Submit to NotebookLM Deep Research (same notebook as the track-layer research).

---

**Topic: Railway wheel flats and out-of-round (OOR) wheels — occurrence
statistics, severity distributions, and modelling for train–track–bridge
interaction simulation**

**Context.** We use a 2D vertical train–track–bridge interaction model
(Zhai-type track) for drive-by bridge monitoring research. We model a wheel
flat as a periodic haversine dip added to that wheel's wheel–rail irregularity
path: period = wheel circumference (R ≈ 0.46 m), depth d = L²/(16R) from the
flat length L. We need literature-anchored sampling parameters to randomize
wheel flats as training nuisances (domain randomization), not labels.

**Research questions.**
1. **Occurrence**: what fraction of in-service wheels carry a flat at any given
   time (wayside impact load detector / WILD statistics, fleet inspection
   campaigns)? Formation rates (per vehicle-year or per million km), seasonal
   effects (low-adhesion autumn braking), and how quickly flats are removed
   (reprofiling intervals / condemning practice).
2. **Severity**: measured distributions of flat LENGTH and DEPTH for newly
   formed vs rounded/run-in flats; regulatory/condemning limits (EN 15313,
   UIC, AAR — typical 30–60 mm length or impact-force limits); ranges used in
   published VTBI simulation studies.
3. **Modelling**: the standard flat representations (haversine / rounded flat;
   chord relation d = L²/16R); validity of injecting the flat as an equivalent
   rail irregularity moving with wheel rotation in a 2D vertical model; at
   what severity/speed wheel–rail contact loss occurs and matters.
4. **Polygonization / periodic OOR (harmonic orders 1–20)**: occurrence and
   amplitude ranges in service fleets; should it be modelled SEPARATELY from
   discrete flats for a passenger-train case at 70–90 km/h, or is it
   second-order for car-body/bogie responses?
5. **Vehicle response**: which frequency bands and which channels (axle-box vs
   bogie vs car-body) do flats excite at 70–90 km/h; the impact repetition
   frequency v/(2πR) ≈ 7–8 Hz and harmonics — any evidence of interference
   with LOW-frequency bridge-condition monitoring from on-board sensors.
6. **Within-train correlation**: do flats cluster within a bogie/vehicle
   (a braking/slide event flats several wheels simultaneously)? Any data on
   the joint distribution across wheels of one train.

**Desired output.** A concrete sampling specification with full citations,
clearly separating cited values from extrapolations:
(i) P(a given wheel carries a flat) for an ordinary in-service passenger fleet;
(ii) flat length (and depth) distribution to sample from, with the run-in vs
new distinction; (iii) per-bogie/per-vehicle correlation structure;
(iv) a verdict on whether polygonization needs separate modelling for our
speed range and sprung-channel focus.

# Deep-research prompt — rail-profile & crack EOV randomization design for drive-by scour ML

Paste the block below into NotebookLM Deep Research (or Gemini Deep Research).
Written 2026-07-12, motivated by the Stage-2 empirical result: with per-passage
FRA-class-{4,5,6} profile regeneration + a fresh random deck crack on EVERY passage,
all suspension-filtered (car-body/bogie) channels collapsed to zero skill on scour
regression, while unsprung wheel channels partially survived. Before regenerating
Stage 2 (and generating stage1_full / Stage 3), the EOV sampling design must be
literature-anchored. See docs/framework_rationale.md (Stage-2 forensics entry).

---

**Topic: Realistic randomization of track irregularity profiles and co-existing deck
damage (cracks) when training machine-learning models on drive-by (vehicle-mounted)
sensor data for railway-bridge condition monitoring — what published studies actually
do, what pass-to-pass variability is physically real, and which vehicle measurement
locations survive roughness variability**

**Context.** We use a 2D vertical coupled train–track–bridge interaction (TTBI) model
(TTB-2D by D. Cantero; Zhai-type track) to train neural networks that estimate bridge
pier scour (support stiffness loss, labelled 0–60%) and abutment bearing seizure
(rotational stiffness, labelled 0–100%) from single-passage drive-by signals: car-body
and bogie vertical accelerations and pitch rates, and wheel (unsprung/axle-level)
vertical accelerations, at 70–90 km/h. As training EOV nuisances (domain
randomization, logged but not labelled) we currently: (a) REGENERATE the track
irregularity profile on EVERY passage from the FRA PSD with new random phases AND a
severity class drawn per passage from FRA classes {4, 5, 6}; (b) apply a fresh random
deck crack on EVERY passage with probability 1.0, local EI loss U(0.05, 0.30), at a
uniform random location in [0.1, 0.9]·L (Sinha-type local EI reduction). Empirically
this destroyed all sprung-channel skill while wheel channels retained partial skill —
we must determine whether this is a defensible finding or an over-aggressive EOV
design before re-generating months of data.

**Research questions.**

1. **Profile randomization policy in published drive-by ML studies.** For studies that
   train learning-based damage detectors/estimators on simulated vehicle response
   (road OR rail): do they (i) keep ONE fixed profile for all runs, (ii) redraw an
   independent profile realization per run/passage, (iii) redraw only the random
   phases within a FIXED roughness class, or (iv) model a per-track base profile with
   small pass-to-pass perturbations and/or slow degradation? Check specifically (and
   verify — do not assume): Locke et al. (2020, J. Sound & Vibration, drive-by deep
   learning with environmental/operational effects); Sarwar & Cantero (deep
   autoencoder drive-by); Corbally & Malekjafarian (2022–2024 drive-by railway
   algorithms); Malekjafarian, OBrien and co-authors' indirect-monitoring reviews;
   Kamariotis et al. (CNN drive-by scour); Fernandes et al. (2024–2026 drive-by
   railway scour/damage ML). For each: profile class/amplitude range, redraw
   frequency, and reported effect on detection performance.

2. **What pass-to-pass profile variability is physically real?** For a SINGLE track
   over days–months (between maintenance): how much does the measured irregularity
   profile actually change between two passages (measurement repeatability of
   track-recording cars / axle-box systems; short-term ballast memory; moisture and
   temperature effects)? Standard deviation growth rates per MGT (Sato, Shenton, EN
   13848-based degradation studies) — over 50 passages of one vehicle, is a change of
   even one FRA class plausible? Verdict: is per-passage INDEPENDENT class-and-phase
   redraw over {4,5,6} a defensible worst-case domain randomization for training, or
   physically indefensible variability that reviewers will reject as the explanation
   for poor performance? Is the literature-consistent alternative a per-damage-state
   (or per-track) draw: one class + one phase realization per state, held for its 50
   passages, plus small per-passage perturbation?

3. **FRA class range vs line speed.** FRA track classes permit maximum speeds
   (class 4 ≈ 60 mph freight / 80 mph passenger; class 5–6 higher). For a 70–90 km/h
   secondary/regional line, which classes are operationally plausible? PSD amplitude
   parameters (A_v ladder, e.g. Berawi) per class 4/5/6 with units, and typical RMS
   vertical irregularity (mm) per class in the 1–25 m wavelength band. Would
   restricting to a single class with random phases (or classes {5,6}) be better
   supported than {4,5,6}?

4. **Co-existing persistent damage as a training nuisance (the crack).** A deck crack
   is a PERSISTENT condition, not a per-passage transient. In studies that include
   co-existing/confounding damage when training damage estimators (e.g. Fernandes et
   al. 2025 crack+bearing+scour; open-set/domain-adaptation SHM work): is the nuisance
   damage sampled per SCENARIO/STATE (held fixed across repeated passes) or
   re-randomized per training sample? Is crack probability 1.0 (every passage cracked,
   5–30% EI loss anywhere on the deck) defensible against field crack prevalence and
   severity statistics for railway bridge decks (concrete/composite)? What
   distributions (prevalence, EI-loss equivalent, location) does the literature
   support for a nuisance-crack model?

5. **Which vehicle measurement locations survive roughness variability?** Evidence
   comparing car-body vs bogie vs axle-box/unsprung measurements for bridge-condition
   information under realistic roughness: (i) axle-box acceleration literature (track
   AND bridge monitoring) — does the unsprung mass, being in direct contact through
   the wheel–rail interface, retain bridge-deflection information that suspension
   filtering removes from body/bogie channels? (ii) any published case where
   body/bogie-based estimation FAILS under roughness variation while axle-box-based
   estimation survives? (iii) frequency-band arguments: quasi-static bridge deflection
   under a moving vehicle vs roughness excitation 0.5–15 Hz at 70–90 km/h — is
   sprung-channel collapse under strong roughness randomization *expected* from first
   principles? We need citations to either defend the empirical inversion we observed
   (wheels beat car-body under profile EOV — opposite of our clean-track Stage 0) or
   to challenge it.

6. **Scour/stiffness drive-by detection with roughness included.** Among drive-by
   scour or support-stiffness studies (Prendergast, Fitzgerald, Malekjafarian scour
   review, Kamariotis, Fernandes): which included road/track roughness variability at
   all, at what severity, and what accuracy did they retain? Has ANY study
   demonstrated scour severity REGRESSION (not just detection) from vehicle response
   with per-run roughness variation?

7. **Multi-sensor fusion as mitigation.** Precedents for fusing an unsprung channel
   (profile reference) with a sprung channel (bridge response) — e.g. residual/
   subtraction methods (OBrien/Keenahan two-axle profile-independent ideas, TSD-style
   approaches) or learned fusion — to cancel roughness and recover bridge information.
   Does the literature support "wheel + car-body pair beats wheel + wheel pair under
   roughness EOV"?

**Deliverable.** For each question: the concrete finding with author/year/venue
citations, distinguishing VERIFIED statements (with quotes/numbers) from inference.
End with a recommended sampling specification we can implement directly:
profile draw frequency (per-state vs per-passage), class set, perturbation model,
crack draw frequency, crack prevalence/severity/location distributions — each
parameter tied to a source, flagging any parameter that remains an assumption.

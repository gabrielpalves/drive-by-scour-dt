# The digital twin — Torzoni 2024 vs 2026, and how ours maps (Paper 2 teaser)

Purpose: a self-contained explainer for the SHM-group talk's DT portion (optional/backup) and
the Paper-2 framing. Explains the two Torzoni frameworks we build on, their difference, and our
staged drive-by DT — "for Torzoni the DBN works like this; for us what changes is…". Grounded
in the local ports (`Torzoni2024/`, `Torzoni2026/`) + our `digital_twin/`. Verify specific
reward/cost numbers against the code/PDFs at write-time (flagged ⚠).

---

## 1. What a predictive digital twin is (one paragraph)

A predictive DT is a probabilistic model of a specific structure that (i) **infers** the hidden
health state from monitoring data (observation model), (ii) **predicts** how that state will
evolve (a degradation/transition model), and (iii) **plans** maintenance actions that trade
off risk against cost over a horizon (a decision/planning layer). The recurring machinery is a
**dynamic Bayesian network (DBN)**: health state `D_t` evolves by a Markov transition that
depends on the chosen action `a_t`; a noisy observation `U_t` (here, a classifier output)
depends on `D_t`; actions are chosen to optimise expected cost/utility. Torzoni gives us two
concrete realisations of the planning layer.

---

## 2. Torzoni 2024 — DBN + explicit planning (POMDP-style)

Local port: `Torzoni2024/bridge/dtframework.py` (+ `beam/` 2- and 4-action variants).

- **State.** Damage class `D ∈ {0…K}` (location × severity levels). Degradation is a
  **first-order Markov** process encoded as a **lower-diagonal transition matrix** (damage only
  advances or holds — `_get_low_diag_transition(prob_adv)`), one matrix per action.
- **Actions.** A small set — e.g. **do-nothing**, **perfect maintenance** (a "restart"
  transition that resets the state — `_get_restart_transition`), and **restricted operation**
  (a slower-degradation transition). The 2- and 4-action beam variants differ only in how many
  of these are exposed.
- **Observation model.** A **confusion matrix** `conf_mat_dt` maps the true damage state to the
  classifier's reported state — i.e. the DT explicitly carries the *classifier's own error*.
  This is the hook where **our drive-by champion model plugs in**: its per-class confusion (or
  regression error binned to classes) *is* `conf_mat_dt`.
- **Planning.** Actions are chosen by combining the action-conditioned transitions with a
  **reward/cost** structure (⚠ Torzoni's rewards ≈ +12 / −8 / −15 / −20 for the
  operate/restrict/maintain/fail-type outcomes — verify in code) over the DBN, POMDP-style
  (belief over `D`, action optimising expected discounted reward).
- **Sensing.** **Structure-mounted** sensors (permanent instrumentation on the bridge).

*In one line:* 2024 = a DBN with hand-specified action-conditioned transitions + a reward
table, planned explicitly. Interpretable, auditable, and the natural home for a
classifier-confusion observation model.

---

## 3. Torzoni 2026 — active inference (the same problem, one objective)

Local port: `Torzoni2026/src/` (`active_inference_loop.py`, `generative_model.py`,
`digital_asset.py`, `physical_asset.py`), on **pymdp**.

- **State.** Joint damage states as the **Cartesian product** of factors (`"location;damage_
  level"`), same physical content as 2024.
- **One objective replaces two.** Instead of a separate observation model + reward table,
  active inference posits a single **generative model** and selects actions by minimising
  **expected free energy (EFE)**. EFE decomposes into a **pragmatic** term (reach preferred/
  safe observations = the "reward") and an **epistemic** term (reduce uncertainty = information
  gain). So **exploration/inspection falls out of the objective** rather than being hand-added.
- **Agent.** A pymdp `Agent`; the loop (`ActiveInfLoop`) couples a **physical asset**
  (ground-truth degradation) to a **digital asset** (the agent's beliefs), optionally **learning**
  the generative-model parameters online.

*In one line:* 2026 = recast the DT decision as active inference; value-of-information
(inspection) is intrinsic (the epistemic term), not bolted on.

## 3.1 The difference, crisply

| | **2024 (DBN + POMDP-style)** | **2026 (active inference)** |
|---|---|---|
| Objective | reward/cost table + explicit planning | single EFE (pragmatic + epistemic) |
| Value of information | must be added (e.g. inspect action's reward) | **intrinsic** (epistemic term) |
| Observation model | explicit **confusion matrix** | likelihood in the generative model |
| Transparency | high (auditable CPDs/rewards) | compact, but EFE is less inspectable |
| Best when | you want interpretable, regulator-facing policies | you want principled exploration/UQ-driven action |

We treat them as **complementary planners** behind the same DT state, not competitors — 2024
is the interpretable baseline; 2026 is the information-theoretic upgrade. (Our `digital_twin/`
already toggles heuristic / POMDP / active-inference planners.)

---

## 4. What changes for US (the drive-by DT — Paper 2)

Same DBN spine; four substantive changes, each a Paper-2 contribution:

1. **Observation source: drive-by, not structure-mounted.** The observation `U_t` is produced
   by an **in-service train passing over the bridge**, decoded by our **champion PAA+N-HiTS
   model** — not by permanent instrumentation. Consequence: observations are **event-triggered**
   (they arrive when a train crosses), and their **quality varies** (speed/EOV, and — from
   Paper 1 — the sensor set). The champion's error map *is* the DT's observation model
   (`conf_mat_dt` for a DBN; a likelihood for AIF). This is the concrete bridge from Paper 1 to
   Paper 2: *the ablation picks the model whose confusion becomes the DT's eyes.*
2. **State: continuous, multi-pier + bearing.** Paper 1's regression heads give **continuous
   per-pier severity + bearing seized-%**, not one coarse class — a richer DT state (and a
   reason the classification→regression move matters downstream). Multi-foundation dependence
   (shared river hydrology) enters as a copula on the transition.
3. **Degradation: flood/shock-driven, not purely gradual.** Scour is **not** a smooth
   monotone advance — it jumps at **flood events** (a compound-Poisson shock term) between long
   quiescent periods. Our transition model threads this physics (`digital_twin/flood.py`): the
   **decision that matters** is *inspect-after-flood vs wait-for-the-next-train*, which the
   gradual-degradation DBNs don't pose. This is the novel decision Paper 2 centres.
4. **Value of information in monetary terms.** We attach a **€ cost model** (⚠ scour
   consequence figures from Kamariotis: ~€20k/€600k/€50M/€150k-day tiers) and quantify the
   **value of a drive-by observation / an inspection** against it — the "is another train pass
   worth waiting for, or do we inspect now?" trade, risk-adjusted (CVaR/entropic toggles).

## 4.1 Staged DT, mirroring the ablation stages

- **DT-Stage A (single scour, 2024 methodology).** One foundation, gradual + flood-shock scour,
  actions {do-nothing, restrict, inspect, maintain}, **DBN** planner, observation = champion on
  a single pier. Reproduce Torzoni-2024's loop with our drive-by observation + flood transition
  → the minimal end-to-end DT. *"Torzoni's DBN works like this; for us the transition gains a
  flood-shock term and the observation comes from a passing train."*
- **DT-Stage B (+ bearing / multi-pier).** Lift to Paper 1's multi-head state; add the
  scour↔bearing leakage (Paper 1 §5.4) as observation-model correlation.
- **DT-Stage C (VoI + risk).** Turn on the €-cost VoI and risk-sensitivity; compare
  **heuristic vs POMDP (2024) vs active-inference (2026)** planners on the same episode
  (flood → inspect-or-wait). This is the Todd-facing centrepiece.
- **DT-Stage D (sensor health / degraded observations).** Let the observation model degrade
  (sensor faults, 2→1 sensor fallback) and show the planner adapt — closes the loop with the
  Paper-1 sensor-economy result.

*In one line for the talk:* Paper 1 chooses the model and quantifies its error; Paper 2 makes
that error the DT's observation model, adds flood-shock scour evolution, and asks the decision
the drive-by setting uniquely poses — **inspect after the flood, or wait for the next train?** —
answered by a DBN (2024) and an active-inference agent (2026) over a € value-of-information.

---

## 5. If asked "why two Torzoni frameworks?"
Because they answer different reviewer instincts: 2024 gives an **interpretable, auditable**
policy (regulators, asset owners); 2026 gives a **principled treatment of information-gathering**
(when is it worth inspecting?). We implement both behind one DT state so we can show they agree
on the easy calls and characterise where the epistemic term changes the inspect/wait boundary.

## Verify-before-use
- ⚠ Torzoni 2024 reward magnitudes and action set (2- vs 4-action) — `Torzoni2024/*/dtframework*.py`.
- ⚠ € cost tiers (Kamariotis) — cross-check `docs/framework_rationale.md` cost entry + the PDFs.
- ⚠ Flood-shock parameters (compound-Poisson rate, jump size) — `digital_twin/flood.py`.
- Confirm which planners are wired in `digital_twin/` and that all three run end-to-end.

"""Reproducible Monte Carlo verification of the track-EOV count/extent priors.

Restores (as reviewable repository code) the retired scratchpad
``check_track_stats.py`` whose absence left the derived statistics quoted in
``docs/track_eov_sampling_spec.md`` without a reproducible verification.

What is verified, per 100 m and for the two bridge-local campaign windows
(window = track_L_app + deck + track_L_after, with the two 30 m margins
parsed live from ttbi.campaign_setup => L60: 30+60+30 = 120.0 m; L99.6: 30+99.6+30 =
159.6 m):

  1. hanging sleepers -- Poisson(rate 3.0 groups/100 m) with group size
     DiscreteUniform{1..5} consecutive sleepers (0.6 m spacing):
     (a) RAW incidence share (overlap-ignoring arithmetic):
         3.0 * 3.0 * 0.6 / 100 = 5.4%;
     (b) EFFECTIVE unique unsupported share (overlap-aware lattice MC,
         fully-contained group starts, homogeneous placement);
  2. ballast patches -- Poisson(rate 1.2/100 m) with length U(5, 20) m,
     fully-contained start-coordinate sampling (start ~ U(0, W - L), as in
     the campaign patch sampler):
     (a) RAW fouled-length fraction (overlap-ignoring): rate * 12.5 / 100;
     (b) UNION (overlap-corrected) fouled fraction via exact interval union;
     plus the prior-level fouling-rate sensitivity lambda in {0.6, 1.2, 2.4};
  3. rail pads -- independent Bernoulli(0.02) failures on the 0.6 m lattice
     (rule 'independent-bernoulli-sleeper-lattice-v1'):
     expected 0.02 * 100 / 0.6 = 3.33 failed positions per 100 m.

PLACEMENT SEMANTICS: this checker uses HOMOGENEOUS placement. Production
additionally applies transition-zone and fouled-patch weights and a 3x
abutment density for patches; those weights redistribute events in space and
therefore change OVERLAP (union/unique values) somewhat, but not the
raw count/extent totals. All union/unique figures printed here are labeled
"homogeneous" accordingly and are regression pins, not campaign priors.

DRIFT GUARD (multi-way, uniqueness-checked). What is machine-bound at run
time:
  (i)   all 21 registered track-EOV prior entries in
        ``scour_MATLAB/+ttbi/campaign_setup.m`` (20 numeric constants plus the
        pad_failure_rule string) are parsed with uniqueness-checked
        patterns: zero or two-plus live (comment-stripped) assignments of a
        guarded name is a hard failure naming that constant, and every
        parsed value must equal the documented expectation below;
  (ii)  sleeper spacing is parsed from its real source -- the single
        ``Track.Sleeper.spacing`` assignment in each of
        ``scour_MATLAB/TrackProp_Zhai_et_al_NoBallastOnBridge.m`` and
        ``scour_MATLAB/TrackProp_Zhai_et_al_WithBallastOnBridge.m`` (both
        must agree and equal 0.6 m) -- and the parsed value is used
        everywhere this checker needs the spacing;
  (iii) the three semantic lines of ``scour_MATLAB/sample_pad_failures.m``
        (inclusive sleeper lattice, one Bernoulli draw per lattice point,
        selection without replacement) must each appear exactly once; this
        is only the structural leg -- the deep source/call-site/mutation
        pinning of that helper lives in ``check_profile_pad_contract.py``;
  (iv)  every guarded numeric is regenerated as the exact phrase the spec
        states (template strings filled from the PARSED values) and each
        phrase is required, whitespace-normalized, in
        ``docs/track_eov_sampling_spec.md`` -- if campaign_setup changes, the
        regenerated phrase vanishes from the spec and the lookup fails; if
        the spec's number or wording changes, the lookup fails too.
Remaining one-way/qualitative: the spec's prose rationale and evidentiary
labels are not machine-checked; the bridge deck lengths (60 m / 99.6 m) and
the canonical 100 m normalization window are checker-chosen constants (the
MC windows themselves are computed from the parsed track_L_app /
track_L_after margins); and the production placement samplers are only
mirrored at prior level (see the scope note below).

Scope note: this verifies the PRIOR-level arithmetic of the sampling spec.
It does not re-implement the production MATLAB samplers (campaign_setup/A00/B54) or their
Python parity mirror; a response-level sensitivity study would require
additional generation rungs and is out of scope for the frozen R11 campaign.

Deterministic: fixed PCG64 seed, no writes, numpy-only. Exit code 0 iff
every assertion (source bindings, analytic matches, structural
inequalities, and pinned regression bands) holds.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

SEED = 20260728
N_MC = 200_000

# Documented expectations, cross-checked against campaign_setup.m at run time.
# EVERY entry must have exactly ONE live (non-comment) assignment in campaign_setup;
# scalar entries are floats, range entries are (lo, hi) tuples. The four
# ballast_eta_* pairs sit two-per-line in campaign_setup, which is why the parser is
# word-boundary anchored (finditer), not line-start anchored.
EXPECTED_PRIORS = {
    # scalars (campaign_setup order)
    "track_L_app": 30.0,
    "track_L_after": 30.0,
    "hang_rate_100m": 3.0,
    "ballast_rate_100m": 1.2,
    "hang_foul_mult": 3.0,
    "ballast_trans_mult": 3.0,
    "ballast_trans_margin": 20.0,
    "ballast_p_wet": 0.5,
    "hang_p_transition": 0.6,  # flagged [assumption] in campaign_setup
    "hang_trans_margin": 15.0,
    "pad_p_fail": 0.02,
    # [lo hi] ranges (campaign_setup order)
    "ballast_patch_len": (5.0, 20.0),
    "ballast_eta_k_dry": (1.2, 2.0),
    "ballast_eta_c_dry": (0.4, 0.8),
    "ballast_eta_k_wet": (0.7, 0.9),
    "ballast_eta_c_wet": (1.5, 4.0),
    "hang_group_size": (1.0, 5.0),
    "pad_chi_range": (1.0, 3.5),
    "pad_weibull": (1.8, 2.2),
    "pad_beta_range": (0.8, 1.2),
}
EXPECT_PAD_RULE = "independent-bernoulli-sleeper-lattice-v1"

# Sleeper spacing is NOT assigned in campaign_setup: production passes
# Track.Sleeper.spacing (from the TrackProp files) into sample_pad_failures.
# Both TrackProp variants must assign it exactly once and agree on this value.
EXPECT_SLEEPER_SPACING_M = 0.6
TRACKPROP_FILES = (
    "TrackProp_Zhai_et_al_NoBallastOnBridge.m",
    "TrackProp_Zhai_et_al_WithBallastOnBridge.m",
)

# sample_pad_failures.m structural contract: the three semantic lines of the
# production helper (inclusive sleeper lattice, one Bernoulli draw per
# position, selection without replacement), each required exactly once. This
# is only the structural leg; check_profile_pad_contract.py pins the helper
# deeply (signature, single terminal assignments, A00 call site, B54
# consumer, and 16 mutation tests).
PAD_HELPER_LINES = (
    "lattice = 0:pad_spacing:track_window;",
    "failed = rand(size(lattice)) < failure_probability;",
    "positions = lattice(failed);",
)

BALLAST_RATE_SENSITIVITY = (0.6, 1.2, 2.4)

# Deck lengths of the two campaign bridge geometries (campaign_setup's L_bridge for the
# L60 and L99.6 decks). These are checker-chosen named constants, NOT parsed;
# the MC windows are COMPUTED as track_L_app + deck + track_L_after from the
# margins parsed out of campaign_setup.
DECK_L60_M = 60.0
DECK_L996_M = 99.6
CANONICAL_WINDOW_M = 100.0  # per-100 m normalization convention (checker choice)

TOL_SIGMA = 4.0  # analytic-vs-MC tolerance in standard errors (raw quantities)

# Regression pins for overlap-aware quantities (homogeneous placement,
# fully-contained sampling). These are drift-detection bands, not campaign
# priors; update only with a written justification.
PIN_UNION_FRACTION = {  # rate -> (value, abs tol), fraction of window length
    0.6: (0.0720, 0.0030),
    1.2: (0.1385, 0.0040),
    2.4: (0.2565, 0.0045),
}
PIN_UNIQUE_UNSUPPORTED = (0.0525, 0.0015)  # effective unique sleeper share


class DriftError(RuntimeError):
    """A uniqueness-checked source binding failed (0 or >=2 live matches)."""


_NUM = r"([0-9]+(?:\.[0-9]+)?)"
_LHS = r"(?<![\w.])"  # not part of a longer identifier, not a struct field


def _strip_matlab_comments(text: str) -> str:
    """Drop %{...%} block comments, then cut each line at its first '%'.

    Naive per-line cut is safe here: none of the guarded assignment lines
    contains a quoted '%' before its terminating ';', and removing comment
    text can only PREVENT false matches, never create them.
    """
    text = re.sub(r"(?ms)^[ \t]*%\{.*?^[ \t]*%\}[ \t]*$", "", text)
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def _require_unique(text: str, name: str, pattern: str, where: str) -> re.Match:
    """finditer + exactly-one-live-assignment check (hard fail otherwise)."""
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) != 1:
        raise DriftError(
            f"{where}: {name}: expected exactly 1 live assignment, "
            f"found {len(matches)}"
        )
    return matches[0]


def parse_campaign_priors(repo_root: Path) -> dict:
    """Parse all registered track-EOV prior constants from campaign_setup.m.

    Every guarded constant is located with re.finditer on comment-stripped
    source; 0 or >=2 matching live assignments raise DriftError naming the
    constant. Trailing % comments are tolerated (stripped before matching).
    """
    raw = (repo_root / "scour_MATLAB" / "+ttbi" / "campaign_setup.m").read_text(
        encoding="utf-8", errors="replace"
    )
    text = _strip_matlab_comments(raw)
    out: dict = {}
    for name, expected in EXPECTED_PRIORS.items():
        if isinstance(expected, tuple):
            pat = rf"{_LHS}{name}\s*=\s*\[\s*{_NUM}\s+{_NUM}\s*\]\s*;"
            m = _require_unique(text, name, pat, "campaign-prior drift guard")
            out[name] = (float(m.group(1)), float(m.group(2)))
        else:
            pat = rf"{_LHS}{name}\s*=\s*{_NUM}\s*;"
            m = _require_unique(text, name, pat, "campaign-prior drift guard")
            out[name] = float(m.group(1))
    m = _require_unique(
        text, "pad_failure_rule",
        rf"{_LHS}pad_failure_rule\s*=\s*'([^']+)'\s*;", "campaign-prior drift guard",
    )
    out["pad_failure_rule"] = m.group(1)
    return out


def parse_sleeper_spacing(repo_root: Path) -> float:
    """Parse Track.Sleeper.spacing from BOTH TrackProp files (must agree)."""
    vals = []
    for fname in TRACKPROP_FILES:
        raw = (repo_root / "scour_MATLAB" / fname).read_text(
            encoding="utf-8", errors="replace"
        )
        text = _strip_matlab_comments(raw)
        pat = rf"{_LHS}Track\.Sleeper\.spacing\s*=\s*{_NUM}\s*;"
        m = _require_unique(text, "Track.Sleeper.spacing", pat, fname)
        vals.append(float(m.group(1)))
    if vals[0] != vals[1]:
        raise DriftError(
            f"TrackProp drift guard: Track.Sleeper.spacing disagrees "
            f"between the two TrackProp files: {vals}"
        )
    return vals[0]


def build_spec_phrases(priors: dict, spacing_m: float) -> list[tuple[str, str, bool]]:
    """(label, phrase, require_exactly_one) triples, REGENERATED from the
    parsed values, that must appear in the whitespace-normalized spec.

    Templates literally reproduce the spec's own byte sequences (Unicode
    lambda/eta/chi/beta, en dash, multiplication sign). require_exactly_one
    is True where the phrase was verified unique in the spec; False (>=1)
    where the spec legitimately states the value more than once.
    """
    def g(v: float) -> str:
        return f"{v:g}"

    p_lo, p_hi = priors["ballast_patch_len"]
    gs_lo, gs_hi = priors["hang_group_size"]
    wb_l, wb_k = priors["pad_weibull"]
    chi_lo, chi_hi = priors["pad_chi_range"]
    be_lo, be_hi = priors["pad_beta_range"]
    ekd, ecd = priors["ballast_eta_k_dry"], priors["ballast_eta_c_dry"]
    ekw, ecw = priors["ballast_eta_k_wet"], priors["ballast_eta_c_wet"]
    p_fail = priors["pad_p_fail"]
    n_lat = int(round(CANONICAL_WINDOW_M / spacing_m))
    w60 = priors["track_L_app"] + DECK_L60_M + priors["track_L_after"]
    w996 = priors["track_L_app"] + DECK_L996_M + priors["track_L_after"]
    sens = ", ".join(g(v) for v in BALLAST_RATE_SENSITIVITY)
    return [
        ("hang_rate_100m (update table)",
         f"λ = {priors['hang_rate_100m']:.1f} per 100 m", True),
        ("hang_rate_100m (sampling body)",
         f"λ={priors['hang_rate_100m']:.1f} per 100 m", True),
        ("ballast_rate_100m (update table)",
         f"λ = {priors['ballast_rate_100m']:.1f} per 100 m", True),
        ("ballast_rate_100m (sampling body)",
         f"λ={priors['ballast_rate_100m']:.1f} per 100 m", True),
        ("ballast_patch_len (sampling body)",
         f"U({g(p_lo)}, {g(p_hi)}) m", True),
        ("ballast_patch_len (narrative)",
         f"U({g(p_lo)},{g(p_hi)}) m", False),
        ("hang_group_size",
         f"Discrete Uniform {g(gs_lo)}–{g(gs_hi)} consecutive sleepers",
         True),
        ("hang_trans_margin",
         f"±{g(priors['hang_trans_margin'])} m abutment transition zone",
         False),
        ("hang_p_transition",
         f"p_transition = {g(priors['hang_p_transition'])}", True),
        ("hang_foul_mult (update table)",
         f"×{g(priors['hang_foul_mult'])} inside a fouled patch", True),
        ("hang_foul_mult (coupling odds)",
         f"{g(priors['hang_foul_mult'])}:1 inside vs outside a fouled patch",
         True),
        ("ballast_trans_mult+margin (update table)",
         f"×{g(priors['ballast_trans_mult'])} within "
         f"{g(priors['ballast_trans_margin'])} m of an abutment", True),
        ("ballast_trans_mult (sampling body)",
         f"×{g(priors['ballast_trans_mult'])} density near bridge "
         f"transitions", True),
        ("ballast_p_wet",
         f"p_wet = {g(priors['ballast_p_wet'])}", True),
        ("ballast_eta dry pair",
         f"η_k ∈ [{ekd[0]:.1f}, {ekd[1]:.1f}], "
         f"η_c ∈ [{ecd[0]:.1f}, {ecd[1]:.1f}]", True),
        ("ballast_eta wet pair",
         f"η_k ∈ [{ekw[0]:.1f}, {ekw[1]:.1f}], "
         f"η_c ∈ [{ecw[0]:.1f}, {ecw[1]:.1f}]", True),
        ("pad_weibull (sampling body)",
         f"Weibull(λ = {g(wb_l)}, k = {g(wb_k)})", True),
        ("pad_weibull (narrative)",
         f"Weibull({g(wb_l)}, {g(wb_k)})", False),
        ("pad_chi_range",
         f"χ_pad ∈ [{chi_lo:.1f}, {chi_hi:.1f}]", True),
        ("pad_beta_range",
         f"β_pad ∈ [{be_lo:.1f}, {be_hi:.1f}]", True),
        ("pad_p_fail (severity)",
         f"p = {g(p_fail)}", True),
        ("pad_p_fail (lattice rule)",
         f"Bernoulli(`p={g(p_fail)}`)", True),
        ("pad expectation (lattice x p)",
         f"(100/{g(spacing_m)}) × {g(p_fail)} Bernoulli", True),
        ("sleeper spacing (spec header)",
         f"sleeper spacing {g(spacing_m)} m", True),
        ("sleeper spacing (lattice rule)",
         f"{g(spacing_m)}-m sleeper/pad lattice position", True),
        ("sleeper spacing (lattice count)",
         f"~{n_lat} sleepers", True),
        ("sleeper spacing (passing frequency)",
         f"f = v/{g(spacing_m)}", True),
        ("windows from track_L_app/track_L_after",
         f"{g(w60)} m at L60, {g(w996)} m at L99.6", True),
        ("lambda sensitivity (checker constant)",
         f"λ ∈ {{{sens}}}", True),
    ]


def interval_union_length(starts: np.ndarray, ends: np.ndarray) -> float:
    """Exact union length of [start, end) intervals (one realization)."""
    if starts.size == 0:
        return 0.0
    order = np.argsort(starts)
    s, e = starts[order], ends[order]
    total = 0.0
    cur_s, cur_e = s[0], e[0]
    for i in range(1, s.size):
        if s[i] <= cur_e:
            cur_e = max(cur_e, e[i])
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s[i], e[i]
    total += cur_e - cur_s
    return float(total)


def mc_hanging(rng: np.random.Generator, window_m: float, rate_100m: float,
               gs_lo: int, gs_hi: int,
               spacing_m: float) -> tuple[float, float, float, float]:
    """(raw mean, raw se, unique mean, unique se) of unsupported share."""
    lam = rate_100m * window_m / 100.0
    n_sleepers = int(round(window_m / spacing_m))
    counts = rng.poisson(lam, size=N_MC)
    raw = np.zeros(N_MC)
    unique = np.zeros(N_MC)
    for i in range(N_MC):
        c = counts[i]
        if c == 0:
            continue
        sizes = rng.integers(gs_lo, gs_hi + 1, size=c)
        starts = np.array([rng.integers(0, n_sleepers - sz + 1) for sz in sizes])
        raw[i] = sizes.sum() / n_sleepers
        unique[i] = interval_union_length(
            starts.astype(float), (starts + sizes).astype(float)
        ) / n_sleepers
    return (
        float(raw.mean()), float(raw.std(ddof=1) / np.sqrt(N_MC)),
        float(unique.mean()), float(unique.std(ddof=1) / np.sqrt(N_MC)),
    )


def mc_ballast(rng: np.random.Generator, window_m: float, rate_100m: float,
               len_lo: float, len_hi: float) -> tuple[float, float, float, float]:
    """(raw mean, raw se, union mean, union se); fully-contained sampler."""
    lam = rate_100m * window_m / 100.0
    counts = rng.poisson(lam, size=N_MC)
    raw = np.zeros(N_MC)
    union = np.zeros(N_MC)
    for i in range(N_MC):
        c = counts[i]
        if c == 0:
            continue
        lengths = rng.uniform(len_lo, len_hi, size=c)
        starts = rng.uniform(0.0, window_m - lengths)  # fully contained
        raw[i] = lengths.sum() / window_m
        union[i] = interval_union_length(starts, starts + lengths) / window_m
    return (
        float(raw.mean()), float(raw.std(ddof=1) / np.sqrt(N_MC)),
        float(union.mean()), float(union.std(ddof=1) / np.sqrt(N_MC)),
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    failures: list[str] = []

    print("check_track_prior_stats: seed=%d, N_MC=%d" % (SEED, N_MC))

    # 0. Source bindings (uniqueness-checked parses) ----------------------
    try:
        priors = parse_campaign_priors(repo_root)
        spacing_m = parse_sleeper_spacing(repo_root)
    except DriftError as exc:
        print(f"FAIL: {exc}")
        return 1

    print("[0] campaign-prior drift guard (scour_MATLAB/+ttbi/campaign_setup.m; exactly one live "
          "assignment found per constant):")
    for key, expected in EXPECTED_PRIORS.items():
        got = priors[key]
        ok = (np.allclose(got, expected) if isinstance(expected, tuple)
              else abs(got - expected) < 1e-12)
        print(f"    {key} = {got} (expected {expected}) -> {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"campaign-prior drift: {key}")
    ok = priors["pad_failure_rule"] == EXPECT_PAD_RULE
    print(f"    pad_failure_rule = '{priors['pad_failure_rule']}' -> "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("campaign-prior drift: pad_failure_rule")

    # 0b. Sleeper spacing from its real source ----------------------------
    ok = abs(spacing_m - EXPECT_SLEEPER_SPACING_M) < 1e-12
    print(f"\n[0b] sleeper spacing parsed from both TrackProp_Zhai_et_al_*.m "
          f"(exactly one assignment each, equal): {spacing_m} m "
          f"(expected {EXPECT_SLEEPER_SPACING_M}) -> {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("TrackProp drift: Track.Sleeper.spacing")

    # 0c. sample_pad_failures.m structural contract -----------------------
    helper_text = (repo_root / "scour_MATLAB" / "sample_pad_failures.m"
                   ).read_text(encoding="utf-8", errors="replace")
    print("\n[0c] sample_pad_failures.m structural contract (structural leg "
          "only; deep pinning in check_profile_pad_contract.py):")
    for needle in PAD_HELPER_LINES:
        n = helper_text.count(needle)
        ok = n == 1
        print(f"    '{needle}' count {n} (need ==1) -> {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"pad helper line count {n}: {needle}")

    # 0d. Spec numeric binding (phrases regenerated from parsed values) ---
    spec_raw = (repo_root / "docs" / "track_eov_sampling_spec.md"
                ).read_text(encoding="utf-8-sig")
    spec_norm = re.sub(r"\s+", " ", spec_raw)
    phrases = build_spec_phrases(priors, spacing_m)
    print(f"\n[0d] spec numeric binding (docs/track_eov_sampling_spec.md, "
          f"whitespace-normalized; {len(phrases)} phrases regenerated from "
          f"the PARSED values):")
    for label, phrase, exact in phrases:
        n = spec_norm.count(phrase)
        ok = (n == 1) if exact else (n >= 1)
        need = "==1" if exact else ">=1"
        print(f"    {label}: count {n} (need {need}) -> {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(
                f"spec phrase [{label}] {ascii(phrase)}: count {n}, need {need}"
            )

    rng = np.random.default_rng(np.random.PCG64(SEED))
    hang_rate = priors["hang_rate_100m"]
    gs_lo, gs_hi = (int(v) for v in priors["hang_group_size"])
    len_lo, len_hi = priors["ballast_patch_len"]
    pad_p = priors["pad_p_fail"]
    # MC windows are computed from parsed campaign margins (canonical 100 m is
    # the per-100 m normalization convention, a checker choice).
    windows_m = (
        CANONICAL_WINDOW_M,
        priors["track_L_app"] + DECK_L60_M + priors["track_L_after"],
        priors["track_L_app"] + DECK_L996_M + priors["track_L_after"],
    )

    # 1. Hanging sleepers -------------------------------------------------
    mean_gs = 0.5 * (gs_lo + gs_hi)
    analytic_raw = hang_rate * mean_gs * spacing_m / 100.0
    print(f"\n[1] hanging sleepers (homogeneous placement, fully-contained "
          f"group starts):\n    RAW incidence share (overlap-ignoring "
          f"arithmetic) analytic = {100 * analytic_raw:.2f}% (spec quotes 5.4%)")
    for window in windows_m:
        raw_m, raw_se, uni_m, uni_se = mc_hanging(
            rng, window, hang_rate, gs_lo, gs_hi, spacing_m
        )
        ok_raw = abs(raw_m - analytic_raw) <= TOL_SIGMA * raw_se
        ok_less = uni_m < raw_m
        pin, tol = PIN_UNIQUE_UNSUPPORTED
        ok_pin = abs(uni_m - pin) <= tol
        print(f"    window {window:6.1f} m: raw MC {100 * raw_m:.3f}% "
              f"-> {'OK' if ok_raw else 'FAIL'}; EFFECTIVE unique share "
              f"{100 * uni_m:.3f}% (se {100 * uni_se:.3f}%) "
              f"[unique<raw: {'OK' if ok_less else 'FAIL'}; pin "
              f"{100 * pin:.2f}+-{100 * tol:.2f}%: {'OK' if ok_pin else 'FAIL'}]")
        if not ok_raw:
            failures.append(f"hanging raw, window {window}")
        if not ok_less:
            failures.append(f"hanging unique>=raw, window {window}")
        if not ok_pin:
            failures.append(f"hanging unique pin, window {window}")

    # 2. Ballast patches + sensitivity -----------------------------------
    print(f"\n[2] ballast patches (homogeneous fully-contained sampling; "
          f"production adds transition/fouling weights, which change "
          f"overlap, not totals). Lengths U({len_lo:.0f},{len_hi:.0f}) m; "
          f"RAW fraction (overlap-ignoring) = rate*{0.5 * (len_lo + len_hi):.1f}/100")
    for rate in BALLAST_RATE_SENSITIVITY:
        analytic_raw = rate * 0.5 * (len_lo + len_hi) / 100.0
        pin, tol = PIN_UNION_FRACTION[rate]
        tag = " (campaign value)" if rate == priors["ballast_rate_100m"] else ""
        for window in windows_m:
            raw_m, raw_se, uni_m, uni_se = mc_ballast(
                rng, window, rate, len_lo, len_hi
            )
            ok_raw = abs(raw_m - analytic_raw) <= TOL_SIGMA * raw_se
            ok_less = uni_m < raw_m
            ok_floor = uni_m >= 0.8 * raw_m
            ok_pin = abs(uni_m - pin) <= tol
            print(f"    rate {rate:3.1f}/100 m{tag}, window {window:6.1f} m: "
                  f"raw MC {100 * raw_m:.2f}% vs analytic "
                  f"{100 * analytic_raw:.2f}% -> {'OK' if ok_raw else 'FAIL'}; "
                  f"union MC {100 * uni_m:.2f}% (se {100 * uni_se:.3f}%) "
                  f"[union<raw: {'OK' if ok_less else 'FAIL'}; "
                  f">=0.8*raw: {'OK' if ok_floor else 'FAIL'}; pin "
                  f"{100 * pin:.1f}+-{100 * tol:.1f}%: {'OK' if ok_pin else 'FAIL'}]")
            if not ok_raw:
                failures.append(f"ballast raw, rate {rate}, window {window}")
            if not ok_less:
                failures.append(f"ballast union>=raw, rate {rate}, window {window}")
            if not ok_floor:
                failures.append(f"ballast union floor, rate {rate}, window {window}")
            if not ok_pin:
                failures.append(f"ballast union pin, rate {rate}, window {window}")
            tag = ""

    # 3. Rail pads --------------------------------------------------------
    n_positions = int(round(100.0 / spacing_m))
    analytic_pads = pad_p * n_positions
    fails = rng.binomial(n_positions, pad_p, size=N_MC).astype(float)
    mean, se = float(fails.mean()), float(fails.std(ddof=1) / np.sqrt(N_MC))
    ok = abs(mean - analytic_pads) <= TOL_SIGMA * se
    print(f"\n[3] rail pads ('{EXPECT_PAD_RULE}'): expected failed "
          f"positions/100 m = {analytic_pads:.2f} (spec quotes ~3.3); "
          f"MC {mean:.3f} (se {se:.3f}) on the {n_positions}-position lattice "
          f"-> {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("pad failures per 100 m")

    print()
    if failures:
        print("FAIL:", "; ".join(failures))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

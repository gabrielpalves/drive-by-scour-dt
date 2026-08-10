"""Executable and mutation-tested rail-profile / pad-generation contract.

This guard closes two easy-to-miss scientific confounds:

* ``fixed`` and ``psd_fra`` must use one identical FRA-v2 class-4 spectrum;
  the registered s13->s14 contrast changes only the persistent phase draw.
* pad failures must be independent Bernoulli draws on unique sleeper-lattice
  positions, not a sampled count followed by rounded continuous coordinates.

The MATLAB smoke named below exercises the live functions.  This checker binds
that smoke to the production source and rejects in-memory source mutations.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Callable


ROOT = Path(__file__).resolve().parent
PATHS = {
    "a00": ROOT / "scour_MATLAB" / "A00_Run.m",
    "campaign_setup": ROOT / "scour_MATLAB" / "+ttbi" / "campaign_setup.m",
    "profile_config": (
        ROOT / "scour_MATLAB" / "+ttbi" / "build_profile_config.m"
    ),
    "state_executor": (
        ROOT / "scour_MATLAB" / "+ttbi" / "execute_generation_state.m"
    ),
    "generation_runner": (
        ROOT / "scour_MATLAB" / "+ttbi" / "run_generation_states.m"
    ),
    "a04": ROOT / "scour_MATLAB" / "A04_Options.m",
    "b19": ROOT / "scour_MATLAB" / "B19_GenerateProfile.m",
    "b54": ROOT / "scour_MATLAB" / "B54_TrackVectors.m",
    "track_damage": (
        ROOT / "scour_MATLAB" / "+ttbi" / "sample_track_damage.m"
    ),
    "pad_helper": ROOT / "scour_MATLAB" / "sample_pad_failures.m",
    "smoke": ROOT / "scour_MATLAB" / "smoke_audit.m",
    "spec": ROOT / "docs" / "track_eov_sampling_spec.md",
}


class ContractError(AssertionError):
    """The registered profile/pad behavior is absent or ambiguous."""


def _require(text: str, token: str, label: str, *, count: int = 1) -> None:
    actual = text.count(token)
    if actual != count:
        raise ContractError(f"{label}: expected {count}, found {actual}")


def _forbid(text: str, token: str, label: str) -> None:
    if token in text:
        raise ContractError(f"{label}: forbidden token returned: {token!r}")


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(
            f"mutation fixture expected one {old!r}, found {text.count(old)}"
        )
    return text.replace(old, new, 1)


def _replace_first_of(
    text: str, old: str, new: str, *, expected_count: int
) -> str:
    if text.count(old) != expected_count:
        raise AssertionError(
            f"mutation fixture expected {expected_count} {old!r}, "
            f"found {text.count(old)}"
        )
    return text.replace(old, new, 1)


def validate_contract(sources: dict[str, str]) -> None:
    a00 = sources["a00"]
    campaign_setup = sources["campaign_setup"]
    profile_config = sources["profile_config"]
    state_executor = sources["state_executor"]
    generation_runner = sources["generation_runner"]
    a04 = sources["a04"]
    b19 = sources["b19"]
    b54 = sources["b54"]
    track_damage = sources["track_damage"]
    pad_helper = sources["pad_helper"]
    smoke = sources["smoke"]
    spec = sources["spec"]

    contract = "fra-v2-class4-cycles-per-m-v1"
    for token, label in (
        ("profile_fra_classes  = 4;", "campaign FRA class"),
        ("profile_fixed_phase_seed = 20260728;", "fixed phase seed"),
        (
            f"profile_spectrum_contract = '{contract}';",
            "campaign spectrum identity",
        ),
    ):
        _require(campaign_setup, token, label)
    for token, label in (
        (
            "profile_config.phase_seed = config.profile_fixed_phase_seed;",
            "fixed-phase production assignment",
        ),
        (
            "profile_config.phase_seed = double(state_phase_seed_id);",
            "per-state phase assignment",
        ),
        (
            "'spectrum_contract', config.profile_spectrum_contract",
            "A00-to-A04 spectrum binding",
        ),
    ):
        _require(profile_config, token, label)
    _require(
        state_executor,
        "ttbi.build_profile_config( ...",
        "state executor-to-profile builder binding",
    )
    _forbid(
        a00,
        "profile_asset_sha256",
        "legacy Type-2 asset is not a registered run field",
    )

    switch_start = a04.index("switch Profile_cfg.mode")
    switch_end = a04.index("% ---- FRA ----", switch_start)
    switch = a04[switch_start:switch_end]
    init_at = a04.index("Calc = struct();")
    if init_at > switch_start:
        raise ContractError("Calc is initialized only after the profile switch")
    for token, label, count in (
        ("case 'fixed'", "fixed registered branch", 1),
        ("case 'psd_fra'", "state-phase registered branch", 1),
        (
            "Calc = local_configure_fra_v2(Calc, Profile_cfg);",
            "single shared FRA-v2 helper",
            3,
        ),
        ("Profile_cfg.phase_seed = 20260728;", "fixed seed default", 2),
    ):
        _require(switch, token, label, count=count)
    _forbid(switch, "Profile.Type = 2", "registered Type-2 asset branch")

    helper = a04[a04.index("function Calc = local_configure_fra_v2") :]
    for token, label in (
        (
            f"expected_contract = '{contract}';",
            "helper spectrum identity",
        ),
        (
            "elseif ~strcmp(Profile_cfg.spectrum_contract, expected_contract)",
            "helper contract mismatch gate",
        ),
        (
            "A_v_classes = [1.2107, 1.0181, 0.6816, 0.5376, 0.2095, 0.0339];",
            "FRA-v2 class ladder",
        ),
        ("Calc.Profile.Type = 1;", "PSD-generated profile type"),
        ("Calc.Profile.inputs(1) = 0.25;", "FRA-v2 k"),
        (
            "Calc.Profile.inputs(2) = A_v_classes(Profile_cfg.fra_class);",
            "FRA-v2 class amplitude",
        ),
        ("Calc.Profile.inputs(3) = 0.8245/(2*pi);", "cycles-per-m corner"),
        ("Calc.Profile.inputs(4) = 1e-4/(2*pi);", "PSD unit conversion"),
        ("Calc.Profile.min_WaveLength = 1.524;", "short cutoff"),
        ("Calc.Profile.max_WaveLength = 304.8;", "long cutoff"),
        (
            "Calc.Profile.spectrum_contract = expected_contract;",
            "runtime spectrum stamp",
        ),
    ):
        _require(helper, token, label)
    for field in (
        "Type",
        "inputs(1)",
        "inputs(2)",
        "inputs(3)",
        "inputs(4)",
        "min_WaveLength",
        "max_WaveLength",
        "fra_class",
        "spectrum_contract",
    ):
        _require(
            helper,
            f"Calc.Profile.{field} =",
            f"single terminal assignment of Profile.{field}",
        )

    # Independent numerical check of the exact implemented unit conversion.
    k = 0.25
    av = 0.5376
    wc = 0.8245
    nc = wc / (2 * math.pi)
    conv = 1e-4 / (2 * math.pi)
    for wavelength in (304.8, 30.0, 7.43, 3.0, 1.524):
        n = 1 / wavelength
        implemented = k * av * nc**2 / (n**2 * (n**2 + nc**2)) * conv
        reference = (
            2
            * math.pi
            * 1e-4
            * (k * av * wc**2)
            / ((2 * math.pi * n) ** 2 * ((2 * math.pi * n) ** 2 + wc**2))
        )
        if not math.isclose(implemented, reference, rel_tol=2e-15, abs_tol=0):
            raise ContractError("FRA-v2 cycles/radian conversion is inconsistent")

    for token, label in (
        (
            "rng_state_ = rng;",
            "seeded profile saves caller RNG",
        ),
        (
            "rng(double(Calc.Profile.phase_seed), 'twister');",
            "seeded profile uses the persisted phase seed and pinned RNG",
        ),
        (
            "rng(rng_state_);",
            "seeded profile restores caller RNG",
        ),
        (
            "Legacy stored synthetic realization (Type 2; no registered R11 rung)",
            "legacy Type-2 boundary",
        ),
    ):
        _require(b19, token, label)

    _require(
        a00,
        "ttbi.run_generation_states( ...\n"
        "    execution_context, completed, "
        "campaign_config.max_parfor_workers);",
        "A00-to-generation runner binding",
    )
    _forbid(
        a00,
        "ttbi.execute_generation_state(",
        "A00 must delegate scheduling to the generation runner",
    )
    _require(
        generation_runner,
        "function run_generation_states(context, completed, max_workers)",
        "generation runner public signature",
    )
    _require(
        generation_runner,
        "ttbi.execute_generation_state( ...\n"
        "        state_index, context, worker_attestation);",
        "generation runner-to-state executor binding",
    )
    _require(
        state_executor,
        "function execute_generation_state(state_index, context, worker_attestation)",
        "state executor public signature",
    )
    _require(
        generation_runner,
        "ttbi.require_generation_worker_attestation( ...\n"
        "        worker_attestation, provenance);",
        "runner validates worker attestation before dispatch",
    )
    _require(
        state_executor,
        "ttbi.require_generation_worker_attestation( ...\n"
        "    worker_attestation, context.provenance);",
        "state executor independently revalidates worker attestation",
    )
    _require(
        state_executor,
        "Track = A02_Track();",
        "A02 return binding in state executor",
    )
    _require(
        state_executor,
        "Damage, track_state, Track, context.track, ...",
        "A02 return is passed to the track sampler",
    )
    pad = track_damage
    for token, label in (
        (
            "pad_failure_rule  = 'independent-bernoulli-sleeper-lattice-v1';",
            "versioned pad rule",
        ),
        ("pad_p_fail        = 0.02;", "per-position failure probability"),
        (
            "pad_failures = sample_pad_failures( ...\n"
            "        track_window, Track.Sleeper.spacing, config.pad_p_fail);",
            "single production sampler call",
        ),
        (
            "track_state.pad_failures = pad_failures;",
            "single descriptor publication",
        ),
    ):
        target = (
            campaign_setup
            if token.startswith(("pad_failure_rule", "pad_p_fail"))
            else pad
        )
        _require(target, token, label)
    for forbidden in ("nf_ =", "randi([", "track_win*rand(", "sort(rand("):
        _forbid(pad, forbidden, "continuous/count pad sampler")
    canonical_publication = (
        "    pad_failures = sample_pad_failures( ...\n"
        "        track_window, Track.Sleeper.spacing, config.pad_p_fail);\n"
        "\n"
        "    track_state.ballast_patches = ballast_patches;\n"
        "    track_state.hanging_groups = hanging_groups;\n"
        "    track_state.pad_stiff_mult = pad_stiffness;\n"
        "    track_state.pad_damp_mult = pad_damping;\n"
        "    track_state.pad_failures = pad_failures;\n"
        "    track_state.pad_failure_rule = config.pad_failure_rule;"
    )
    _require(
        pad,
        canonical_publication,
        "adjacent sampler-to-descriptor publication",
    )
    fx_writes = re.findall(
        r"(?m)^\s*pad_failures(?:\s*\([^=\r\n]*\))?\s*=", pad
    )
    if len(fx_writes) != 1:
        raise ContractError(
            "pad failure output has an extra direct/indexed assignment"
        )
    if track_damage.count("track_state.pad_failures") != 1:
        raise ContractError(
            "track_state.pad_failures has more than one write/read"
        )
    for forbidden in (
        "track_state.(",
        "eval(",
        "assignin(",
        "setfield(",
        "subsasgn(",
    ):
        _forbid(pad, forbidden, "dynamic pad-descriptor mutation")

    for token, label in (
        (
            "function [positions, lattice, failed] = sample_pad_failures( ...",
            "testable pad sampler signature",
        ),
        (
            "lattice = 0:pad_spacing:track_window;",
            "inclusive helper lattice",
        ),
        (
            "failed = rand(size(lattice)) < failure_probability;",
            "one helper draw per lattice point",
        ),
        (
            "positions = lattice(failed);",
            "selection without replacement",
        ),
    ):
        _require(pad_helper, token, label)
    for name in ("lattice", "failed", "positions"):
        _require(
            pad_helper,
            f"{name} =",
            f"single terminal helper assignment of {name}",
        )

    for token, label in (
        (
            "pad_indices = local_validate_pad_failures(T.pad_failures,x,tol);",
            "strict pad-descriptor validation call",
        ),
        (
            "function indices = local_validate_pad_failures(positions,x,tol)",
            "strict pad-descriptor validator",
        ),
        (
            "[distance,indices(item)] = min(abs(x - positions(item)));",
            "descriptor-to-sleeper distance calculation",
        ),
        ("if distance > tol", "off-lattice pad rejection"),
        (
            "if numel(unique(indices)) ~= numel(indices)",
            "duplicate failed-pad rejection",
        ),
        ("i0 = pad_indices(r);", "validated pad-index consumption"),
        ("mult_pad_k(i0) = KILL;", "failed pad stiffness removal"),
        ("mult_pad_c(i0) = KILL;", "failed pad damping removal"),
    ):
        _require(b54, token, label)

    for token, label in (
        (
            "isequal(FixedA_.Profile.h, Same_.Profile.h)",
            "live equal-seed realization check",
        ),
        (
            "~isequal(FixedA_.Profile.h, Other_.Profile.h)",
            "live different-phase realization check",
        ),
        (
            "isequal(FixedA_.Profile.PSD_Y, Other_.Profile.PSD_Y)",
            "live phase-only PSD check",
        ),
        (
            "[sampled_, lattice_, failed_] = sample_pad_failures(",
            "smoke-to-generator pad binding",
        ),
        (
            "sampler_ok_ = isequal(lattice_, expected_lattice_)",
            "live sampler replay check",
        ),
        (
            "pad_mapping_ok_ = DbgP_.frame_offset == -1.2",
            "live non-zero-offset pad mapping check",
        ),
    ):
        _require(smoke, token, label)

    for token, label in (
        (
            "one state-wise global Weibull stiffness multiplier and one global "
            "uniform damping multiplier",
            "documented global pad service-condition rule",
        ),
        (
            "There is no imposed cap on consecutive failures.",
            "documented absence of run cap",
        ),
        (
            "the source report's proposed sleeper-wise ARIMA(5,1,0)",
            "documented ARIMA limitation",
        ),
    ):
        _require(spec, token, label)


def _mutations(baseline: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
    cases: list[tuple[str, dict[str, str]]] = []

    def add(name: str, key: str, transform: Callable[[str], str]) -> None:
        mutant = dict(baseline)
        mutant[key] = transform(mutant[key])
        cases.append((name, mutant))

    add(
        "fixed branch no longer uses shared helper",
        "a04",
        lambda s: _replace_first_of(
            s,
            "        Calc = local_configure_fra_v2(Calc, Profile_cfg);",
            "        Calc.Profile.Type = 2; % MUTANT",
            expected_count=3,
        ),
    )
    add(
        "FRA corner reverts to rad/m",
        "a04",
        lambda s: _replace_once(
            s, "Calc.Profile.inputs(3) = 0.8245/(2*pi);",
            "Calc.Profile.inputs(3) = 0.8245;",
        ),
    )
    add(
        "fixed phase seed drifts",
        "campaign_setup",
        lambda s: _replace_once(
            s, "profile_fixed_phase_seed = 20260728;",
            "profile_fixed_phase_seed = 20260729;",
        ),
    )
    add(
        "spectrum identity drifts in campaign setup",
        "campaign_setup",
        lambda s: _replace_once(
            s,
            "profile_spectrum_contract = 'fra-v2-class4-cycles-per-m-v1';",
            "profile_spectrum_contract = 'fra-v2-class5-cycles-per-m-v1';",
        ),
    )
    add(
        "legacy asset is reintroduced into run fingerprint",
        "a00",
        lambda s: s + "\n% profile_asset_sha256 MUTANT\n",
    )
    add(
        "fixed profile phase becomes passage-specific",
        "profile_config",
        lambda s: _replace_once(
            s,
            "profile_config.phase_seed = config.profile_fixed_phase_seed;",
            "profile_config.phase_seed = double(passage_profile_seed_id);",
        ),
    )
    add(
        "pad failure probability drifts in campaign setup",
        "campaign_setup",
        lambda s: _replace_once(
            s,
            "pad_p_fail        = 0.02;",
            "pad_p_fail        = 0.03;",
        ),
    )
    add(
        "pad failure rule identity drifts in campaign setup",
        "campaign_setup",
        lambda s: _replace_once(
            s,
            "pad_failure_rule  = 'independent-bernoulli-sleeper-lattice-v1';",
            "pad_failure_rule  = 'independent-bernoulli-sleeper-lattice-v2';",
        ),
    )
    add(
        "pad lattice shifts off sleeper positions",
        "pad_helper",
        lambda s: _replace_once(
            s,
            "lattice = 0:pad_spacing:track_window;",
            "lattice = 0.1:pad_spacing:track_window;",
        ),
    )
    add(
        "pad failures use one scalar draw",
        "pad_helper",
        lambda s: _replace_once(
            s,
            "failed = rand(size(lattice)) < failure_probability;",
            "failed = rand() < failure_probability;",
        ),
    )
    add(
        "A00 no longer delegates to the generation runner",
        "a00",
        lambda s: _replace_once(
            s,
            "ttbi.run_generation_states( ...\n"
            "    execution_context, completed, "
            "campaign_config.max_parfor_workers);",
            "ttbi.run_generation_states( ...\n"
            "    execution_context, completed, 16); % MUTANT",
        ),
    )
    add(
        "generation runner no longer delegates to the state executor",
        "generation_runner",
        lambda s: _replace_once(
            s,
            "    ttbi.execute_generation_state( ...\n"
            "        state_index, context, worker_attestation);",
            "    execute_generation_state( ...\n"
            "        state_index, context, worker_attestation); % MUTANT",
        ),
    )
    add(
        "production uses nonexistent caller Track instead of A02 return",
        "state_executor",
        lambda s: _replace_once(
            s,
            "    Track = A02_Track();",
            "    Track_local = A02_Track();",
        ),
    )
    add(
        "production overrides helper output with continuous positions",
        "track_damage",
        lambda s: _replace_once(
            s,
            "        track_window, Track.Sleeper.spacing, config.pad_p_fail);",
            "        track_window, Track.Sleeper.spacing, config.pad_p_fail);\n"
            "    pad_failures = sort(rand(1, numel(pad_failures)) * "
            "track_window);",
        ),
    )
    add(
        "production mutates helper output through indexed assignment",
        "track_damage",
        lambda s: _replace_once(
            s,
            "        track_window, Track.Sleeper.spacing, config.pad_p_fail);",
            "        track_window, Track.Sleeper.spacing, config.pad_p_fail);\n"
            "    pad_failures(:) = rand(1, numel(pad_failures)) * "
            "track_window;",
        ),
    )
    add(
        "production overwrites descriptor through dynamic setfield",
        "track_damage",
        lambda s: _replace_once(
            s,
            "    track_state.pad_failure_rule = config.pad_failure_rule;",
            "    track_state.pad_failure_rule = config.pad_failure_rule;\n"
            "    track_state = setfield(track_state, 'pad_failures', "
            "pad_failures + 0.1);",
        ),
    )
    add(
        "seeded profile stops pinning the Twister algorithm",
        "b19",
        lambda s: _replace_once(
            s,
            "        rng(double(Calc.Profile.phase_seed), 'twister');",
            "        rng(double(Calc.Profile.phase_seed)); % MUTANT",
        ),
    )
    add(
        "seeded profile leaks RNG state",
        "b19",
        lambda s: _replace_once(s, "        rng(rng_state_);", "        % MUTANT"),
    )
    add(
        "failed pad retains damping",
        "b54",
        lambda s: _replace_once(
            s, "            mult_pad_c(i0) = KILL;",
            "            mult_pad_c(i0) = 1; % MUTANT",
        ),
    )
    add(
        "off-lattice pad descriptor silently snaps to nearest sleeper",
        "b54",
        lambda s: _replace_once(
            s,
            "    if distance > tol",
            "    if false % MUTANT: accept off-lattice descriptor",
        ),
    )
    add(
        "duplicate failed-pad descriptors are accepted",
        "b54",
        lambda s: _replace_once(
            s,
            "if numel(unique(indices)) ~= numel(indices)",
            "if false % MUTANT: accept duplicate pad descriptors",
        ),
    )
    add(
        "runtime phase contrast assertion is inverted",
        "smoke",
        lambda s: _replace_once(
            s,
            "~isequal(FixedA_.Profile.h, Other_.Profile.h)",
            "isequal(FixedA_.Profile.h, Other_.Profile.h)",
        ),
    )
    add(
        "documentation claims a consecutive-failure cap",
        "spec",
        lambda s: _replace_once(
            s,
            "There is no imposed cap on consecutive failures.",
            "Failures are capped at three consecutive positions.",
        ),
    )
    return cases


def main() -> int:
    baseline = {
        key: path.read_text(encoding="utf-8") for key, path in PATHS.items()
    }
    validate_contract(baseline)
    print("[BASELINE PASS] live profile/pad source and MATLAB smoke are bound")
    cases = _mutations(baseline)
    for index, (name, mutant) in enumerate(cases, start=1):
        try:
            validate_contract(mutant)
        except ContractError:
            print(f"[MUTATION PASS {index}/{len(cases)}] rejected: {name}")
        else:
            raise AssertionError(f"profile/pad mutation escaped: {name}")
    print("PROFILE/PAD CONTRACT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

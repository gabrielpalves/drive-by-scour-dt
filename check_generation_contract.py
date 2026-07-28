"""Executable R11 MATLAB-generation provenance and pool contract.

This checker ties together the MATLAB generator, Python loader/study tag,
campaign environment lock, canonical MATLAB-environment identity, reviewed
generator-source root, atomic state serializer, resume guards, and bounded
process pool. Mutations operate only on in-memory strings.

Run:  python check_generation_contract.py
"""
from __future__ import annotations

import ast
import dis
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re
import types

from core.campaign_contract import (
    EXPECTED_GENERATION_BEHAVIOR_VERSION,
    EXPECTED_GEN_SCHEMA,
    EXPECTED_PROTOCOL_SCHEMA_TAG,
)


ROOT = Path(__file__).resolve().parent
A00_PATH = ROOT / "scour_MATLAB" / "A00_Run.m"
SAVE_PROGRESS_PATH = ROOT / "scour_MATLAB" / "save_progress.m"
ENV_IDENTITY_PATH = ROOT / "scour_MATLAB" / "matlab_environment_identity.m"
CURRENT_ENV_PATH = ROOT / "scour_MATLAB" / "current_matlab_environment.m"
SOURCE_ROOT_PATH = ROOT / "scour_MATLAB" / "generator_source_root.m"
CRN_SMOKE_PATH = ROOT / "scour_MATLAB" / "smoke_crn_state_design.m"
PROVENANCE_SMOKE_PATH = (
    ROOT / "scour_MATLAB" / "smoke_r11_provenance_serialization.m"
)
MAKE_MICRO_PATH = ROOT / "make_micro_smoke.py"
DATASET_PATH = ROOT / "core" / "dataset.py"
DRIVER_PATH = ROOT / "comprehensive_ablation_multidamage.py"
ENVIRONMENT_PATH = ROOT / "environment" / "campaign-py313-cu128.json"
SOURCE_MANIFEST_PATH = ROOT / "bundle_source_files.txt"

EXPECTED_SCHEMA = EXPECTED_GEN_SCHEMA
EXPECTED_BEHAVIOR_VERSION = EXPECTED_GENERATION_BEHAVIOR_VERSION
EXPECTED_STUDY_TAG = EXPECTED_PROTOCOL_SCHEMA_TAG
EXPECTED_MATLAB_RELEASE = "R2025b"
EXPECTED_ENVIRONMENT_SCHEMA = "ttbi-campaign-environment-v2"
EXPECTED_MATLAB_ENVIRONMENT_SHA256 = (
    "958e7fe28f70577e9cb77aba0443c127d0a99726042a4618f7cce88d557fce79"
)
EXPECTED_MATLAB_ENVIRONMENT = {
    "release": "R2025b",
    "version": "25.2.0.3177638 (R2025b) Update 5",
    "arch": "win64",
    "blas": (
        "Intel(R) oneAPI Math Kernel Library Version 2024.1-Product Build "
        "20240215 for Intel(R) 64 architecture applications (CNR branch AVX2)"
    ),
    "lapack": (
        "Intel(R) oneAPI Math Kernel Library Version 2024.1-Product Build "
        "20240215 for Intel(R) 64 architecture applications (CNR branch AVX2) "
        "supporting Linear Algebra PACKage (LAPACK 3.11.0)"
    ),
    "matlab_product_version": "25.2",
    "statistics_toolbox_version": "25.2",
    "parallel_toolbox_version": "25.2",
}
MATLAB_ENVIRONMENT_FIELDS = tuple(sorted(EXPECTED_MATLAB_ENVIRONMENT))
BEHAVIOR_KEY = "generation_behavior_version"
LEGACY_KEYS = ("gen_rule_ver", "track_eov_impl")

ENVIRONMENT_SOURCE = ENVIRONMENT_PATH.read_text(encoding="utf-8")
SAVE_PROGRESS_SOURCE = SAVE_PROGRESS_PATH.read_text(encoding="utf-8")
ENV_IDENTITY_SOURCE = ENV_IDENTITY_PATH.read_text(encoding="utf-8")
CURRENT_ENV_SOURCE = CURRENT_ENV_PATH.read_text(encoding="utf-8")
SOURCE_ROOT_SOURCE = SOURCE_ROOT_PATH.read_text(encoding="utf-8")
CRN_SMOKE_SOURCE = CRN_SMOKE_PATH.read_text(encoding="utf-8")
PROVENANCE_SMOKE_SOURCE = PROVENANCE_SMOKE_PATH.read_text(encoding="utf-8")
MAKE_MICRO_SOURCE = MAKE_MICRO_PATH.read_text(encoding="utf-8")


class ContractError(AssertionError):
    """A generation-identity invariant is absent or ambiguous."""


def _one(pattern: str, text: str, label: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ContractError(
            f"{label}: expected exactly one declaration, found {len(matches)}"
        )
    return matches[0]


def _once(text: str, token: str, label: str) -> None:
    count = text.count(token)
    if count != 1:
        raise ContractError(f"{label}: expected once, found {count}")


def _driver_schema_tag(driver: str) -> str:
    """Require one direct module binding to the canonical protocol constant."""
    try:
        tree = ast.parse(driver, filename=str(DRIVER_PATH))
    except SyntaxError as exc:
        raise ContractError(
            f"Python campaign driver is not valid syntax: {exc.msg}"
        ) from exc

    try:
        module_code = compile(
            driver,
            str(DRIVER_PATH),
            "exec",
            dont_inherit=True,
            optimize=0,
        )
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ContractError(
            f"Python campaign driver cannot be compiled: {exc}"
        ) from exc

    schema_writes: list[tuple[str, int | None, str]] = []
    schema_deletes: list[tuple[str, int | None, str]] = []

    def inspect_bindings(code: types.CodeType, *, module_scope: bool) -> None:
        write_ops = {"STORE_GLOBAL"}
        delete_ops = {"DELETE_GLOBAL"}
        if module_scope:
            write_ops.add("STORE_NAME")
            delete_ops.add("DELETE_NAME")
        for instruction in dis.get_instructions(code):
            if instruction.argval != "SCHEMA_TAG":
                continue
            record = (code.co_name, instruction.offset, instruction.opname)
            if instruction.opname in write_ops:
                schema_writes.append(record)
            elif instruction.opname in delete_ops:
                schema_deletes.append(record)
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                inspect_bindings(constant, module_scope=False)

    inspect_bindings(module_code, module_scope=True)
    if len(schema_writes) != 1 or schema_deletes:
        raise ContractError(
            "Python study schema tag must have exactly one static binding; "
            f"found {len(schema_writes)} writes and "
            f"{len(schema_deletes)} deletes"
        )

    schema_assignments = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SCHEMA_TAG"
        )
    ]
    if len(schema_assignments) != 1:
        raise ContractError(
            "Python study schema tag must be one direct module assignment"
        )
    value = schema_assignments[0].value
    if not (
        isinstance(value, ast.Name)
        and value.id == "EXPECTED_PROTOCOL_SCHEMA_TAG"
    ):
        raise ContractError(
            "Python study schema tag does not derive from campaign_contract"
        )
    return EXPECTED_PROTOCOL_SCHEMA_TAG


def _matlab_environment_descriptor(environment: dict[str, str]) -> str:
    if set(environment) != set(MATLAB_ENVIRONMENT_FIELDS):
        raise ContractError("environment lock has the wrong MATLAB field set")
    lines: list[str] = []
    for field in MATLAB_ENVIRONMENT_FIELDS:
        value = environment[field]
        if (
            not isinstance(value, str)
            or not value
            or any(mark in value for mark in ("\r", "\n", "\x00"))
        ):
            raise ContractError(
                f"matlab_environment.{field} is not one safe text line"
            )
        lines.append(f"{field}={value}")
    # Cross-language contract: LF separators, UTF-8, and NO terminal LF.
    return "\n".join(lines)


def _validate_environment(environment_source: str) -> dict:
    try:
        lock = json.loads(environment_source)
    except json.JSONDecodeError as exc:
        raise ContractError("environment lock is not valid JSON") from exc
    if lock.get("schema") != EXPECTED_ENVIRONMENT_SCHEMA:
        raise ContractError("environment lock is not the reviewed v2 schema")
    if "matlab_release" in lock:
        raise ContractError("coarse top-level matlab_release key returned")
    matlab_environment = lock.get("matlab_environment")
    if matlab_environment != EXPECTED_MATLAB_ENVIRONMENT:
        raise ContractError("campaign MATLAB environment differs from exact lock")
    descriptor = _matlab_environment_descriptor(matlab_environment)
    actual_sha = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
    if actual_sha != EXPECTED_MATLAB_ENVIRONMENT_SHA256:
        raise ContractError(
            f"reviewed cross-language environment fixture drifted: {actual_sha}"
        )
    if lock.get("matlab_environment_sha256") != actual_sha:
        raise ContractError("environment lock digest does not authenticate descriptor")
    return lock


@lru_cache(maxsize=1)
def _validate_helpers() -> None:
    manifest_entries = [
        line
        for line in SOURCE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if manifest_entries != sorted(manifest_entries):
        raise ContractError("bundle source manifest is not sorted")
    if len(manifest_entries) != len(set(manifest_entries)):
        raise ContractError("bundle source manifest contains duplicates")
    required_manifest_entries = {
        "bundle_source_files.txt",
        "check_source_provenance.py",
        "core/source_provenance.py",
        "scour_MATLAB/current_matlab_environment.m",
        "scour_MATLAB/generator_source_root.m",
        "scour_MATLAB/matlab_environment_identity.m",
        "scour_MATLAB/smoke_crn_state_design.m",
        "scour_MATLAB/smoke_r11_provenance_serialization.m",
    }
    missing_manifest = required_manifest_entries.difference(manifest_entries)
    if missing_manifest:
        raise ContractError(
            f"bundle source manifest lacks R11 dependencies: "
            f"{sorted(missing_manifest)!r}"
        )
    missing_files = [
        name for name in manifest_entries if not (ROOT / name).is_file()
    ]
    if missing_files:
        raise ContractError(
            f"bundle source manifest contains missing files: {missing_files!r}"
        )

    for token in (
        "release_info = matlabRelease;",
        "full_version = char(version);",
        "matlab_product_version = regexp(full_version, '^\\d+\\.\\d+', ...",
        "'release', char(release_info.Release)",
        "'version', full_version",
        "'arch', char(computer('arch'))",
        "'blas', strtrim(char(version('-blas')))",
        "'lapack', strtrim(char(version('-lapack')))",
        "'matlab_product_version', matlab_product_version",
        "'statistics_toolbox_version', char(statistics_info.Version)",
        "'parallel_toolbox_version', char(parallel_info.Version)",
        "matlab_environment_identity(environment);",
    ):
        _once(CURRENT_ENV_SOURCE, token, f"actual MATLAB capture {token}")
    for token in (
        "required = sort({",
        "actual = sort(fieldnames(environment))'",
        "if ~isequal(actual, required)",
        "descriptor = strjoin(lines, newline);",
        "bytes = unicode2native(descriptor, 'UTF-8');",
        "java.security.MessageDigest.getInstance('SHA-256')",
        "'^[0-9a-f]{64}$",
    ):
        # The SHA regexp belongs to A00/save_progress, not this helper.
        if token == "'^[0-9a-f]{64}$":
            continue
        _once(ENV_IDENTITY_SOURCE, token, f"environment identity {token}")
    if "descriptor = [strjoin(" in ENV_IDENTITY_SOURCE:
        raise ContractError("MATLAB environment descriptor regained terminal bytes")
    for field in MATLAB_ENVIRONMENT_FIELDS:
        if ENV_IDENTITY_SOURCE.count(f"'{field}'") != 1:
            raise ContractError(
                f"MATLAB environment helper must name {field!r} exactly once"
            )

    for token in (
        "'bundle_source_files.txt'",
        "local_validate_manifest_path(entry);",
        "entry_path = fullfile(repository_root, entry_parts{:});",
        "'Manifest entry is not a regular file: %s'",
        "numel(unique(entries)) ~= numel(entries)",
        "numel(unique(lower_entries)) ~= numel(entries)",
        "selected = entries(startsWith(entries, 'scour_MATLAB/'));",
        "selected = sort(selected);",
        "fopen(path, 'rb')",
        "sprintf('%s:%s', relative_name,",
        "digest_lines = strjoin(sort(lines), newline);",
        "unicode2native(text, 'UTF-8')",
    ):
        _once(SOURCE_ROOT_SOURCE, token, f"generator source helper {token}")
    for unsafe in ("contains(entry, '\\')", "strcmp(components, '..')"):
        _once(SOURCE_ROOT_SOURCE, unsafe, f"unsafe manifest guard {unsafe}")


def _semantic_state_uids(
    bridge_length: float, spans: int, targets: tuple[int, ...]
) -> tuple[list[str], list[str]]:
    """Mirror the declared semantic UID grammar for a collision/count fixture."""

    geometry = "".join(f"{target:02d}" for target in targets)

    def uid(
        family: str, target: int, level: int, replica: int
    ) -> str:
        return (
            f"ttbi-state-v1|Lmm={round(1000 * bridge_length):06d}|"
            f"spans={spans}|scour={geometry}|family={family}|"
            f"target={target:02d}|level={level:04d}|rep={replica:03d}"
        )

    uids: list[str] = []
    families: list[str] = []
    for replica in range(1, 51):
        uids.append(uid("target_healthy", 0, 0, replica))
        families.append("target_healthy")
    for target in targets:
        for replica in range(1, 6):
            for level in range(1, 6):
                uids.append(uid("scour_only", target, level, replica))
                families.append("scour_only")
    for target in (1, 2):
        for replica in range(1, 6):
            for level in range(1, 6):
                uids.append(uid("bearing_only", target, level, replica))
                families.append("bearing_only")
    for replica in range(1, 51):
        uids.append(uid("nuisance_only", 0, 0, replica))
        families.append("nuisance_only")
    for row in range(1, 251):
        uids.append(uid("joint", 0, row, 1))
        families.append("joint")
    return uids, families


def _seed32(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


@lru_cache(maxsize=1)
def _validate_crn_numeric_fixture() -> None:
    """Exercise current full campaign counts and every named seed namespace."""

    schedule = "uid-named-substreams-v2"
    state_names = (
        "operations",
        "crack",
        "profile-state",
        "track",
        "profile-phase",
    )
    passage_names = ("profile-passage", "oor-passage")
    for length, spans, targets, expected in (
        (60.0, 3, (2, 3), 450),
        (99.6, 4, (2, 3, 4), 475),
    ):
        uids, families = _semantic_state_uids(length, spans, targets)
        if len(uids) != expected or len(set(uids)) != expected:
            raise ContractError(
                f"semantic UID fixture for L={length} is not {expected} unique states"
            )
        if families.count("joint") != 250:
            raise ContractError("primary paired joint UID population is not 250")
        roots = [
            _seed32(f"ttbi-state-seed-v1|damage_seed=1|{uid}")
            for uid in uids
        ]
        if length == 60.0:
            expected_first_state = (
                1571465212,
                3550721733,
                412983905,
                2757240308,
                3647310888,
            )
            expected_first_passage = (2623793449, 2345927504)
            actual_first_state = tuple(
                _seed32(
                    f"{schedule}|root={roots[0]}|uid={uids[0]}|stream={name}"
                )
                for name in state_names
            )
            actual_first_passage = tuple(
                _seed32(
                    f"{schedule}|root={roots[0]}|uid={uids[0]}|stream={name}|"
                    "pass=00001"
                )
                for name in passage_names
            )
            if (
                (roots[0], roots[-1]) != (1818075665, 2898326234)
                or actual_first_state != expected_first_state
                or actual_first_passage != expected_first_passage
            ):
                raise ContractError(
                    "independent SHA-256 CRN numeric oracle drifted"
                )
        all_ids = list(roots)
        for uid, root in zip(uids, roots):
            all_ids.extend(
                _seed32(f"{schedule}|root={root}|uid={uid}|stream={name}")
                for name in state_names
            )
            all_ids.extend(
                _seed32(
                    f"{schedule}|root={root}|uid={uid}|stream={name}|"
                    f"pass={passage:05d}"
                )
                for passage in range(1, 51)
                for name in passage_names
            )
        if 0 in all_ids or len(set(all_ids)) != len(all_ids):
            raise ContractError(
                f"named seed fixture for L={length} has zero/collision"
            )


def validate_contract(
    a00: str,
    dataset: str,
    driver: str,
    environment: str = ENVIRONMENT_SOURCE,
    save_progress: str = SAVE_PROGRESS_SOURCE,
    make_micro: str = MAKE_MICRO_SOURCE,
    crn_smoke: str = CRN_SMOKE_SOURCE,
    provenance_smoke: str = PROVENANCE_SMOKE_SOURCE,
) -> None:
    """Raise ContractError unless all R11 generation invariants hold."""
    lock = _validate_environment(environment)
    _validate_helpers()
    _validate_crn_numeric_fixture()

    matlab_schema = _one(
        r"^\s*gen_schema\s*=\s*'([^']+)';", a00, "MATLAB gen_schema"
    )
    _once(
        dataset,
        "_EXPECTED_GEN_SCHEMA = EXPECTED_GEN_SCHEMA",
        "Python expected gen_schema derivation",
    )
    python_schema = EXPECTED_GEN_SCHEMA
    study_tag = _driver_schema_tag(driver)
    behavior_version = _one(
        rf"^\s*{BEHAVIOR_KEY}\s*=\s*'([^']+)';",
        a00,
        "MATLAB generation behavior version",
    )
    qualification_literal = _one(
        r"^\s*qualification_run\s*=\s*(true|false);",
        a00,
        "production qualification literal",
    )
    qualification_source = _one(
        r"^\s*qualification_source_sha256\s*=\s*'([^']+)';",
        a00,
        "qualification source literal",
    )
    max_workers = int(
        _one(
            r"^\s*max_parfor_workers\s*=\s*(\d+);",
            a00,
            "MATLAB generation worker cap",
        )
    )
    crn_integer_literals = {
        "n_states_multi": 250,
        "n_anchor_levels": 5,
        "n_anchor_reps": 5,
        "n_healthy_states": 50,
        "n_nuisance_states": 50,
        "Npass": 50,
        "n_latent_bear": 2,
    }
    for name, expected in crn_integer_literals.items():
        value = int(
            _one(
                rf"^\s*{name}\s*=\s*(\d+);",
                a00,
                f"CRN production literal {name}",
            )
        )
        if value != expected:
            raise ContractError(
                f"{name}={value} differs from CRN design value {expected}"
            )
    for n_targets, expected in ((2, 450), (3, 475)):
        count = (
            crn_integer_literals["n_healthy_states"]
            + n_targets
            * crn_integer_literals["n_anchor_levels"]
            * crn_integer_literals["n_anchor_reps"]
            + 2
            * crn_integer_literals["n_anchor_levels"]
            * crn_integer_literals["n_anchor_reps"]
            + crn_integer_literals["n_nuisance_states"]
            + crn_integer_literals["n_states_multi"]
        )
        if count != expected:
            raise ContractError(
                f"fixed state universe count {count} != {expected}"
            )
    stream_schedule = _one(
        r"^\s*random_stream_schedule_version\s*=\s*'([^']+)';",
        a00,
        "named RNG stream schedule",
    )
    if stream_schedule != "uid-named-substreams-v2":
        raise ContractError("named RNG stream schedule is not collision-free v2")
    reviewed_eov_literals = {
        "profile_jitter_sd_mm": 0.0,
        "hang_rate_100m": 3.0,
        "ballast_rate_100m": 1.2,
        "pad_p_fail": 0.02,
    }
    for name, expected in reviewed_eov_literals.items():
        value = float(
            _one(
                rf"^\s*{name}\s*=\s*([0-9]+(?:\.[0-9]+)?);",
                a00,
                f"reviewed EOV literal {name}",
            )
        )
        if value != expected:
            raise ContractError(
                f"{name}={value} differs from reviewed value {expected}"
            )
    expected_unsupported_share = reviewed_eov_literals["hang_rate_100m"] * 3 / 167
    if not (0.0538 < expected_unsupported_share < 0.0540):
        raise ContractError("analytic unsupported-sleeper share moved from 5.4%")
    if (
        "analytic expected unsupported-sleeper share is 3*3/167 = 5.4%."
        not in a00
    ):
        raise ContractError("analytic 5.4% unsupported-sleeper derivation is absent")
    for token in (
        "_CAMPAIGN_ENVIRONMENT_LOCK = load_environment_lock("
            "_ENVIRONMENT_LOCK_PATH)",
        "_EXPECTED_MATLAB_ENVIRONMENT = (\n"
            "    _CAMPAIGN_ENVIRONMENT_LOCK['spec']['matlab_environment']\n)",
        "_EXPECTED_MATLAB_ENVIRONMENT_SHA256 = (\n"
            "    _CAMPAIGN_ENVIRONMENT_LOCK['spec']"
            "['matlab_environment_sha256']\n)",
        "_EXPECTED_MATLAB_RELEASE = _EXPECTED_MATLAB_ENVIRONMENT['release']",
        "_EXPECTED_GENERATION_BEHAVIOR_VERSION = (\n"
            "    EXPECTED_GENERATION_BEHAVIOR_VERSION\n)",
    ):
        _once(dataset, token, f"Python authenticated environment derivation {token}")

    if matlab_schema != EXPECTED_SCHEMA or python_schema != matlab_schema:
        raise ContractError(
            f"MATLAB/Python generator schemas diverge: "
            f"{matlab_schema!r}, {python_schema!r}"
        )
    if behavior_version != EXPECTED_BEHAVIOR_VERSION:
        raise ContractError(
            "MATLAB behavior version differs from campaign_contract"
        )
    if study_tag != EXPECTED_STUDY_TAG or not study_tag.endswith("r11"):
        raise ContractError("study tag does not identify the R11 protocol")
    if qualification_literal != "false" or qualification_source != "PRODUCTION":
        raise ContractError("production A00 qualification literals are not immutable")
    if "A00_RELEASE_QUALIFICATION" in a00:
        raise ContractError("environment-variable qualification bypass returned")
    if "validated_matlab_releases" in a00:
        raise ContractError("coarse validated-release allowlist returned")
    if "if qualification_run && (n_states > 64 || Npass > 5)" not in a00:
        raise ContractError("qualification mode is not structurally micro-only")
    qualification_start = a00.index("if qualification_run")
    qualification_end = a00.index(
        "elseif ~strcmp(actual_matlab_environment_sha256", qualification_start
    )
    qualification_block = a00[qualification_start:qualification_end]
    for token in (
        "qualification_script_path_ = mfilename('fullpath');",
        "local_qualification_script_identity(qualification_script_path_, ...",
        "if ~strcmp(executed_qualification_source_sha256_, ...",
        "qualification_source_sha256)",
        "qualification_sha_placeholder_ = ...",
        "qualification_folder_placeholder_ = ...",
    ):
        if token not in qualification_block:
            raise ContractError(
                f"qualification executable self-authentication missing: {token}"
            )
    for token in (
        "fullfile(run_folder, 'qualification_executed.m')",
        "copyfile(qualification_script_path_, ...",
        "local_file_sha256(qualification_evidence_path_)",
        "qualification_executed_file_sha256_",
    ):
        if token not in a00:
            raise ContractError(
                f"qualification executable evidence copy missing: {token}"
            )
    local_function_block = a00[a00.index(
        "function [canonical_sha, raw_sha] = "
        "local_qualification_script_identity("
    ):]
    for token in (
        "fopen(fpath, 'rb')",
        "raw_sha = local_sha256_bytes(bytes);",
        "canonical = local_replace_unique_bytes(bytes, ...",
        "canonical_sha = local_sha256_bytes(canonical);",
        "if numel(matches) ~= 1",
    ):
        if token not in local_function_block:
            raise ContractError(
                f"qualification byte canonicalisation missing: {token}"
            )
    if max_workers != 4:
        raise ContractError("generation worker cap is not reviewed value 4")
    for legacy in LEGACY_KEYS:
        if legacy in a00:
            raise ContractError(f"superseded behavior key returned: {legacy}")

    # Strong common-random-number state design (generation-rules-v6).
    design_start = a00.index("%  Build the damage-state matrix")
    design_end = a00.index("tempo_inicial = datetime", design_start)
    design_block = a00[design_start:design_end]
    for token in (
        "n_latent_bear = 2;",
        "state_identity_version = 'semantic-state-v1';",
        "joint_lhs_design = 'master-scour-plus-two-bearing-v1';",
        "random_stream_schedule_version = 'uid-named-substreams-v2';",
        "state_stream_names = {'operations','crack','profile-state','track','profile-phase'};",
        "passage_stream_names = {'profile-passage','oor-passage'};",
        "n_nuis_here = n_nuisance_states;",
        "levels_b = linspace(bearing_fixity_max / n_anchor_levels, ...",
        "lhs = lhsdesign(n_states_multi, n_tgt + n_latent_bear);",
        "joint_s(:, scour_supports) = lhs(:, 1:n_tgt) * dano_max;",
        "joint_bf = lhs(:, n_tgt+1:n_tgt+n_latent_bear) * bearing_fixity_max;",
        "LatentBearingFixity = [anchors_bf; joint_bf];",
        "StateUID     = [uid_; joint_uid_];",
        "expected_states_ = n_healthy_states + ...",
        "StateSeedID = local_state_seed_ids(StateUID, damage_seed);",
        "local_named_stream_seed_ids(StateSeedID, StateUID, Npass, ...",
        "LatentCrackOn(strcmp(StateFamily, 'nuisance_only')) = true;",
        "CrackOn = logical(use_crack_eov) & LatentCrackOn;",
        "BearingFixity = LatentBearingFixity;",
    ):
        if token not in design_block:
            raise ContractError(f"strong CRN state design missing: {token}")
    for forbidden in (
        "n_nuisance_states * double(use_crack_eov)",
        "lhsdesign(n_states_multi, n_tgt + n_bear)",
        "rng(damage_seed + 424243)",
        "CrackOn(is_joint_) = rand(",
    ):
        if forbidden in design_block:
            raise ContractError(f"row/toggle-dependent state design returned: {forbidden}")
    if len(re.findall(r"^\s*if include_anchors\s*$", design_block, re.M)) != 1:
        raise ContractError("the complete anchor inventory became conditional")
    anchor_inventory = design_block[
        design_block.index("n_nuis_here = n_nuisance_states;"):
        design_block.index("lhs = lhsdesign(", design_block.index(
            "n_nuis_here = n_nuisance_states;"
        ))
    ]
    anchor_code = re.sub(r"%.*$", "", anchor_inventory, flags=re.M)
    for forbidden in ("use_crack_eov", "bearing_mode", "use_track_eov"):
        if forbidden in anchor_code:
            raise ContractError(
                f"latent anchor inventory depends on mechanism toggle: {forbidden}"
            )
    latent_crack_block = design_block[
        design_block.index("% ---- Per-state crack ACTIVATION"):
        design_block.index("% Bearing state per file")
    ]
    if "rand(" in latent_crack_block or "rng(" in latent_crack_block:
        raise ContractError("latent crack activation is sequential rather than UID-keyed")

    uid_start = a00.index("function uid = local_state_uid(")
    uid_end = a00.index("function ids = local_state_seed_ids(", uid_start)
    uid_block = a00[uid_start:uid_end]
    for token in (
        "ttbi-state-v1|Lmm=%06d|spans=%d|scour=%s|",
        "family=%s|target=%02d|level=%04d|rep=%03d",
        "round(1000 * L_bridge), num_spans, sprintf('%02d', scour_supports), ...",
    ):
        if token not in uid_block:
            raise ContractError(f"semantic UID grammar missing: {token}")

    seed_start = a00.index("function ids = local_state_seed_ids(")
    seed_end = a00.index("function [state_seeds, passage_seeds]", seed_start)
    seed_block = a00[seed_start:seed_end]
    for token in (
        "'ttbi-state-seed-v1|damage_seed=%.0f|%s'",
        "damage_seed, state_uids{k}));",
        "ids(k) = uint32(hex2dec(h(1:8)));",
        "if any(ids == 0) || numel(unique(ids)) ~= numel(ids)",
    ):
        if token not in seed_block:
            raise ContractError(f"stable root StateSeedID guard missing: {token}")

    named_start = a00.index("function [state_seeds, passage_seeds]")
    named_end = a00.index("function u = local_state_uniform(", named_start)
    named_block = a00[named_start:named_end]
    for token in (
        "schedule_version, state_seed_ids(i_), state_uids{i_}, ...",
        "passage_names{stream_}, pass_",
        "all_ids_ = [state_seed_ids(:); state_seeds(:); passage_seeds(:)];",
        "if any(all_ids_ == 0) || numel(unique(all_ids_)) ~= numel(all_ids_)",
        "seed = uint32(hex2dec(h(1:8)));",
    ):
        if token not in named_block:
            raise ContractError(f"named RNG substream guard missing: {token}")

    generation_rng_start = a00.index(
        "% Reproducible semantic-state RNG stream"
    )
    generation_rng_end = a00.index("% Data Processing", generation_rng_start)
    generation_rng = a00[generation_rng_start:generation_rng_end]
    for token in (
        "rng(double(StateNamedStreamSeedID(DC, 1)), 'twister'); % operations",
        "rng(double(StateNamedStreamSeedID(DC, 2)), 'twister'); % crack",
        "rng(double(PassageNamedStreamSeedID(DC, j_pass, 1)), ...",
        "rng(double(StateNamedStreamSeedID(DC, 3)), ...",
        "Profile_cfg.phase_seed  = double(StateNamedStreamSeedID(DC, 5));",
        "rng(double(StateNamedStreamSeedID(DC, 4)), 'twister'); % track",
        "rng(double(PassageNamedStreamSeedID(DC, j_pass, 2)), ...",
    ):
        if token not in generation_rng:
            raise ContractError(f"named RNG use missing: {token}")
    for forbidden in (
        "damage_seed * 100000 + DC",
        "damage_seed*100000 + DC",
        "1e9 + damage_seed",
        "rng(double(StateSeedID(DC))",
        "phase_seed  = DC",
        "phase_seed = DC",
    ):
        if forbidden in generation_rng:
            raise ContractError(f"DC/sequential RNG seed returned: {forbidden}")

    for token in (
        "size(off60.DamageStates, 1) == 450",
        "size(on99.DamageStates, 1) == 475",
        "isequal(off60.StateUID, on60.StateUID)",
        "isequal(off60.StateSeedID, on60.StateSeedID)",
        "isequal(off60.DamageStates, on60.DamageStates)",
        "isequal(off60.LatentBearingFixity, on60.LatentBearingFixity)",
        "isequal(off60.LatentCrackOn, on60.LatentCrackOn)",
        "uint32([1818075665 2898326234])",
        "uint32([1571465212 3550721733 412983905 2757240308 3647310888])",
        "uint32([2623793449 2345927504])",
        "local_namespace_draws(seed_row_, pass_seed_, 1000)",
        "isequal(base_.track, mutated_.track)",
        "isequal(base_.oor, mutated_.oor)",
        "schedule = 'uid-named-substreams-v2';",
    ):
        if token not in crn_smoke:
            raise ContractError(f"focused MATLAB CRN smoke missing: {token}")

    for token in (
        "'state_uid', ...",
        "'state_seed_id', uint32(123456789), ...",
        "'random_stream_schedule_version', 'uid-named-substreams-v2', ...",
        '"file_state_uid", "file_state_seed_id", ...',
        '"file_random_stream_schedule_version", ...',
        "'file_state_uid', 'state_uid'; ...",
        "'file_random_stream_schedule_version', ...",
        "isa(saved.file_state_seed_id, 'uint32')",
        "bad.state_uid = '';",
        "bad.state_seed_id = uint32(0);",
        "bad.random_stream_schedule_version = ...",
        "(15 mutations rejected; temp cleaned)",
    ):
        if token not in provenance_smoke:
            raise ContractError(
                f"R11 state-serialization smoke missing CRN guard: {token}"
            )

    micro_patches = (
        ("n_states_multi   = 250;", "n_states_multi   = 10;"),
        ("Npass = 50;", "Npass = 3;"),
        ("n_healthy_states  = 50;", "n_healthy_states  = 3;"),
        ("n_anchor_levels  = 5;", "n_anchor_levels  = 2;"),
        ("n_anchor_reps     = 5;", "n_anchor_reps     = 2;"),
        ("n_nuisance_states = 50;", "n_nuisance_states = 6;"),
    )
    for production, micro in micro_patches:
        if make_micro.count(production) != 1 or make_micro.count(micro) != 1:
            raise ContractError(
                f"guarded qualification patch missing/ambiguous: "
                f"{production} -> {micro}"
            )

    for token in (
        "current_working_dir_ = local_canonical_execution_path(pwd);",
        "expected_working_dir_ = local_canonical_execution_path(script_dir_);",
        "if ~strcmp(current_working_dir_, expected_working_dir_)",
        "error('A00:WorkingDirectory', ...",
        "file_ = javaObject('java.io.File', char(raw_path));",
        "normalized = char(file_.getCanonicalPath());",
        r"normalized = strrep(normalized, '\', '/');",
        "is_drive_root_ = ~isempty(regexp(normalized, '^[A-Za-z]:/$', 'once'));",
        "if ispc",
        "normalized = lower(normalized);",
        "environment_lock_ = jsondecode(fileread(environment_lock_path_));",
        "'ttbi-campaign-environment-v2'",
        "campaign_matlab_environment = environment_lock_.matlab_environment;",
        "matlab_environment_identity(campaign_matlab_environment);",
        "actual_matlab_environment = current_matlab_environment();",
        "matlab_environment_identity(actual_matlab_environment);",
        "~strcmp(locked_matlab_environment_sha256_, ...",
        "campaign_matlab_environment_sha256)",
        "[generator_source_root_sha256, generator_source_digest_lines, ...",
        "generator_source_file_count] = generator_source_root();",
        "qualification_host_receipt_ = local_qualification_host_receipt(",
        "local_write_qualification_host_receipt( ...",
        "fullfile(run_folder, 'qualification_host_receipt.json')",
        "declared_host_id_ = strtrim(getenv('TTBI_QUALIFICATION_HOST_ID'));",
        "if isempty(regexp(declared_host_id_, ...",
        "'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'",
        "'schema=ttbi-matlab-qualification-host-v1'",
        "'host_diagnostic_sha256', local_sha256(descriptor_)",
        "if ~isequal(observed_bytes_, expected_bytes_)",
        "error('A00:QualificationHostReceiptCollision', ...",
        "[moved_, move_message_] = movefile(temp_path_, path, 'f');",
    ):
        if token not in a00:
            raise ContractError(f"MATLAB environment/source gate missing: {token}")

    cwd_guard = "if ~strcmp(current_working_dir_, expected_working_dir_)"
    _once(a00, cwd_guard, "canonical working-directory guard")
    if not (
        a00.index("script_dir_ = fileparts(mfilename('fullpath'));")
        < a00.index(cwd_guard)
        < a00.index("environment_lock_ = jsondecode")
        < a00.index("generator_source_file_count] = generator_source_root();")
    ):
        raise ContractError(
            "working-directory guard must precede environment/source execution"
        )

    exact_environment_gate = (
        "elseif ~strcmp(actual_matlab_environment_sha256, ...\n"
        "        campaign_matlab_environment_sha256)"
    )
    _once(a00, exact_environment_gate, "production exact-environment gate")
    if a00.index("if qualification_run") > a00.index(exact_environment_gate):
        raise ContractError("qualification does not exclusively bypass env equality")
    if a00.index("environment_lock_ = jsondecode") > a00.index(
        "if qualification_run"
    ):
        raise ContractError("qualification bypasses environment-lock validation")
    if a00.index("generator_source_file_count] = generator_source_root();") < a00.index(
        exact_environment_gate
    ):
        raise ContractError("source identity call is unexpectedly inside env gate")

    definition_at = a00.index(f"{BEHAVIOR_KEY} =")
    fp_start = a00.index("fp_cfg = struct(")
    fp_end = a00.index("fp_cfg.DamageStates", fp_start)
    fp_block = a00[fp_start:fp_end]
    fp_binding = f"'{BEHAVIOR_KEY}', {BEHAVIOR_KEY}"
    if definition_at > fp_start:
        raise ContractError("behavior version is defined after fp_cfg")
    for binding in (
        fp_binding,
        "'schema', gen_schema",
        "'campaign_matlab_release', ['R' campaign_matlab_release]",
        "'campaign_matlab_environment_sha256', ...",
        "campaign_matlab_environment_sha256, ...",
        "'generator_source_root_sha256', generator_source_root_sha256",
        "'qualification_source_sha256', qualification_source_sha256",
        "'max_parfor_workers', max_parfor_workers",
        "'state_identity_version', state_identity_version",
        "'joint_lhs_design', joint_lhs_design",
        "'n_latent_bearing_dims', n_latent_bear",
        "'random_stream_schedule_version', random_stream_schedule_version",
        "'state_stream_names', {state_stream_names}",
        "'passage_stream_names', {passage_stream_names}",
    ):
        if fp_block.count(binding) != 1:
            raise ContractError(f"fp_cfg binding missing/ambiguous: {binding}")
    for forbidden in (
        "actual_matlab_environment_sha256",
        "actual_matlab_environment_descriptor",
        "fp_cfg.matlab_release",
        "qualification_host",
        "declared_host_id",
        "cpu_identifier",
        "logical_processors",
    ):
        if forbidden in fp_block:
            raise ContractError(
                f"actual executable environment entered comparison fingerprint: "
                f"{forbidden}"
            )

    fp_assignment_end = a00.index(
        "generation_config_json = jsonencode(fp_cfg);", fp_end
    )
    fp_assignment_block = a00[fp_end:fp_assignment_end]
    for field in (
        "DamageStates",
        "BearingStates",
        "BearingFixity",
        "StateFamily",
        "AnchorTarget",
        "AnchorLevel",
        "StateUID",
        "StateSeedID",
        "StateNamedStreamSeedID",
        "PassageNamedStreamSeedIDFlat",
        "LatentBearingFixity",
        "LatentCrackOn",
        "CrackOn",
    ):
        _once(
            fp_assignment_block,
            f"fp_cfg.{field}",
            f"fingerprinted CRN field {field}",
        )

    manifest_start = a00.index("case_info = struct(")
    manifest_end = a00.index(
        "save(fullfile(run_folder, 'case_info.mat')", manifest_start
    )
    manifest_block = a00[manifest_start:manifest_end]
    for binding in (
        fp_binding,
        "'matlab_release', ['R' matlab_release]",
        "'campaign_matlab_release', ['R' campaign_matlab_release]",
        "'actual_matlab_environment_descriptor', ...",
        "actual_matlab_environment_descriptor, ...",
        "'actual_matlab_environment_sha256', ...",
        "actual_matlab_environment_sha256, ...",
        "'campaign_matlab_environment_descriptor', ...",
        "campaign_matlab_environment_descriptor, ...",
        "'campaign_matlab_environment_sha256', ...",
        "campaign_matlab_environment_sha256, ...",
        "'generator_source_root_sha256', generator_source_root_sha256",
        "'generator_source_digest_lines', generator_source_digest_lines",
        "'generator_source_file_count', generator_source_file_count",
        "'qualification_source_sha256', qualification_source_sha256",
        "'release_qualification_run', qualification_run",
        "'max_parfor_workers', max_parfor_workers",
        "'state_identity_version', state_identity_version",
        "'joint_lhs_design', joint_lhs_design",
        "'n_latent_bearing_dims', n_latent_bear",
        "'random_stream_schedule_version', random_stream_schedule_version",
        "'state_stream_names', strjoin(state_stream_names, ',')",
        "'passage_stream_names', strjoin(passage_stream_names, ',')",
    ):
        if manifest_block.count(binding) != 1:
            raise ContractError(f"case_info binding missing/ambiguous: {binding}")

    damage_save_start = a00.index(
        "save(fullfile(run_folder, 'damage_states.mat')", manifest_end
    )
    damage_save_end = a00.index(");", damage_save_start) + 2
    damage_save_block = a00[damage_save_start:damage_save_end]
    for field in (
        "DamageStates",
        "BearingStates",
        "BearingFixity",
        "LatentBearingFixity",
        "StateFamily",
        "AnchorTarget",
        "AnchorLevel",
        "StateUID",
        "StateSeedID",
        "StateNamedStreamSeedID",
        "PassageNamedStreamSeedID",
        "PassageNamedStreamSeedIDFlat",
        "random_stream_schedule_version",
        "state_stream_names",
        "passage_stream_names",
        "LatentCrackOn",
        "CrackOn",
    ):
        if f"'{field}'" not in damage_save_block:
            raise ContractError(f"damage_states.mat omits CRN field {field}")

    resume_block = a00[a00.index("if exist(existing_ci_, 'file')"):manifest_start]
    for token in (
        "~strcmp(prev_release_, ['R' matlab_release])",
        "~strcmp(prev_campaign_release_, ['R' campaign_matlab_release])",
        "prev_qualification_ ~= qualification_run",
        "~strcmp(prev_actual_env_descriptor_, ...",
        "actual_matlab_environment_descriptor)",
        "~strcmp(prev_actual_env_sha_, ...",
        "actual_matlab_environment_sha256)",
        "~strcmp(prev_campaign_env_descriptor_, ...",
        "campaign_matlab_environment_descriptor)",
        "~strcmp(prev_campaign_env_sha_, ...",
        "campaign_matlab_environment_sha256)",
        "~strcmp(prev_generator_source_root_, ...",
        "generator_source_root_sha256)",
        "~strcmp(prev_generator_source_lines_, ...",
        "generator_source_digest_lines)",
        "prev_generator_source_count_ ~= generator_source_file_count",
        "~strcmp(prev_qualification_source_, qualification_source_sha256)",
    ):
        if token not in resume_block:
            raise ContractError(f"case_info resume invariant missing: {token}")

    state_resume_start = a00.index("saved_files = dir(")
    state_write_start = a00.index("data2save.gen_schema")
    state_resume = a00[state_resume_start:state_write_start]
    for field in (
        "file_state_uid",
        "file_state_seed_id",
        "file_random_stream_schedule_version",
        "file_actual_matlab_environment_sha256",
        "file_campaign_matlab_environment_sha256",
        "file_generator_source_root_sha256",
        "file_qualification_source_sha256",
    ):
        if state_resume.count(f"'{field}'") < 2:
            raise ContractError(f"state resume does not inventory/load {field}")
    for token in (
        "~strcmp(S_.file_state_uid, StateUID{dc_idx})",
        "~isa(S_.file_state_seed_id, 'uint32')",
        "S_.file_state_seed_id ~= StateSeedID(dc_idx)",
        "~strcmp(S_.file_random_stream_schedule_version, ...",
        "~strcmp(S_.file_matlab_release, ['R' matlab_release])",
        "~strcmp(S_.file_campaign_matlab_release, ...",
        "S_.file_release_qualification_run ~= qualification_run",
        "~strcmp(S_.file_actual_matlab_environment_sha256, ...",
        "actual_matlab_environment_sha256)",
        "~strcmp(S_.file_campaign_matlab_environment_sha256, ...",
        "campaign_matlab_environment_sha256)",
        "~strcmp(S_.file_generator_source_root_sha256, ...",
        "generator_source_root_sha256)",
        "~strcmp(S_.file_qualification_source_sha256, ...",
        "qualification_source_sha256)",
        "~strcmp(char(d_.actual_matlab_environment_descriptor), ...",
        "~strcmp(char(d_.actual_matlab_environment_sha256), ...",
        "~strcmp(char(d_.campaign_matlab_environment_descriptor), ...",
        "~strcmp(char(d_.campaign_matlab_environment_sha256), ...",
        "~strcmp(char(d_.generator_source_root_sha256), ...",
        "~strcmp(char(d_.generator_source_digest_lines), ...",
        "d_.generator_source_file_count ~= generator_source_file_count",
        "~strcmp(char(d_.qualification_source_sha256), ...",
    ):
        if token not in state_resume:
            raise ContractError(f"per-state resume invariant missing: {token}")
    for token in (
        "'state_uid','state_seed_id','latent_bearing_fixity'",
        "'latent_crack_on','crack_on','random_stream_schedule_version'",
        "'state_named_stream_seed_id','passage_named_stream_seed_id'",
        "~strcmp(char(d_.state_uid), StateUID{dc_idx})",
        "~isa(d_.state_seed_id, 'uint32')",
        "d_.state_seed_id ~= StateSeedID(dc_idx)",
        "~strcmp(char(d_.random_stream_schedule_version), ...",
        "StateNamedStreamSeedID(dc_idx, :)",
        "PassageNamedStreamSeedID(dc_idx, :, :)",
        "LatentBearingFixity(dc_idx, :)",
        "d_.latent_crack_on ~= LatentCrackOn(dc_idx)",
        "d_.crack_on ~= CrackOn(dc_idx)",
    ):
        if token not in state_resume:
            raise ContractError(f"per-state CRN resume invariant missing: {token}")

    state_end = a00.index("save_progress(data2save", state_write_start)
    state_block = a00[state_write_start:state_end]
    for field in (
        "actual_matlab_environment_descriptor",
        "actual_matlab_environment_sha256",
        "campaign_matlab_environment_descriptor",
        "campaign_matlab_environment_sha256",
        "generator_source_root_sha256",
        "generator_source_digest_lines",
        "generator_source_file_count",
        "qualification_source_sha256",
        "release_qualification_run",
    ):
        _once(
            state_block,
            f"data2save.{field}",
            f"per-state payload provenance {field}",
        )
    full_state_write_start = a00.index("data2save.state_family")
    full_state_block = a00[full_state_write_start:state_end]
    for field in (
        "state_uid",
        "state_seed_id",
        "random_stream_schedule_version",
        "state_named_stream_seed_id",
        "passage_named_stream_seed_id",
        "latent_bearing_fixity",
        "latent_crack_on",
        "crack_on",
    ):
        _once(
            full_state_block,
            f"data2save.{field}",
            f"per-state CRN payload {field}",
        )
    call_end = a00.index(");", state_end) + 2
    call_block = a00[state_end:call_end]
    for token in (
        "['R' matlab_release]",
        "['R' campaign_matlab_release]",
        "qualification_run",
        "actual_matlab_environment_sha256",
        "campaign_matlab_environment_sha256",
        "generator_source_root_sha256",
        "qualification_source_sha256",
    ):
        _once(call_block, token, f"save_progress call argument {token}")

    for token in (
        "narginchk(12, 12);",
        "required_payload = {",
        "'state_uid', 'state_seed_id', ...",
        "'random_stream_schedule_version', ...",
        "state_uid = local_text(data.state_uid, 'data.state_uid');",
        "~isa(data.state_seed_id, 'uint32')",
        "data.state_seed_id == 0",
        "random_stream_schedule_version = local_text( ...",
        "file_state_uid = state_uid;",
        "file_state_seed_id = data.state_seed_id;",
        "file_random_stream_schedule_version = random_stream_schedule_version;",
        "'file_state_uid', 'file_state_seed_id', ...",
        "'file_random_stream_schedule_version', ...",
        "file_actual_matlab_environment_sha256 = ...",
        "file_campaign_matlab_environment_sha256 = ...",
        "file_generator_source_root_sha256 = generator_source_root_sha256;",
        "file_qualification_source_sha256 = qualification_source_sha256;",
        "'file_actual_matlab_environment_sha256', ...",
        "'file_campaign_matlab_environment_sha256', ...",
        "'file_generator_source_root_sha256', ...",
        "'file_qualification_source_sha256');",
        "if ~strcmp(local_text_sha256(actual_descriptor), ...\n"
            "            actual_matlab_environment_sha256)",
        "if ~strcmp(local_text_sha256(campaign_descriptor), ...\n"
            "            campaign_matlab_environment_sha256)",
        "if ~strcmp(local_text_sha256(source_lines), "
            "generator_source_root_sha256)",
        "data.generator_source_file_count ~= source_line_count",
    ):
        if token not in save_progress:
            raise ContractError(f"save_progress provenance contract missing: {token}")
    if re.search(r"\bnargin\s*>=", save_progress):
        raise ContractError("save_progress regained permissive optional provenance")
    for field in (
        "actual_matlab_environment_sha256",
        "campaign_matlab_environment_sha256",
        "generator_source_root_sha256",
        "qualification_source_sha256",
    ):
        if (
            f"'{field}'" not in save_progress
            or save_progress.count(f"data.{field}") != 1
        ):
            raise ContractError(
                f"save_progress does not require and compare data.{field}"
            )

    # Preserve the fail-closed, bounded process-pool contract.
    parallel_start = a00.index("if all(completed)")
    parallel_end = a00.index("% Mark end time of the run", parallel_start)
    parallel_block = a00[parallel_start:parallel_end]
    skip_delimiter = "else\n    pool_ = gcp('nocreate');"
    _once(parallel_block, "if all(completed)", "all-complete resume guard")
    _once(parallel_block, skip_delimiter, "all-complete pool bypass")
    all_complete_branch = parallel_block[: parallel_block.index(skip_delimiter)]
    if any(token in all_complete_branch for token in (
        "gcp(", "parcluster(", "parpool(", "parfor "
    )):
        raise ContractError("all-complete resume branch touches parallel pool")
    for token in (
        "~isa(pool_, 'parallel.ProcessPool')",
        "pool_.NumWorkers > max_parfor_workers",
        "delete(pool_);",
        "cluster_ = parcluster('Processes');",
        "pool_workers_ = min(max_parfor_workers, cluster_.NumWorkers)",
        "pool_ = parpool(cluster_, pool_workers_)",
        "parfor (DC = 1:n_states, pool_workers_)",
    ):
        if token not in parallel_block:
            raise ContractError(f"bounded process-pool invariant missing: {token}")
    if parallel_block.count("~isa(pool_, 'parallel.ProcessPool')") != 2:
        raise ContractError("process-pool type is not checked pre/post setup")
    if parallel_block.count("parcluster(") != 1:
        raise ContractError("generation cluster selection is ambiguous")

    completion_start = a00.index("if present_ == n_states")
    completion_end = a00.index(
        "% =========================================================================\n"
        "%  Local functions",
        completion_start,
    )
    completion_block = a00[completion_start:completion_end]
    if completion_block.count("generator_source_root();") != 2:
        raise ContractError(
            "generator source root must be re-read before digest and marker writes"
        )
    for token in (
        "~strcmp(completion_source_root_, generator_source_root_sha256)",
        "~strcmp(completion_source_lines_, generator_source_digest_lines)",
        "completion_source_count_ ~= generator_source_file_count",
        "~strcmp(marker_source_root_, generator_source_root_sha256)",
        "~strcmp(marker_source_lines_, generator_source_digest_lines)",
        "marker_source_count_ ~= generator_source_file_count",
        "delete(fullfile(run_folder, 'file_digests.mat'));",
    ):
        if token not in completion_block:
            raise ContractError(
                f"end-of-run generator-source gate missing: {token}"
            )
    first_completion_gate = completion_block.index(
        "completion_source_count_ ~= generator_source_file_count"
    )
    digest_write = completion_block.index(
        "save(fullfile(run_folder, 'file_digests.mat')"
    )
    second_completion_gate = completion_block.index(
        "marker_source_count_ ~= generator_source_file_count"
    )
    marker_write = completion_block.index(
        "movefile(tmp_marker_, marker_path, 'f')"
    )
    if not (
        first_completion_gate < digest_write
        < second_completion_gate < marker_write
    ):
        raise ContractError(
            "source-stability gates do not bracket digest/marker writes"
        )

    f1_start = a00.index(
        "% ---- Deck fundamental frequency: SELF-DOCUMENT every dataset ----"
    )
    f1_end = a00.index(
        "% Contact-validity metrics for this passage", f1_start
    )
    f1_block = a00[f1_start:f1_end]
    for token in (
        "beam_f1 = Beam_local.Modal.f(1);",
        "if ~isfinite(beam_f1) || beam_f1 < 0.2 || beam_f1 > 15",
        "if strcmp(StateFamily{DC}, 'target_healthy')",
        "healthy_f1_bounds_ = [3, 6];",
        "healthy_f1_bounds_ = [2, 4];",
        "if beam_f1 < healthy_f1_bounds_(1) || ...",
        "beam_f1 > healthy_f1_bounds_(2)",
        "error(['A00: healthy deck f1 %.6g Hz outside [%g,%g] Hz ' ...",
    ):
        if token not in f1_block:
            raise ContractError(f"deck-f1 sanity gate missing: {token}")
    if "data2save.beam_f1_Hz = beam_f1;" not in a00:
        raise ContractError("deck-f1 attestation is not saved per state")

    config_at = a00.index("generation_config_json = jsonencode(fp_cfg);")
    hash_at = a00.index(
        "gen_fingerprint = local_sha256(generation_config_json);"
    )
    if not (fp_end < config_at < hash_at < manifest_start):
        raise ContractError(
            "fp_cfg canonical JSON is not hashed before case_info is written"
        )


def _must_reject(
    name: str,
    a00: str,
    dataset: str,
    driver: str,
    environment: str = ENVIRONMENT_SOURCE,
    save_progress: str = SAVE_PROGRESS_SOURCE,
    make_micro: str = MAKE_MICRO_SOURCE,
    crn_smoke: str = CRN_SMOKE_SOURCE,
    provenance_smoke: str = PROVENANCE_SMOKE_SOURCE,
) -> None:
    try:
        validate_contract(
            a00,
            dataset,
            driver,
            environment,
            save_progress,
            make_micro,
            crn_smoke,
            provenance_smoke,
        )
    except (ContractError, ValueError):
        print(f"  [PASS] mutation rejected: {name}")
        return
    raise AssertionError(f"mutation escaped generation-contract guard: {name}")


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(
            f"mutation fixture expected exactly one {old!r}, found {text.count(old)}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    a00 = A00_PATH.read_text(encoding="utf-8")
    dataset = DATASET_PATH.read_text(encoding="utf-8")
    driver = DRIVER_PATH.read_text(encoding="utf-8")
    validate_contract(a00, dataset, driver)
    print("GENERATION CONTRACT CHECKS")
    print("  [PASS] live MATLAB/Python R11 environment/source contract")

    mutations: list[
        tuple[str, str, str, str, str, str, str, str, str]
    ] = []

    def add(
        name: str,
        *,
        ma: str = a00,
        py: str = dataset,
        dr: str = driver,
        env: str = ENVIRONMENT_SOURCE,
        save: str = SAVE_PROGRESS_SOURCE,
        micro: str = MAKE_MICRO_SOURCE,
        smoke: str = CRN_SMOKE_SOURCE,
        provenance_smoke: str = PROVENANCE_SMOKE_SOURCE,
    ) -> None:
        mutations.append(
            (name, ma, py, dr, env, save, micro, smoke, provenance_smoke)
        )

    def mutate_fp(old: str, new: str) -> str:
        start = a00.index("fp_cfg = struct(")
        end = a00.index("fp_cfg.DamageStates", start)
        block = a00[start:end]
        if block.count(old) != 1:
            raise AssertionError(
                f"fingerprint mutation expected one {old!r}, "
                f"found {block.count(old)}"
            )
        return a00[:start] + block.replace(old, new, 1) + a00[end:]

    def mutate_save_call(old: str, new: str) -> str:
        start = a00.index("save_progress(data2save")
        end = a00.index(");", start) + 2
        block = a00[start:end]
        if block.count(old) != 1:
            raise AssertionError(
                f"save-call mutation expected one {old!r}, "
                f"found {block.count(old)}"
            )
        return a00[:start] + block.replace(old, new, 1) + a00[end:]

    def mutate_damage_save(old: str, new: str) -> str:
        start = a00.index("save(fullfile(run_folder, 'damage_states.mat')")
        end = a00.index(");", start) + 2
        block = a00[start:end]
        if block.count(old) != 1:
            raise AssertionError(
                f"damage-save mutation expected one {old!r}, "
                f"found {block.count(old)}"
            )
        return a00[:start] + block.replace(old, new, 1) + a00[end:]

    add("MATLAB schema drift", ma=_replace_once(
        a00, EXPECTED_SCHEMA, "audit-2026-07-27-r10"))
    add(
        "Python schema drift",
        py=_replace_once(
            dataset,
            "_EXPECTED_GEN_SCHEMA = EXPECTED_GEN_SCHEMA",
            "_EXPECTED_GEN_SCHEMA = 'audit-2026-07-27-r10'",
        ),
    )
    add(
        "Python behavior version stopped deriving from campaign contract",
        py=_replace_once(
            dataset,
            "_EXPECTED_GENERATION_BEHAVIOR_VERSION = (\n"
            "    EXPECTED_GENERATION_BEHAVIOR_VERSION\n)",
            "_EXPECTED_GENERATION_BEHAVIOR_VERSION = 'generation-rules-v6'",
        ),
    )
    behavior_binding = f"'{BEHAVIOR_KEY}', {BEHAVIOR_KEY}, ..."
    if a00.count(behavior_binding) != 2:
        raise AssertionError("behavior mutation fixture requires two bindings")
    first = a00.index(behavior_binding)
    fingerprint_mutation = (
        a00[:first]
        + "'unrelated_config_key', 1, ..."
        + a00[first + len(behavior_binding):]
    )
    add("behavior removed from fingerprint", ma=fingerprint_mutation)
    second = a00.index(behavior_binding, first + 1)
    manifest_mutation = (
        a00[:second]
        + "'kept_in_fingerprint_only', 1, ..."
        + a00[second + len(behavior_binding):]
    )
    add("behavior removed from case_info", ma=manifest_mutation)
    add(
        "legacy behavior key returned",
        ma=a00.replace(
            "if use_track_eov",
            "track_eov_impl = 'legacy';\nif use_track_eov",
            1,
        ),
    )
    add("behavior version drift", ma=_replace_once(
        a00,
        f"{BEHAVIOR_KEY} = '{EXPECTED_BEHAVIOR_VERSION}';",
        f"{BEHAVIOR_KEY} = 'generation-rules-v3';",
    ))
    for literal, drifted, label in (
        ("n_states_multi   = 250;", "n_states_multi   = 249;", "joint states"),
        ("Npass = 50;", "Npass = 49;", "passages"),
        ("n_anchor_levels  = 5;", "n_anchor_levels  = 4;", "anchor levels"),
        ("n_anchor_reps     = 5;", "n_anchor_reps     = 4;", "anchor replicas"),
        ("n_healthy_states  = 50;", "n_healthy_states  = 49;", "healthy states"),
        ("n_nuisance_states = 50;", "n_nuisance_states = 49;", "nuisance states"),
    ):
        add(
            f"fixed CRN production count drift: {label}",
            ma=_replace_once(a00, literal, drifted),
        )
    add(
        "master LHS dimensionality made bearing-toggle dependent",
        ma=_replace_once(
            a00,
            "lhs = lhsdesign(n_states_multi, n_tgt + n_latent_bear);",
            "lhs = lhsdesign(n_states_multi, n_tgt + "
            "double(strcmp(bearing_mode, 'target')) * n_latent_bear);",
        ),
    )
    add(
        "master scour columns shifted by one",
        ma=_replace_once(
            a00,
            "joint_s(:, scour_supports) = lhs(:, 1:n_tgt) * dano_max;",
            "joint_s(:, scour_supports) = lhs(:, 2:n_tgt+1) * dano_max;",
        ),
    )
    add(
        "nuisance inventory conditioned on crack toggle",
        ma=_replace_once(
            a00,
            "n_nuis_here = n_nuisance_states;",
            "n_nuis_here = n_nuisance_states * double(use_crack_eov);",
        ),
    )
    add(
        "bearing anchor inventory conditioned on bearing toggle",
        ma=_replace_once(
            a00,
            "for bi = 1:2",
            "for bi = 1:(2 * double(strcmp(bearing_mode, 'target')))",
        ),
    )
    add(
        "semantic state UID made row-dependent",
        ma=_replace_once(
            a00,
            "round(1000 * L_bridge), num_spans, sprintf('%02d', scour_supports), ...",
            "round(1000 * L_bridge) + DC, num_spans, "
            "sprintf('%02d', scour_supports), ...",
        ),
    )
    add(
        "root state seed made loop-order dependent",
        ma=_replace_once(
            a00,
            "damage_seed, state_uids{k}));",
            "damage_seed + k, state_uids{k}));",
        ),
    )
    add(
        "root state seed zero/collision gate removed",
        ma=_replace_once(
            a00,
            "if any(ids == 0) || numel(unique(ids)) ~= numel(ids)",
            "if false",
        ),
    )
    add(
        "latent crack activation made sequential",
        ma=_replace_once(
            a00,
            "local_state_uniform(StateUID{state_}, ...\n"
            "        damage_seed, 'latent-crack-v1') <= crack_p;",
            "rand() <= crack_p;",
        ),
    )
    add(
        "crack mechanism toggle bypassed",
        ma=_replace_once(
            a00,
            "CrackOn = logical(use_crack_eov) & LatentCrackOn;",
            "CrackOn = LatentCrackOn;",
        ),
    )
    add(
        "named RNG schedule reverted to colliding v1",
        ma=_replace_once(
            a00,
            "random_stream_schedule_version = 'uid-named-substreams-v2';",
            "random_stream_schedule_version = 'uid-named-substreams-v1';",
        ),
    )
    add(
        "named RNG zero/collision gate removed",
        ma=_replace_once(
            a00,
            "if any(all_ids_ == 0) || "
            "numel(unique(all_ids_)) ~= numel(all_ids_)",
            "if false",
        ),
    )
    add(
        "state stream names expand fingerprint struct",
        ma=_replace_once(
            a00,
            "'state_stream_names', {state_stream_names}",
            "'state_stream_names', state_stream_names",
        ),
    )
    add(
        "passage stream names expand fingerprint struct",
        ma=_replace_once(
            a00,
            "'passage_stream_names', {passage_stream_names}",
            "'passage_stream_names', passage_stream_names",
        ),
    )
    add(
        "operations draw uses crack namespace",
        ma=_replace_once(
            a00,
            "rng(double(StateNamedStreamSeedID(DC, 1)), 'twister'); % operations",
            "rng(double(StateNamedStreamSeedID(DC, 2)), 'twister'); % operations",
        ),
    )
    add(
        "profile phase uses profile-class namespace",
        ma=_replace_once(
            a00,
            "Profile_cfg.phase_seed  = double(StateNamedStreamSeedID(DC, 5));",
            "Profile_cfg.phase_seed  = double(StateNamedStreamSeedID(DC, 3));",
        ),
    )
    add(
        "per-passage OOR uses profile-passage namespace",
        ma=_replace_once(
            a00,
            "rng(double(PassageNamedStreamSeedID(DC, j_pass, 2)), ...",
            "rng(double(PassageNamedStreamSeedID(DC, j_pass, 1)), ...",
        ),
    )
    add(
        "semantic UID removed from comparison fingerprint",
        ma=_replace_once(
            a00,
            "fp_cfg.StateUID      = StateUID;",
            "fp_cfg.UnrelatedUID  = StateUID;",
        ),
    )
    add(
        "named passage seeds removed from damage-state manifest",
        ma=mutate_damage_save(
            "'PassageNamedStreamSeedID', ...",
            "'UnrelatedPassageSeedID', ...",
        ),
    )
    add(
        "state UID removed from per-state payload",
        ma=_replace_once(
            a00,
            "data2save.state_uid = StateUID{DC};",
            "data2save.unrelated_uid = StateUID{DC};",
        ),
    )
    add(
        "cheap state-identity resume inventory removed",
        ma=_replace_once(
            a00,
            "'file_state_uid','file_state_seed_id', ...",
            "'unrelated_state_uid','file_state_seed_id', ...",
        ),
    )
    add(
        "Python study schema tag stopped deriving from campaign contract",
        dr=_replace_once(
            driver,
            "SCHEMA_TAG = EXPECTED_PROTOCOL_SCHEMA_TAG",
            f'SCHEMA_TAG = "{EXPECTED_PROTOCOL_SCHEMA_TAG}"',
        ),
    )
    add(
        "Python study schema tag gained a later runtime rebind",
        dr=_replace_once(
            driver,
            "SCHEMA_TAG = EXPECTED_PROTOCOL_SCHEMA_TAG",
            "SCHEMA_TAG = EXPECTED_PROTOCOL_SCHEMA_TAG\n"
            'SCHEMA_TAG = "reviewer-bypass"',
        ),
    )
    add(
        "Python study schema tag was rebound by a module definition",
        dr=_replace_once(
            driver,
            "SCHEMA_TAG = EXPECTED_PROTOCOL_SCHEMA_TAG",
            "SCHEMA_TAG = EXPECTED_PROTOCOL_SCHEMA_TAG\n"
            "def SCHEMA_TAG():\n"
            '    return "reviewer-bypass"',
        ),
    )
    add("production qualification enabled", ma=_replace_once(
        a00, "qualification_run = false;", "qualification_run = true;"))
    add("working-directory guard removed", ma=_replace_once(
        a00,
        "if ~strcmp(current_working_dir_, expected_working_dir_)",
        "if false",
    ))
    add("production qualification source forged", ma=_replace_once(
        a00,
        "qualification_source_sha256 = 'PRODUCTION';",
        f"qualification_source_sha256 = '{'b' * 64}';",
    ))
    add("qualification executable self-check laundered", ma=_replace_once(
        a00,
        "if ~strcmp(executed_qualification_source_sha256_, ...\n"
        "            qualification_source_sha256)",
        "if false",
    ))
    add("qualification canonicalisation uniqueness removed", ma=_replace_once(
        a00,
        "if numel(matches) ~= 1",
        "if false",
    ))
    add("qualification evidence copies another file", ma=_replace_once(
        a00,
        "copyfile(qualification_script_path_, ...",
        "copyfile(environment_lock_path_, ...",
    ))
    add("qualification host-ID gate removed", ma=_replace_once(
        a00,
        "if isempty(regexp(declared_host_id_, ...",
        "if false && isempty(regexp(declared_host_id_, ...",
    ))
    add("qualification host receipt not persisted", ma=_replace_once(
        a00,
        "local_write_qualification_host_receipt( ...",
        "local_ignore_qualification_host_receipt( ...",
    ))
    add("qualification host collision guard removed", ma=_replace_once(
        a00,
        "if ~isequal(observed_bytes_, expected_bytes_)",
        "if false",
    ))
    add("qualification host digest unbound", ma=_replace_once(
        a00,
        "'host_diagnostic_sha256', local_sha256(descriptor_)",
        "'host_diagnostic_sha256', local_sha256(declared_host_id_)",
    ))
    add("profile jitter re-enabled", ma=_replace_once(
        a00, "profile_jitter_sd_mm = 0;", "profile_jitter_sd_mm = 0.5;"))
    add("hanging-sleeper rate drift", ma=_replace_once(
        a00, "hang_rate_100m    = 3.0;", "hang_rate_100m    = 2.5;"))
    add("ballast-patch rate drift", ma=_replace_once(
        a00, "ballast_rate_100m = 1.2;", "ballast_rate_100m = 2.0;"))
    add("pad-failure prevalence drift", ma=_replace_once(
        a00, "pad_p_fail        = 0.02;", "pad_p_fail        = 0.10;"))
    add("analytic unsupported-share statement removed", ma=_replace_once(
        a00,
        "analytic expected unsupported-sleeper share is 3*3/167 = 5.4%.",
        "unsupported-sleeper share was allegedly Monte Carlo verified.",
    ))

    env_obj = json.loads(ENVIRONMENT_SOURCE)
    env_obj["matlab_environment"]["version"] = "25.2.0.FORGED"
    add("environment descriptor changed without digest", env=json.dumps(env_obj))
    env_obj = json.loads(ENVIRONMENT_SOURCE)
    env_obj["matlab_environment_sha256"] = "b" * 64
    add("environment digest forged", env=json.dumps(env_obj))
    env_obj = json.loads(ENVIRONMENT_SOURCE)
    env_obj["matlab_release"] = "R2025b"
    add("coarse release key restored", env=json.dumps(env_obj))

    gate = (
        "elseif ~strcmp(actual_matlab_environment_sha256, ...\n"
        "        campaign_matlab_environment_sha256)"
    )
    add("actual-vs-campaign environment gate removed", ma=_replace_once(
        a00, gate, "elseif false"))
    add("actual-vs-campaign environment gate inverted", ma=_replace_once(
        a00, gate,
        "elseif strcmp(actual_matlab_environment_sha256, ...\n"
        "        campaign_matlab_environment_sha256)",
    ))
    add("environment gate became self-comparison", ma=_replace_once(
        a00, gate,
        "elseif ~strcmp(actual_matlab_environment_sha256, ...\n"
        "        actual_matlab_environment_sha256)",
    ))
    add("source root removed from fingerprint", ma=mutate_fp(
        "'generator_source_root_sha256', generator_source_root_sha256, ...",
        "'unrelated_source', generator_source_root_sha256, ...",
    ))
    add("campaign environment removed from fingerprint", ma=mutate_fp(
        "'campaign_matlab_environment_sha256', ...",
        "'unrelated_environment_sha256', ...",
    ))
    add("qualification source removed from fingerprint", ma=mutate_fp(
        "'qualification_source_sha256', qualification_source_sha256, ...",
        "'unrelated_qualification_source', qualification_source_sha256, ...",
    ))
    add("actual environment improperly added to comparison fingerprint",
        ma=mutate_fp(
            "'campaign_matlab_environment_sha256', ...",
            "'actual_matlab_environment_sha256', ...",
        ))

    add("manifest actual environment self-comparison", ma=_replace_once(
        a00,
        "~strcmp(prev_actual_env_sha_, ...\n"
        "                actual_matlab_environment_sha256)",
        "~strcmp(actual_matlab_environment_sha256, ...\n"
        "                actual_matlab_environment_sha256)",
    ))
    add("manifest campaign environment self-comparison", ma=_replace_once(
        a00,
        "~strcmp(prev_campaign_env_sha_, ...\n"
        "                campaign_matlab_environment_sha256)",
        "~strcmp(campaign_matlab_environment_sha256, ...\n"
        "                campaign_matlab_environment_sha256)",
    ))
    add("manifest source root self-comparison", ma=_replace_once(
        a00,
        "~strcmp(prev_generator_source_root_, ...\n"
        "                generator_source_root_sha256)",
        "~strcmp(generator_source_root_sha256, ...\n"
        "                generator_source_root_sha256)",
    ))
    add("manifest mode self-comparison", ma=_replace_once(
        a00,
        "prev_qualification_ ~= qualification_run",
        "qualification_run ~= qualification_run",
    ))
    add("manifest release self-comparison", ma=_replace_once(
        a00,
        "~strcmp(prev_release_, ['R' matlab_release])",
        "~strcmp(['R' matlab_release], ['R' matlab_release])",
    ))

    add("state actual environment self-comparison", ma=_replace_once(
        a00,
        "~strcmp(S_.file_actual_matlab_environment_sha256, ...\n"
        "                actual_matlab_environment_sha256)",
        "~strcmp(actual_matlab_environment_sha256, ...\n"
        "                actual_matlab_environment_sha256)",
    ))
    add("state campaign environment self-comparison", ma=_replace_once(
        a00,
        "~strcmp(S_.file_campaign_matlab_environment_sha256, ...\n"
        "                campaign_matlab_environment_sha256)",
        "~strcmp(campaign_matlab_environment_sha256, ...\n"
        "                campaign_matlab_environment_sha256)",
    ))
    add("state source root self-comparison", ma=_replace_once(
        a00,
        "~strcmp(S_.file_generator_source_root_sha256, ...\n"
        "                generator_source_root_sha256)",
        "~strcmp(generator_source_root_sha256, ...\n"
        "                generator_source_root_sha256)",
    ))
    add("state mode self-comparison", ma=_replace_once(
        a00,
        "S_.file_release_qualification_run ~= qualification_run",
        "qualification_run ~= qualification_run",
    ))
    add("state release self-comparison", ma=_replace_once(
        a00,
        "~strcmp(S_.file_matlab_release, ['R' matlab_release])",
        "~strcmp(['R' matlab_release], ['R' matlab_release])",
    ))

    add("top-level source root omitted from save", save=_replace_once(
        SAVE_PROGRESS_SOURCE,
        "'file_generator_source_root_sha256', ...",
        "'unrelated_source_root', ...",
    ))
    add("top-level semantic UID omitted from save", save=_replace_once(
        SAVE_PROGRESS_SOURCE,
        "'file_state_uid', 'file_state_seed_id', ...",
        "'unrelated_state_uid', 'file_state_seed_id', ...",
    ))
    add("top-level RNG schedule omitted from save", save=_replace_once(
        SAVE_PROGRESS_SOURCE,
        "'file_random_stream_schedule_version', ...",
        "'unrelated_stream_schedule_version', ...",
    ))
    add("state seed type/nonzero guard removed from save", save=_replace_once(
        SAVE_PROGRESS_SOURCE,
        "if ~isa(data.state_seed_id, 'uint32') || ...\n"
        "            ~isscalar(data.state_seed_id) || data.state_seed_id == 0",
        "if false",
    ))
    add("save_progress regained optional provenance", save=_replace_once(
        SAVE_PROGRESS_SOURCE, "narginchk(12, 12);", "if nargin < 12; return; end"))
    add("save_progress stopped authenticating environment descriptor",
        save=_replace_once(
            SAVE_PROGRESS_SOURCE,
            "if ~strcmp(local_text_sha256(actual_descriptor), ...\n"
            "            actual_matlab_environment_sha256)",
            "if false",
        ))
    add("save_progress stopped authenticating source descriptor",
        save=_replace_once(
            SAVE_PROGRESS_SOURCE,
            "if ~strcmp(local_text_sha256(source_lines), "
            "generator_source_root_sha256)",
            "if false",
        ))
    add("save call omits source root", ma=mutate_save_call(
        "campaign_matlab_environment_sha256, ...\n"
        "            generator_source_root_sha256, qualification_source_sha256);",
        "campaign_matlab_environment_sha256, ...\n"
        "            qualification_source_sha256, qualification_source_sha256);",
    ))

    add("unsafe automatic worker count restored", ma=_replace_once(
        a00, "max_parfor_workers = 4;", "max_parfor_workers = 16;"))
    add("parfor worker cap bypassed", ma=_replace_once(
        a00,
        "parfor (DC = 1:n_states, pool_workers_)",
        "parfor DC = 1:n_states",
    ))
    add("default cluster profile restored", ma=_replace_once(
        a00, "cluster_ = parcluster('Processes');", "cluster_ = parcluster();"))
    add("thread-pool profile requested", ma=_replace_once(
        a00,
        "cluster_ = parcluster('Processes');",
        "cluster_ = parcluster('Threads');",
    ))
    add("unsafe existing pool not deleted", ma=_replace_once(
        a00, "        delete(pool_);\n", ""))
    add("all-complete resume guard removed", ma=_replace_once(
        a00, "if all(completed)", "if false"))
    add("pre-digest source-root gate laundered", ma=_replace_once(
        a00,
        "~strcmp(completion_source_root_, generator_source_root_sha256)",
        "~strcmp(generator_source_root_sha256, generator_source_root_sha256)",
    ))
    add("pre-marker source-lines gate laundered", ma=_replace_once(
        a00,
        "~strcmp(marker_source_lines_, generator_source_digest_lines)",
        "~strcmp(generator_source_digest_lines, generator_source_digest_lines)",
    ))
    add("global deck-f1 gate removed", ma=_replace_once(
        a00,
        "if ~isfinite(beam_f1) || beam_f1 < 0.2 || beam_f1 > 15",
        "if false",
    ))
    add("healthy deck-f1 family gate laundered", ma=_replace_once(
        a00,
        "if strcmp(StateFamily{DC}, 'target_healthy')",
        "if true",
    ))
    add("healthy deck-f1 bounds removed", ma=_replace_once(
        a00,
        "if beam_f1 < healthy_f1_bounds_(1) || ...\n"
        "                        beam_f1 > healthy_f1_bounds_(2)",
        "if false",
    ))

    for production, micro_value, label in (
        ("n_states_multi   = 250;", "n_states_multi   = 10;", "joint states"),
        ("Npass = 50;", "Npass = 3;", "passages"),
        ("n_healthy_states  = 50;", "n_healthy_states  = 3;", "healthy states"),
        ("n_anchor_levels  = 5;", "n_anchor_levels  = 2;", "anchor levels"),
        ("n_anchor_reps     = 5;", "n_anchor_reps     = 2;", "anchor replicas"),
        ("n_nuisance_states = 50;", "n_nuisance_states = 6;", "nuisance states"),
    ):
        add(
            f"micro guard lost production literal: {label}",
            micro=_replace_once(
                MAKE_MICRO_SOURCE,
                production,
                production.replace(";", " + 0;"),
            ),
        )
        add(
            f"micro guard lost replacement literal: {label}",
            micro=_replace_once(
                MAKE_MICRO_SOURCE,
                micro_value,
                micro_value.replace(";", " + 0;"),
            ),
        )
    add(
        "MATLAB CRN smoke lost namespace-isolation assertion",
        smoke=_replace_once(
            CRN_SMOKE_SOURCE,
            "assert(isequal(base_.track, mutated_.track));",
            "assert(true);",
        ),
    )
    add(
        "MATLAB CRN smoke reverted to seed schedule v1",
        smoke=_replace_once(
            CRN_SMOKE_SOURCE,
            "schedule = 'uid-named-substreams-v2';",
            "schedule = 'uid-named-substreams-v1';",
        ),
    )
    add(
        "MATLAB CRN smoke numeric oracle drifted",
        smoke=_replace_once(
            CRN_SMOKE_SOURCE,
            "uint32([1818075665 2898326234])",
            "uint32([1818075666 2898326234])",
        ),
    )
    add(
        "R11 serialization smoke lost state-seed type assertion",
        provenance_smoke=_replace_once(
            PROVENANCE_SMOKE_SOURCE,
            "assert(isa(saved.file_state_seed_id, 'uint32') && ...",
            "assert(true && ...",
        ),
    )
    add(
        "R11 serialization smoke lost zero-seed mutation",
        provenance_smoke=_replace_once(
            PROVENANCE_SMOKE_SOURCE,
            "bad.state_seed_id = uint32(0);",
            "bad.state_seed_id = uint32(1);",
        ),
    )

    for (
        name,
        ma,
        py,
        dr,
        env,
        save,
        micro,
        smoke,
        provenance_smoke,
    ) in mutations:
        _must_reject(
            name,
            ma,
            py,
            dr,
            env,
            save,
            micro,
            smoke,
            provenance_smoke,
        )
    print(f"GENERATION CONTRACT: ALL PASS ({len(mutations)} mutations caught)")


if __name__ == "__main__":
    main()

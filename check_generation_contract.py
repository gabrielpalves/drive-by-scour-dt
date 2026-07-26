"""Executable contract for the MATLAB/Python R9 generation identity.

Python treats ``gen_fingerprint`` as an opaque MATLAB-produced digest, so this
cross-language invariant necessarily lives in the source declarations. This
checker proves that:

* MATLAB and Python require the same global generator schema;
* the study tag is advanced with that schema;
* exactly one, unconditional generation-behaviour key enters both the
  fingerprint and the human-readable manifest; and
* the two superseded overlapping keys cannot return silently.

The mutations at the bottom operate only on in-memory strings. Each known
regression must make the validator fail; no repository file, data, result, or
bundle is modified.

Run:  python check_generation_contract.py
"""
from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
A00_PATH = ROOT / "scour_MATLAB" / "A00_Run.m"
DATASET_PATH = ROOT / "core" / "dataset.py"
DRIVER_PATH = ROOT / "comprehensive_ablation_multidamage.py"

EXPECTED_SCHEMA = "audit-2026-07-25-r9"
EXPECTED_BEHAVIOR_VERSION = "generation-rules-v1"
EXPECTED_STUDY_TAG = "gs6a20260725r9"
BEHAVIOR_KEY = "generation_behavior_version"
LEGACY_KEYS = ("gen_rule_ver", "track_eov_impl")


class ContractError(AssertionError):
    """A generation-identity invariant is absent or ambiguous."""


def _one(pattern: str, text: str, label: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ContractError(
            f"{label}: expected exactly one declaration, found {len(matches)}"
        )
    return matches[0]


def validate_contract(a00: str, dataset: str, driver: str) -> None:
    """Raise ContractError unless all R9 identity invariants hold."""
    matlab_schema = _one(
        r"^\s*gen_schema\s*=\s*'([^']+)';",
        a00,
        "MATLAB gen_schema",
    )
    python_schema = _one(
        r"^\s*_EXPECTED_GEN_SCHEMA\s*=\s*['\"]([^'\"]+)['\"]",
        dataset,
        "Python expected gen_schema",
    )
    study_tag = _one(
        r"^\s*SCHEMA_TAG\s*=\s*['\"]([^'\"]+)['\"]",
        driver,
        "Python study schema tag",
    )
    behavior_version = _one(
        rf"^\s*{BEHAVIOR_KEY}\s*=\s*'([^']+)';",
        a00,
        "MATLAB generation behavior version",
    )

    if matlab_schema != EXPECTED_SCHEMA:
        raise ContractError(
            f"MATLAB schema {matlab_schema!r} != pinned {EXPECTED_SCHEMA!r}"
        )
    if python_schema != matlab_schema:
        raise ContractError(
            f"Python schema {python_schema!r} != MATLAB schema {matlab_schema!r}"
        )
    if study_tag != EXPECTED_STUDY_TAG:
        raise ContractError(
            f"study tag {study_tag!r} != pinned {EXPECTED_STUDY_TAG!r}"
        )
    if not study_tag.endswith("r9"):
        raise ContractError("study tag does not advertise the R9 contract")
    if behavior_version != EXPECTED_BEHAVIOR_VERSION:
        raise ContractError(
            f"behavior version {behavior_version!r} != pinned "
            f"{EXPECTED_BEHAVIOR_VERSION!r}"
        )

    for legacy in LEGACY_KEYS:
        if legacy in a00:
            raise ContractError(f"superseded behavior key returned: {legacy}")

    definition_at = a00.index(f"{BEHAVIOR_KEY} =")
    fp_start = a00.index("fp_cfg = struct(")
    fp_end = a00.index("fp_cfg.DamageStates", fp_start)
    fp_block = a00[fp_start:fp_end]
    fp_binding = f"'{BEHAVIOR_KEY}', {BEHAVIOR_KEY}"
    if definition_at > fp_start:
        raise ContractError("behavior version is defined after fp_cfg")
    if fp_block.count(fp_binding) != 1:
        raise ContractError(
            "behavior version must enter fp_cfg exactly once and unconditionally"
        )
    if fp_block.count("'schema', gen_schema") != 1:
        raise ContractError("gen_schema must enter fp_cfg exactly once")

    manifest_start = a00.index("case_info = struct(")
    manifest_end = a00.index(
        "save(fullfile(run_folder, 'case_info.mat')", manifest_start
    )
    manifest_block = a00[manifest_start:manifest_end]
    if manifest_block.count(fp_binding) != 1:
        raise ContractError(
            "behavior version must enter case_info exactly once for inspection"
        )

    hash_at = a00.index("gen_fingerprint = local_sha256(jsonencode(fp_cfg));")
    if not (fp_end < hash_at < manifest_start):
        raise ContractError("fp_cfg is not hashed before case_info is written")


def _must_reject(name: str, a00: str, dataset: str, driver: str) -> None:
    try:
        validate_contract(a00, dataset, driver)
    except (ContractError, ValueError):
        print(f"  [PASS] mutation rejected: {name}")
        return
    raise AssertionError(f"mutation escaped generation-contract guard: {name}")


def main() -> None:
    a00 = A00_PATH.read_text(encoding="utf-8")
    dataset = DATASET_PATH.read_text(encoding="utf-8")
    driver = DRIVER_PATH.read_text(encoding="utf-8")

    validate_contract(a00, dataset, driver)
    print("GENERATION CONTRACT CHECKS")
    print("  [PASS] live MATLAB/Python R9 contract")

    _must_reject(
        "MATLAB schema drift",
        a00.replace(EXPECTED_SCHEMA, "audit-2026-07-19-r8", 1),
        dataset,
        driver,
    )
    _must_reject(
        "Python schema drift",
        a00,
        dataset.replace(EXPECTED_SCHEMA, "audit-2026-07-19-r8", 1),
        driver,
    )
    _must_reject(
        "behavior key removed from fingerprint",
        a00.replace(
            f"'{BEHAVIOR_KEY}', {BEHAVIOR_KEY}, ...",
            "'unrelated_config_key', 1, ...",
            1,
        ),
        dataset,
        driver,
    )

    # Replace only the second binding: the first one is in fp_cfg.
    first_binding = f"'{BEHAVIOR_KEY}', {BEHAVIOR_KEY}, ..."
    first_at = a00.index(first_binding)
    second_at = a00.index(first_binding, first_at + len(first_binding))
    manifest_mutation = (
        a00[:second_at]
        + "'kept_in_fingerprint_only', 1, ..."
        + a00[second_at + len(first_binding):]
    )
    _must_reject(
        "behavior key removed from case_info",
        manifest_mutation,
        dataset,
        driver,
    )
    _must_reject(
        "legacy conditional key reintroduced",
        a00.replace(
            "if use_track_eov",
            "track_eov_impl = 'legacy';\nif use_track_eov",
            1,
        ),
        dataset,
        driver,
    )
    _must_reject(
        "behavior version changed without contract update",
        a00.replace(EXPECTED_BEHAVIOR_VERSION, "generation-rules-v2", 1),
        dataset,
        driver,
    )
    _must_reject(
        "study tag left on R8",
        a00,
        dataset,
        driver.replace(EXPECTED_STUDY_TAG, "gs5a20260719r8", 1),
    )

    print("GENERATION CONTRACT: ALL PASS (7 mutations caught)")


if __name__ == "__main__":
    main()

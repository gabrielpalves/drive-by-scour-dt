"""Self-test driver for the contact-closure gate checker.

``python check_contact_closure_gate.py`` with no arguments dispatches here:
this module owns the checker's fast synthetic/mutation suite - the 420-case
inventory and source-contract guards, the direct case-evidence mutation
matrix, the create-once/TOCTOU/ABA receipt and snapshot probes, and the
top-level summary/inventory mutations.

It holds no verification logic of its own.  Every decision primitive is
imported from ``check_contact_closure_gate`` and every synthetic artifact from
``contact_gate_fixtures``, so a self-test can never assert against a private
reimplementation of the rule it is meant to exercise.

Like the fixture module, this driver is imported ONLY from the checker's
argument-free self-test branch (a deliberately lazy import inside ``main``).
Its bytes nevertheless belong to the checker's explicit transitive source
root, together with every production verifier module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import contact_gate_verifier_identity as verifier_identity
from check_contact_closure_gate import (
    GATE_SOURCE_FILES,
    GateError,
    EXPECTED_TOTAL_CASES,
    REVIEWED_TRACKPROP_DISPATCH,
    ROOT,
    SOLVER_SOURCE_FILES,
    STAGES,
    STUDY_HARNESS_FILES,
    SelectionRow,
    _authorize_gate,
    _contact_source_set,
    _contracts,
    _exact_keys,
    _gci,
    _mat_jsonish,
    _matlab_statements,
    _publish_receipt,
    _recompute_case,
    _require_clean_status,
    _revalidate_existing_receipt,
    _sha256_bytes,
    _strict_json_equivalent,
    _strict_json_file,
    _strict_json_text,
    _validate_gate_inventory,
    _validate_inventory,
    _validate_matlab_source_contract,
    _validate_actual_matlab_environment,
    _validate_receipt_location,
    _validate_summary_datasets,
    verifier_source_root,
    verify_gate,
)
from contact_gate_fixtures import (
    _build_synthetic_gate,
    _synthetic_rows,
    _write_canonical_json,
)
from contact_gate_verifier_identity import _static_local_imports_from_text


def _expect_failure(label: str, fn: Any) -> None:
    try:
        fn()
    except GateError:
        print(f"[PASS] mutation rejected: {label}")
    else:
        raise AssertionError(f"mutation unexpectedly passed: {label}")


def _expect_failure_message(label: str, fragment: str, fn: Any) -> None:
    """Reject a probe at its intended guard, not an unrelated later check."""
    try:
        fn()
    except GateError as exc:
        if fragment not in str(exc):
            raise AssertionError(
                f"{label} failed at the wrong guard: {exc}"
            ) from exc
        print(f"[PASS] mutation rejected: {label}")
    else:
        raise AssertionError(f"mutation unexpectedly passed: {label}")


def _mutated_statement_source(
    original: str,
    old: str,
    new: str,
    *,
    expected: int = 1,
) -> str:
    """Rewrite ``old`` -> ``new`` in EXECUTABLE STATEMENT lines only.

    ``_validate_matlab_source_contract`` evaluates its pins against
    ``_matlab_statements`` output, so a probe that edits a whole-line comment
    changes nothing the guard can see and its ``_expect_failure`` wrapper then
    reports the guard as broken.  That is not hypothetical: the study modules
    quote their own invariants verbatim in their rationale headers, so a plain
    ``str.replace(..., 1)`` hits the header, not the statement.

    This helper removes that entire failure class rather than one instance:
    whole-line comments are skipped, the number of rewritten occurrences must
    equal ``expected``, and the result must differ from the original AFTER
    comment stripping.  A probe therefore cannot silently degrade into a
    no-op - it fails loudly at construction time instead.
    """
    rewritten: list[str] = []
    replaced = 0
    for raw in original.splitlines(keepends=True):
        body = raw.rstrip("\n")
        terminator = raw[len(body):]
        if not body.strip().startswith("%") and old in body:
            replaced += body.count(old)
            body = body.replace(old, new)
        rewritten.append(body + terminator)
    if replaced != expected:
        raise AssertionError(
            f"mutation probe rewrote {replaced} statement occurrence(s) of "
            f"{old!r}, expected {expected}: the probe does not target the "
            "executable statement it claims to mutate"
        )
    mutated = "".join(rewritten)
    if _matlab_statements(mutated) == _matlab_statements(original):
        raise AssertionError(
            f"mutation probe for {old!r} leaves the comment-stripped source "
            "unchanged, so it cannot exercise the source-contract guard"
        )
    return mutated


def _recompute_synthetic_case(
    fixture: dict[str, Any],
    case: dict[str, Any],
) -> None:
    row = fixture["rows"][0]
    descriptor = fixture["descriptors"][0]
    _recompute_case(
        case,
        row,
        dataset=descriptor,
        policy_sha=fixture["policy_sha"],
        selection_sha=fixture["selection_sha"],
        source_root=fixture["source_root"],
        environment_sha=fixture["environment_sha"],
        solver_root=fixture["solver_root"],
        harness_sha=fixture["harness_sha"],
        b66_sha=fixture["b66_sha"],
        expected_descriptor=fixture["expected_physical_descriptors"][
            (row.stage, row.state_index, row.passage_index)
        ],
    )


def _check_inventory_and_numerics() -> None:
    """Case inventory, contraction/GCI branches, and the real MAT projection."""
    rows = _synthetic_rows()
    _validate_inventory(rows)
    print(f"[PASS] exact {EXPECTED_TOTAL_CASES}-case inventory")
    assert _contracts([1000, 800, 700])
    assert not _contracts([700, 800, 700])
    steps = [0.001, 0.0005, 0.00025]
    assert _gci([10000, 9000, 8500], steps, 24000)[0]
    assert _gci([0, 0, 0], steps, 24000)[0]
    assert not _gci([10000, 9000, 9500], steps, 24000)[0]
    print("[PASS] contraction/GCI branches")

    # A coherent, non-reference MATLAB descriptor is valid evidence. Exact
    # release equality is not a qualification rule, while descriptor/SHA and
    # release/descriptor consistency remain fail-closed.
    from check_contact_closure_gate import _locked_matlab_environment

    _, reference_descriptor, _ = _locked_matlab_environment()
    replacements = {
        "release": "R2024b",
        "version": "24.2.0 portable fixture (R2024b)",
        "matlab_product_version": "24.2",
        "statistics_toolbox_version": "24.2",
        "parallel_toolbox_version": "24.2",
    }
    portable_lines = []
    for line in reference_descriptor.split("\n"):
        field, value = line.split("=", 1)
        portable_lines.append(f"{field}={replacements.get(field, value)}")
    portable_descriptor = "\n".join(portable_lines)
    portable_sha = _sha256_bytes(portable_descriptor.encode("utf-8"))
    parsed = _validate_actual_matlab_environment(
        portable_descriptor, portable_sha, "R2024b"
    )
    assert parsed["release"] == "R2024b"
    _expect_failure(
        "portable MATLAB descriptor with foreign SHA",
        lambda: _validate_actual_matlab_environment(
            portable_descriptor, "0" * 64, "R2024b"
        ),
    )
    _expect_failure(
        "portable MATLAB descriptor with foreign release stamp",
        lambda: _validate_actual_matlab_environment(
            portable_descriptor, portable_sha, "R2025b"
        ),
    )
    print("[PASS] coherent non-reference MATLAB environment accepted")

    # Exercise the same SciPy options and recursive conversion used for real
    # MATLAB evidence.  MATLAB logicals, empty char vectors and nonscalar
    # struct arrays each have distinct level-5 MAT representations that
    # otherwise look deceptively similar to JSON 0/1, [] and opaque objects.
    import numpy as np
    from scipy.io import loadmat, savemat
    with tempfile.TemporaryDirectory() as mat_raw:
        mat_path = Path(mat_raw) / "mat_jsonish_roundtrip.mat"
        struct_array = np.empty(
            (2, 2),
            dtype=[("status", "O"), ("flag", "O"), ("values", "O")],
        )
        statuses = (("X", "Y"), ("X", "X"))
        for row_index in range(2):
            for column_index in range(2):
                struct_array[row_index, column_index] = (
                    statuses[row_index][column_index],
                    np.array([[True]], dtype=bool),
                    np.array([[1.0, 2.0, 3.0]]),
                )
        savemat(
            mat_path,
            {
                "payload": {
                    "scalar_flag": True,
                    "flags": np.array([True, False], dtype=bool),
                    "empty_text": "",
                    "empty_numeric": np.array([], dtype=float),
                    "empty_cell": np.array([], dtype=object),
                    "struct_array": struct_array,
                }
            },
        )
        loaded_payload = loadmat(
            mat_path, simplify_cells=True, mat_dtype=True)["payload"]
        observed_payload = _mat_jsonish(
            loaded_payload, "MAT/JSON round-trip fixture")
        expected_structs = [
            [
                {
                    "status": statuses[i][j],
                    "flag": True,
                    "values": [1.0, 2.0, 3.0],
                }
                for j in range(2)
            ]
            for i in range(2)
        ]
        expected_payload = {
            "scalar_flag": True,
            "flags": [True, False],
            "empty_text": "",
            "empty_numeric": [],
            "empty_cell": [],
            "struct_array": expected_structs,
        }
        if not _strict_json_equivalent(observed_payload, expected_payload):
            raise AssertionError(
                "logical/empty/struct-array MAT projection differs from JSON")
    print("[PASS] real MAT logical/empty/struct-array projection")


def _check_strict_parsing_and_inventory_mutations() -> None:
    """Strict JSON/Git parsing and the selection-inventory mutation matrix."""
    rows = _synthetic_rows()
    _expect_failure(
        "strict JSON duplicate key",
        lambda: _strict_json_text('{"x": 1, "x": 2}', "duplicate fixture"),
    )
    _expect_failure(
        "strict JSON NaN",
        lambda: _strict_json_text('{"x": NaN}', "NaN fixture"),
    )
    _expect_failure(
        "strict JSON Infinity",
        lambda: _strict_json_text('{"x": Infinity}', "Infinity fixture"),
    )
    _expect_failure(
        "dirty tracked worktree",
        lambda: _require_clean_status(" M tracked.m\n"),
    )
    _expect_failure(
        "dirty untracked worktree",
        lambda: _require_clean_status("?? untracked.tmp\n"),
    )

    _expect_failure("one omitted case", lambda: _validate_inventory(rows[:-1]))
    duplicated = rows.copy()
    duplicated[-1] = duplicated[-2]
    _expect_failure("duplicated case", lambda: _validate_inventory(duplicated))
    reordered = rows.copy()
    reordered[0], reordered[1] = reordered[1], reordered[0]
    _expect_failure("case reorder", lambda: _validate_inventory(reordered))
    changed_family = rows.copy()
    changed_family[0] = replace(changed_family[0], state_family="joint")
    _expect_failure(
        "family-count drift", lambda: _validate_inventory(changed_family)
    )
    changed_sha = rows.copy()
    changed_sha[0] = replace(changed_sha[0], state_file_sha256="bad")
    _expect_failure("malformed state SHA", lambda: _validate_inventory(changed_sha))


def _check_source_contract() -> None:
    """Executable Paper-1 MATLAB source contract and TrackProp closure."""

    study_sources = _contact_source_set(STUDY_HARNESS_FILES)
    gate_sources = _contact_source_set(GATE_SOURCE_FILES)
    solver_sources = _contact_source_set(SOLVER_SOURCE_FILES)
    _validate_matlab_source_contract(
        study_sources,
        gate_sources,
        solver_sources=solver_sources,
    )
    if _static_local_imports_from_text(
        "static-import-probe.py", "import contact_gate_core\n"
    ) != {"contact_gate_core.py"}:
        raise AssertionError(
            "verifier identity did not discover a direct local import"
        )
    _expect_failure(
        "dynamic local Python import evades static verifier closure",
        lambda: _static_local_imports_from_text(
            "dynamic-import-probe.py",
            "from importlib import import_module\n"
            "import_module('contact_gate_core')\n",
        ),
    )
    relative_imports = _static_local_imports_from_text(
        "core/relative-import-probe.py",
        "from . import preprocessing\n",
    )
    if relative_imports != {
        "core/__init__.py",
        "core/preprocessing.py",
    }:
        raise AssertionError(
            "verifier identity did not resolve a package-relative import"
        )
    _expect_failure(
        "invalid top-level relative Python import",
        lambda: _static_local_imports_from_text(
            "relative-import-probe.py", "from . import shadow\n"
        ),
    )
    _expect_failure(
        "aliased importlib loader evades static verifier closure",
        lambda: _static_local_imports_from_text(
            "dynamic-import-alias-probe.py",
            "from importlib import import_module as load_local\n"
            "load_local(name='contact_gate_core')\n",
        ),
    )
    _expect_failure(
        "assignment-aliased importlib loader evades static verifier closure",
        lambda: _static_local_imports_from_text(
            "dynamic-assignment-alias-probe.py",
            "import importlib as loader_library\n"
            "load_local = loader_library.import_module\n"
            "load_local(name='contact_gate_core')\n",
        ),
    )
    _expect_failure(
        "getattr-obtained importlib loader evades static verifier closure",
        lambda: _static_local_imports_from_text(
            "dynamic-getattr-probe.py",
            "import importlib\n"
            "getattr(importlib, 'import_module')('contact_gate_core')\n",
        ),
    )
    indirect_loader_probes = (
        (
            "concatenated getattr loader name",
            "import importlib\n"
            "getattr(importlib, 'import_' + 'module')('contact_gate_core')\n",
            "reflective getattr",
        ),
        (
            "concatenated importlib.__dict__ loader name",
            "import importlib\n"
            "importlib.__dict__['import_' + 'module']('contact_gate_core')\n",
            "__dict__",
        ),
        (
            "loader factory passed through functools.partial",
            "import functools\nimport importlib\n"
            "factory = functools.partial(importlib.import_module, "
            "'contact_gate_core')\nfactory()\n",
            "factory/reference",
        ),
    )
    for label, probe_source, diagnostic in indirect_loader_probes:
        if "contact_gate_core" not in probe_source or len(probe_source) < 40:
            raise AssertionError(f"vacuous loader-indirection probe: {label}")
        _expect_failure_message(
            label,
            diagnostic,
            lambda source=probe_source: _static_local_imports_from_text(
                "dynamic-indirection-probe.py", source
            ),
        )

    for builtin_name, probe_source in (
        ("exec", "payload = 'x = 1'\nexec(payload)\n"),
        ("eval", "payload = '1 + 1'\neval(payload)\n"),
        (
            "compile",
            "payload = 'x = 1'\ncompile(payload, 'probe', 'exec')\n",
        ),
    ):
        if len(probe_source) < 25:
            raise AssertionError(
                f"vacuous contact-verifier {builtin_name} probe"
            )
        _expect_failure_message(
            f"bare {builtin_name} unavailable to contact verifier",
            f"bare {builtin_name}()",
            lambda source=probe_source: _static_local_imports_from_text(
                "contact-verifier-builtin-probe.py", source
            ),
        )

    snapshot_target = ROOT / "contact_gate_policy.py"
    original_target_bytes = snapshot_target.read_bytes()
    original_snapshot_source = verifier_identity._snapshot_source
    snapshot_probe_fired = {"value": False}

    def mutate_after_verifier_snapshot(relative: str):
        snapshot = original_snapshot_source(relative)
        if (
            relative == "contact_gate_policy.py"
            and not snapshot_probe_fired["value"]
        ):
            snapshot_probe_fired["value"] = True
            snapshot_target.write_bytes(
                original_target_bytes + b"\n# verifier-root TOCTOU probe\n"
            )
        return snapshot

    verifier_identity._snapshot_source = mutate_after_verifier_snapshot
    try:
        _expect_failure(
            "verifier source changes after structural/hash snapshot",
            verifier_source_root,
        )
    finally:
        verifier_identity._snapshot_source = original_snapshot_source
        snapshot_target.write_bytes(original_target_bytes)
    if not snapshot_probe_fired["value"]:
        raise AssertionError("verifier-root TOCTOU mutation probe was vacuous")
    seed_owner = "contact_profile_descriptor.m"
    window_owner = "contact_closure_study.m"

    # Cover the anti-vacuity machinery itself before relying on it: the source
    # probes below are only evidence if a mutation that touches nothing the
    # guard can see is rejected AT CONSTRUCTION rather than reported as a
    # guard failure.  The seed literal is deliberately quoted in the owning
    # module's rationale header, so the raw text holds it twice and the
    # statements exactly once - the precise asymmetry that made a plain
    # str.replace(..., 1) probe a silent no-op.
    seed_statement = "cfg.phase_seed = descriptor_contract.profile_phase_seed;"
    if _matlab_statements(study_sources[seed_owner]).count(seed_statement) != 1:
        raise AssertionError(
            f"{seed_owner} must assign the frozen phase seed exactly once as "
            "an executable statement"
        )
    for label, old, new, expected in (
        ("comment-only target", "% RATIONALE", "% MUTATED", 1),
        ("wrong occurrence count", seed_statement, seed_statement, 2),
    ):
        try:
            _mutated_statement_source(
                study_sources[seed_owner], old, new, expected=expected
            )
        except AssertionError:
            print(f"[PASS] vacuous mutation probe rejected: {label}")
        else:
            raise AssertionError(
                f"vacuous mutation probe accepted: {label}"
            )
    _expect_failure(
        "profile-phase seed arithmetic mutation",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    seed_owner: _mutated_statement_source(
                        study_sources[seed_owner],
                        "cfg.phase_seed = descriptor_contract."
                        "profile_phase_seed;",
                        "cfg.phase_seed = descriptor_contract."
                        "profile_phase_seed + state_index;",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "comparison-window span mutation (18.30 -> 18.31)",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    window_owner: _mutated_statement_source(
                        study_sources[window_owner],
                        "+ 18.30;",
                        "+ 18.31;",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact study canonicalized a linked dataset root before inspection",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    window_owner: _mutated_statement_source(
                        study_sources[window_owner],
                        "dataset_identity = "
                        "contact_unlinked_path_identity(dataset_dir);",
                        "dataset_identity = struct('canonical_path', "
                        "char(java.io.File(dataset_dir).getCanonicalPath()));",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "digest validator accepted a linked dataset root",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "validate_dataset_digest_manifest.m":
                    _mutated_statement_source(
                        study_sources["validate_dataset_digest_manifest.m"],
                        "dataset_identity = "
                        "contact_unlinked_path_identity(dataset_dir);",
                        "dataset_identity = struct('exists', true, "
                        "'is_directory', true, 'canonical_path', dataset_dir);",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact filesystem identity lost the Windows native fallback",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_filesystem_identity.m":
                    _mutated_statement_source(
                        study_sources["contact_filesystem_identity.m"],
                        "identity = ['windows|' "
                        "contact_windows_file_identity(path)];",
                        "identity = ['windows|' path];",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact filesystem identity accepted an empty Java key string",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_filesystem_identity.m":
                    _mutated_statement_source(
                        study_sources["contact_filesystem_identity.m"],
                        "if ~isempty(key_text)",
                        "if true",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact Java Boolean boundary accepted arbitrary numeric metadata",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_java_boolean_value.m":
                    _mutated_statement_source(
                        study_sources["contact_java_boolean_value.m"],
                        "(raw == 0 || raw == 1)",
                        "true",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact alias guard bypassed the Java Boolean boundary",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_path_component_is_link_alias.m":
                    _mutated_statement_source(
                        study_sources[
                            "contact_path_component_is_link_alias.m"
                        ],
                        "contact_java_boolean_value(symbolic)",
                        "logical(symbolic)",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact native metadata process lost its finite timeout",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_run_small_process.m":
                    _mutated_statement_source(
                        study_sources["contact_run_small_process.m"],
                        "if ~finished",
                        "if false",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact Windows identity bypassed the bounded process runner",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_windows_file_identity.m":
                    _mutated_statement_source(
                        study_sources["contact_windows_file_identity.m"],
                        "[lines, exit_code] = "
                        "contact_run_small_process( ...",
                        "[exit_code, lines] = system( ...",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact Windows identity trusted an environment-selected executable",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_windows_file_identity.m":
                    _mutated_statement_source(
                        study_sources["contact_windows_file_identity.m"],
                        "system_directory = "
                        "char(System.Environment.SystemDirectory);",
                        "system_directory = getenv('SystemRoot');",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact hard-link query bypassed the bounded process runner",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_regular_nonsymlink.m":
                    _mutated_statement_source(
                        study_sources["contact_regular_nonsymlink.m"],
                        "[raw_lines, exit_code] = "
                        "contact_run_small_process( ...",
                        "[exit_code, raw_lines] = system( ...",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact Windows identity lost its volume binding",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_windows_file_identity.m":
                    _mutated_statement_source(
                        study_sources["contact_windows_file_identity.m"],
                        "identity = ['volume-vsn=' volume_before "
                        "'|file-id=' file_id];",
                        "identity = ['file-id=' file_id];",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact Windows identity accepted ambiguous fsutil output",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_windows_file_identity.m":
                    _mutated_statement_source(
                        study_sources["contact_windows_file_identity.m"],
                        "if exit_code ~= 0 || numel(matches) ~= 1",
                        "if exit_code ~= 0 || isempty(matches)",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact file observation lost its second identity fence",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_file_observation.m":
                    _mutated_statement_source(
                        study_sources["contact_file_observation.m"],
                        "confirmed_identity = "
                        "contact_unlinked_path_identity(path);",
                        "confirmed_identity = identity;",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact path observation lost its second identity fence",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    "contact_unlinked_path_identity.m":
                    _mutated_statement_source(
                        study_sources[
                            "contact_unlinked_path_identity.m"
                        ],
                        "confirmed_file_key = "
                        "contact_filesystem_identity(absolute_native);",
                        "confirmed_file_key = file_key;",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "contact gate canonicalized linked qualification roots",
        lambda: _validate_matlab_source_contract(
            study_sources,
            _contact_source_set(
                GATE_SOURCE_FILES,
                {
                    "contact_closure_gate.m": _mutated_statement_source(
                        gate_sources["contact_closure_gate.m"],
                        "dataset_identity = contact_unlinked_path_identity("
                        "dataset_dirs{k});",
                        "dataset_identity = struct('exists', true, "
                        "'is_directory', true, 'canonical_path', "
                        "dataset_dirs{k});",
                    )
                },
            ),
        ),
    )
    _expect_failure(
        "gate selection re-read a state path for its reported SHA",
        lambda: _validate_matlab_source_contract(
            study_sources,
            _contact_source_set(
                GATE_SOURCE_FILES,
                {
                    "contact_gate_build_selection.m":
                    _mutated_statement_source(
                        gate_sources["contact_gate_build_selection.m"],
                        "state_sha = state_snapshot.sha256;",
                        "state_sha = common.file_sha256(state_path);",
                    )
                },
            ),
        ),
    )
    _expect_failure(
        "gate selection omitted the final four-dataset reassertion",
        lambda: _validate_matlab_source_contract(
            study_sources,
            _contact_source_set(
                GATE_SOURCE_FILES,
                {
                    "contact_gate_build_selection.m":
                    _mutated_statement_source(
                        gate_sources["contact_gate_build_selection.m"],
                        "all_dataset_snapshots{stage_index});",
                        "dataset_snapshots); % MUTANT",
                    )
                },
            ),
        ),
    )
    _expect_failure(
        "profile-phase seed assignment migrated out of its owning module",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    seed_owner: _mutated_statement_source(
                        study_sources[seed_owner],
                        "cfg.phase_seed = descriptor_contract."
                        "profile_phase_seed;",
                        "cfg.phase_seed = local_phase_seed("
                        "descriptor_contract);",
                    )
                },
            ),
            gate_sources,
        ),
    )
    _expect_failure(
        "gate policy literal moved out of the policy module",
        lambda: _validate_matlab_source_contract(
            study_sources,
            _contact_source_set(
                GATE_SOURCE_FILES,
                {
                    "contact_gate_policy_definition.m":
                    _mutated_statement_source(
                        gate_sources["contact_gate_policy_definition.m"],
                        f"policy.expected_cases = {EXPECTED_TOTAL_CASES};",
                        "",
                    )
                },
            ),
        ),
    )
    # The executed-module inventories are only auditable while the chain stays
    # statically analysable, so prove both halves of that rule fail closed.
    _expect_failure(
        "dynamic dispatch introduced into a contact-chain module",
        lambda: _validate_matlab_source_contract(
            study_sources,
            _contact_source_set(
                GATE_SOURCE_FILES,
                {"contact_gate_selection.m": "selection = eval('struct()');\n"},
            ),
        ),
    )
    _expect_failure(
        "reviewed direct TrackProp call repointed to the NoBallast variant",
        lambda: _validate_matlab_source_contract(
            study_sources,
            gate_sources,
            solver_sources=_contact_source_set(
                SOLVER_SOURCE_FILES,
                {
                    "A02_Track.m": _mutated_statement_source(
                        solver_sources["A02_Track.m"],
                        "TrackProp_Zhai_et_al_WithBallastOnBridge",
                        "TrackProp_Zhai_et_al_NoBallastOnBridge",
                    )
                },
            ),
        ),
    )
    _expect_failure(
        "reviewed direct TrackProp call malformed in A02_Track",
        lambda: _validate_matlab_source_contract(
            study_sources,
            gate_sources,
            solver_sources=_contact_source_set(
                SOLVER_SOURCE_FILES,
                {
                    "A02_Track.m": _mutated_statement_source(
                        solver_sources["A02_Track.m"],
                        REVIEWED_TRACKPROP_DISPATCH,
                        "Track = TrackProp_Zhai_et_al_WithBallastOnBridge();",
                    )
                },
            ),
        ),
    )
    _expect_failure(
        "dynamic run reintroduced in A02_Track",
        lambda: _validate_matlab_source_contract(
            study_sources,
            gate_sources,
            solver_sources=_contact_source_set(
                SOLVER_SOURCE_FILES,
                {
                    "A02_Track.m": _mutated_statement_source(
                        solver_sources["A02_Track.m"],
                        REVIEWED_TRACKPROP_DISPATCH,
                        "run('TrackProp_Zhai_et_al_WithBallastOnBridge');",
                    )
                },
            ),
        ),
    )
    for label, owner, old, new in (
        (
            "dynamic eval introduced in reviewed TrackProp function",
            "TrackProp_Zhai_et_al_WithBallastOnBridge.m",
            "Track.Rail.Prop.E = 2.059e11;",
            "Track.Rail.Prop.E = eval('2.059e11');",
        ),
        (
            "dynamic eval introduced in reviewed TrainProp function",
            "TrainProp_ObrienCalibrate.m",
            "var_BodyMass = (x(1)*0.10*36852);",
            "var_BodyMass = eval('x(1)*0.10*36852');",
        ),
        (
            "dynamic feval introduced in non-entry solver module",
            "B66_ContactForce.m",
            "ones_1_x_num_t = ones(1,Calc.Solver.num_t);",
            "ones_1_x_num_t = feval('ones',1,Calc.Solver.num_t);",
        ),
    ):
        _expect_failure(
            label,
            lambda owner=owner, old=old, new=new: (
                _validate_matlab_source_contract(
                    study_sources,
                    gate_sources,
                    solver_sources=_contact_source_set(
                        SOLVER_SOURCE_FILES,
                        {
                            owner: _mutated_statement_source(
                                solver_sources[owner],
                                old,
                                new,
                            )
                        },
                    ),
                )
            ),
        )
    solver_inventory_owner = "contact_solver_modules.m"
    _expect_failure(
        "static solver inventory drops direct TrackProp dependency",
        lambda: _validate_matlab_source_contract(
            _contact_source_set(
                STUDY_HARNESS_FILES,
                {
                    solver_inventory_owner: _mutated_statement_source(
                        study_sources[solver_inventory_owner],
                        "'TrackProp_Zhai_et_al_WithBallastOnBridge';",
                        "'TrainProp_ObrienCalibrate';",
                    )
                },
            ),
            gate_sources,
            solver_sources=solver_sources,
        ),
    )
    print("[PASS] executable Paper-1 source-contract guard")


def _check_case_evidence(fixture: dict[str, Any], raw: str) -> dict[str, Any]:
    """Per-case evidence recomputation and its mutation matrix."""
    gate_dir = fixture["gate_dir"]
    commit = fixture["source_commit"]
    valid_case = fixture["first_case"]
    _recompute_synthetic_case(fixture, valid_case)
    forged_plain = deepcopy(valid_case)
    forged_plain["report_plain"]["run_table"]["actual_dt_ms"][0] *= 0.9
    _expect_failure(
        "canonical plain report differs from decisive projection",
        lambda: _recompute_synthetic_case(fixture, forged_plain),
    )
    _expect_failure(
        "MAT publication missing exact selection identity root",
        lambda: _exact_keys(
            {"summary": {}, "selection_descriptor": "x"},
            {"summary", "selection_descriptor", "selection_sha256"},
            "forged publication",
        ),
    )
    assert any(
        abs(actual - requested) > math.ulp(max(1.0, requested))
        for actual, requested in zip(
            valid_case["actual_dt_ms"], valid_case["requested_dt_ms"])
    )
    print("[PASS] nonintegral B11 ceil-grid identity")

    def reject_case(label: str, mutate: Any) -> None:
        changed = deepcopy(valid_case)
        mutate(changed)
        _expect_failure(
            label, lambda: _recompute_synthetic_case(fixture, changed))

    reject_case(
        "report passage swap",
        lambda case: case.__setitem__("report_passage_index", 2),
    )
    reject_case(
        "report state index boolean masquerading as one",
        lambda case: case.__setitem__("report_state_index", True),
    )
    reject_case(
        "report family mismatch",
        lambda case: case.__setitem__("report_state_family", "joint"),
    )
    reject_case(
        "nonintegral profile-phase stream index",
        lambda case: case.__setitem__("profile_phase_stream_index", 5.999),
    )
    reject_case(
        "numeric string in projected policy",
        lambda case: case.__setitem__("fraction_gate", "0.002"),
    )
    reject_case(
        "signed/derived contact peak mismatch",
        lambda case: case["peak_tension_N"].__setitem__(0, 1.0),
    )
    reject_case(
        "contact flag outside exact 0/1 domain",
        lambda case: case["contact_lost_track"].__setitem__(0, 2),
    )
    reject_case(
        "forged contact flag inconsistent with signed reaction",
        lambda case: case["contact_lost_track"].__setitem__(0, 1),
    )
    reject_case(
        "negative tension fraction",
        lambda case: case["tension_fraction"].__setitem__(0, -1e-6),
    )
    reject_case(
        "negative waveform NRMSE",
        lambda case: case["channel_metrics"][
            "nrmse_vs_finest"].__setitem__(0, -1e-6),
    )
    reject_case(
        "boolean masquerading as numeric waveform metric",
        lambda case: case["channel_metrics"][
            "nrmse_vs_finest"].__setitem__(2, False),
    )
    reject_case(
        "numeric string in waveform metric",
        lambda case: case["channel_metrics"][
            "nrmse_vs_finest"].__setitem__(0, "0.01"),
    )
    reject_case(
        "numeric string in QOI",
        lambda case: case["channel_qoi"][
            "signal_rms"].__setitem__(0, "1"),
    )
    reject_case(
        "numeric string in reconstruction",
        lambda case: case["saved_reconstruction"][
            "max_tolerance_ratio"].__setitem__(0, "0"),
    )
    reject_case(
        "waveform correlation above one",
        lambda case: case["channel_metrics"][
            "correlation_vs_finest"].__setitem__(0, 1.0001),
    )
    reject_case(
        "finest waveform row is not self-identity",
        lambda case: case["channel_metrics"][
            "nmax_vs_finest"].__setitem__(2, 1e-6),
    )
    reject_case(
        "saved contact differs from frozen selection",
        lambda case: case["saved_contact_log"].__setitem__(3, -99999),
    )
    reject_case(
        "1-ms rerun contact differs from frozen selection",
        lambda case: case["rerun_contact_log_1ms"].__setitem__(3, -99999),
    )
    reject_case(
        "reconstruction fallback mode",
        lambda case: case.__setitem__(
            "saved_baseline_mode", "interpolated_fallback"),
    )
    reject_case(
        "reconstruction tolerance ratio above one",
        lambda case: case["saved_reconstruction"][
            "max_tolerance_ratio"].__setitem__(0, 1.0001),
    )
    reject_case(
        "B11 sample-count identity mutation",
        lambda case: case["n_samples"].__setitem__(
            0, case["n_samples"][0] - 1),
    )
    reject_case(
        "acceptance projection differs from recomputation",
        lambda case: case["acceptance"][
            "contact_peak_N"].__setitem__(0, 1.0),
    )
    reject_case(
        "acceptance GCI differs from recomputation",
        lambda case: case["acceptance"]["peak_gci"].__setitem__(
            "upper_bound", 1.0),
    )
    reject_case(
        "acceptance channel QOI GCI is forged",
        lambda case: case["acceptance"].__setitem__(
            "channel_qoi_gci", {"forged": "arbitrary"}),
    )
    reject_case(
        "integer contact classification masquerading as boolean",
        lambda case: case["acceptance"].__setitem__(
            "contact_classification",
            [[int(item) for item in row_] for row_ in
             case["acceptance"]["contact_classification"]],
        ),
    )
    reject_case(
        "oscillatory channel QOI GCI",
        lambda case: [
            case["channel_qoi"]["signal_rms"].__setitem__(index, value)
            for index, value in zip((0, 1, 2), (1.0, 2.0, 1.0))
        ],
    )
    reject_case(
        "forged canonical solver-source payload",
        lambda case: case["report_plain"]["solver_source_sha256"][
            "sha256"].__setitem__(0, "0" * 64),
    )
    reject_case(
        "forged numeric hash convention selfcheck",
        lambda case: case["report_plain"].__setitem__(
            "numeric_hash_selfcheck", "0" * 64),
    )
    reject_case(
        "numeric string in canonical report descriptor vector",
        lambda case: case["report_plain"]["descriptor"][
            "scour_vector"].__setitem__(0, "0"),
    )
    reject_case(
        "valid numeric physical descriptor forgery",
        lambda case: case["report_plain"]["descriptor"][
            "scour_vector"].__setitem__(0, 0.125),
    )
    reject_case(
        "boolean masquerading as canonical report descriptor integer",
        lambda case: case["report_plain"]["descriptor"].__setitem__(
            "num_spans", True),
    )
    print("[PASS] direct case-evidence mutation matrix")
    return valid_case


def _check_receipt_publication(
    fixture: dict[str, Any],
    raw: str,
    no_git: Any,
    no_dataset: Any,
    no_mat: Any,
) -> tuple[Path, dict[str, Any]]:
    """Full synthetic gate authorization and retained-receipt revalidation."""
    gate_dir = fixture["gate_dir"]
    commit = fixture["source_commit"]
    receipt_path = (Path(raw) / "authorization.json").resolve()
    receipt = _authorize_gate(
        gate_dir,
        commit,
        receipt_path,
        revalidate_receipt=False,
        git_check=no_git,
        dataset_validator=no_dataset,
        mat_validator=no_mat,
    )
    _authorize_gate(
        gate_dir,
        commit,
        receipt_path,
        revalidate_receipt=True,
        git_check=no_git,
        dataset_validator=no_dataset,
        mat_validator=no_mat,
    )
    print(
        f"[PASS] full {EXPECTED_TOTAL_CASES}-case synthetic gate and "
        "retained receipt"
    )
    return receipt_path, receipt


def _check_receipt_and_summary_mutations(
    fixture: dict[str, Any],
    raw: str,
    receipt_path: Path,
    receipt: dict[str, Any],
    no_git: Any,
    no_dataset: Any,
    no_mat: Any,
) -> None:
    """Receipt location/create-once/TOCTOU/ABA probes and summary mutations."""
    gate_dir = fixture["gate_dir"]
    commit = fixture["source_commit"]
    gate_alias = gate_dir / ".." / gate_dir.name
    _expect_failure(
        "lexical alias for gate directory",
        lambda: verify_gate(
            gate_alias,
            commit,
            git_check=no_git,
            dataset_validator=no_dataset,
            mat_validator=no_mat,
        ),
    )
    receipt_parent = Path(raw) / "receipt-parent"
    receipt_parent.mkdir()
    receipt_alias = (
        receipt_parent / ".." / receipt_parent.name / "aliased.json"
    )
    _expect_failure(
        "lexical alias for receipt parent",
        lambda: _validate_receipt_location(receipt_alias, gate_dir),
    )
    summary = _strict_json_file(
        gate_dir / "gate_summary.json",
        "path-alias summary",
    )
    dataset_alias = (
        Path(raw)
        / f"dataset_{STAGES[0]}"
        / ".."
        / f"dataset_{STAGES[0]}"
    )
    alias_sha = _sha256_bytes(str(dataset_alias).encode("utf-8"))
    alias_items = deepcopy(summary["datasets"])
    alias_items[0]["dataset_dir"] = str(dataset_alias)
    alias_items[0]["dataset_dir_sha256"] = alias_sha
    alias_descriptors = list(fixture["descriptors"])
    alias_descriptors[0] = replace(
        alias_descriptors[0],
        dataset_dir_sha256=alias_sha,
    )
    _expect_failure(
        "coherently rebound lexical dataset alias",
        lambda: _validate_summary_datasets(
            alias_items,
            alias_descriptors,
        ),
    )
    symlink_gate = Path(raw) / "gate-symlink"
    try:
        symlink_gate.symlink_to(gate_dir, target_is_directory=True)
    except OSError:
        print("[N/A] gate symlink path probe (host privilege unavailable)")
    else:
        try:
            _expect_failure(
                "symlink alias for gate directory",
                lambda: verify_gate(
                    symlink_gate,
                    commit,
                    git_check=no_git,
                    dataset_validator=no_dataset,
                    mat_validator=no_mat,
                ),
            )
        finally:
            symlink_gate.unlink()

    _expect_failure(
        "receipt located inside gate",
        lambda: _validate_receipt_location(
            gate_dir / "authorization.json", gate_dir),
    )
    _expect_failure(
        "receipt located inside qualification dataset",
        lambda: _validate_receipt_location(
            (Path(raw) / f"dataset_{STAGES[0]}" / "authorization.json"),
            gate_dir,
        ),
    )
    _expect_failure(
        "receipt located inside repository",
        lambda: _validate_receipt_location(
            ROOT / "authorization.json", gate_dir),
    )
    _expect_failure(
        "create-once receipt overwrite",
        lambda: _publish_receipt(receipt_path, receipt),
    )
    original_receipt = receipt_path.read_bytes()
    changed_receipt = _strict_json_file(
        receipt_path, "synthetic receipt")
    changed_receipt["status"] = "FORGED"
    _write_canonical_json(receipt_path, changed_receipt)
    _expect_failure(
        "retained receipt mutation",
        lambda: _revalidate_existing_receipt(receipt_path, receipt),
    )
    receipt_path.write_bytes(original_receipt)
    timestamped_receipt = _strict_json_file(
        receipt_path, "synthetic receipt")
    timestamped_receipt["validated_utc"] = "2099-01-01T00:00:00+00:00"
    _write_canonical_json(receipt_path, timestamped_receipt)
    _expect_failure(
        "receipt completion timestamp drift",
        lambda: _revalidate_existing_receipt(receipt_path, receipt),
    )
    receipt_path.write_bytes(original_receipt)
    reformatted = _strict_json_file(receipt_path, "synthetic receipt")
    receipt_path.write_bytes(
        (json.dumps(reformatted, ensure_ascii=True) + "\n").encode("utf-8")
    )
    _expect_failure(
        "retained receipt byte reformat/reorder",
        lambda: _revalidate_existing_receipt(receipt_path, receipt),
    )
    receipt_path.write_bytes(original_receipt)

    race_path = (Path(raw) / "race_authorization.json").resolve()
    original_link = os.link

    def racing_link(source: Any, destination: Any) -> None:
        Path(destination).write_bytes(b"competitor-create-once\n")
        original_link(source, destination)

    os.link = racing_link
    try:
        _expect_failure(
            "atomic create-once receipt race",
            lambda: _publish_receipt(race_path, receipt),
        )
    finally:
        os.link = original_link
    if race_path.read_bytes() != b"competitor-create-once\n":
        raise AssertionError("receipt race overwrote competitor bytes")
    race_path.unlink()

    poisoned_path = (Path(raw) / "poisoned_authorization.json").resolve()

    def poisoning_link(source: Any, destination: Any) -> None:
        original_link(source, destination)
        Path(source).write_bytes(b"post-link-poison\n")

    os.link = poisoning_link
    try:
        _expect_failure(
            "receipt bytes altered through temporary hardlink",
            lambda: _publish_receipt(poisoned_path, receipt),
        )
    finally:
        os.link = original_link
    if poisoned_path.read_bytes() != b"post-link-poison\n":
        raise AssertionError("receipt hardlink poison probe did not execute")
    poisoned_path.unlink()

    policy_mat = gate_dir / "closure_policy.mat"
    original_policy_mat = policy_mat.read_bytes()
    policy_mat.write_bytes(original_policy_mat + b"tamper")
    _expect_failure(
        "MAT artifact tamper on retained receipt revalidation",
        lambda: _authorize_gate(
            gate_dir,
            commit,
            receipt_path,
            revalidate_receipt=True,
            git_check=no_git,
            dataset_validator=no_dataset,
            mat_validator=no_mat,
        ),
    )
    policy_mat.write_bytes(original_policy_mat)

    toctou_mat = gate_dir / "cases" / "0001_case.mat"
    original_toctou_mat = toctou_mat.read_bytes()

    def mutate_after_mat_validation(*_args: Any, **_kwargs: Any) -> None:
        toctou_mat.write_bytes(original_toctou_mat + b"post-validation")

    _expect_failure(
        "post-MAT-validation TOCTOU replacement",
        lambda: verify_gate(
            gate_dir,
            commit,
            git_check=no_git,
            dataset_validator=no_dataset,
            mat_validator=mutate_after_mat_validation,
        ),
    )
    toctou_mat.write_bytes(original_toctou_mat)

    forged_toctou_mat = original_toctou_mat + b"forged-before-snapshot"
    toctou_mat.write_bytes(forged_toctou_mat)

    def aba_during_mat_validation(
        frozen_gate_dir: Path, *_args: Any, **_kwargs: Any,
    ) -> None:
        toctou_mat.write_bytes(original_toctou_mat)
        try:
            frozen_bytes = (
                Path(frozen_gate_dir)
                / "cases"
                / "0001_case.mat"
            ).read_bytes()
            if frozen_bytes != forged_toctou_mat:
                raise AssertionError(
                    "validator received the mutable live gate, not a "
                    "private frozen snapshot")
            raise GateError(
                "forged pre-snapshot MAT remains forged for validation")
        finally:
            toctou_mat.write_bytes(forged_toctou_mat)

    _expect_failure(
        "ABA substitution cannot alter private validation snapshot",
        lambda: verify_gate(
            gate_dir,
            commit,
            git_check=no_git,
            dataset_validator=no_dataset,
            mat_validator=aba_during_mat_validation,
        ),
    )
    toctou_mat.write_bytes(original_toctou_mat)

    live_dataset_dir = Path(raw) / f"dataset_{STAGES[0]}"
    live_state = live_dataset_dir / "0001.mat"
    original_live_state = live_state.read_bytes()

    def mutate_live_dataset_after_validation(
        frozen_paths: list[Path],
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        if (
            frozen_paths[0] / "0001.mat"
        ).read_bytes() != original_live_state:
            raise AssertionError("dataset validator did not receive snapshot")
        live_state.write_bytes(
            original_live_state + b"post-dataset-validation")

    _expect_failure(
        "post-validation qualification dataset replacement",
        lambda: verify_gate(
            gate_dir,
            commit,
            git_check=no_git,
            dataset_validator=mutate_live_dataset_after_validation,
            mat_validator=no_mat,
        ),
    )
    live_state.write_bytes(original_live_state)

    def dataset_aba_probe(label: str, name: str) -> None:
        live_path = live_dataset_dir / name
        legitimate = live_path.read_bytes()
        forged = legitimate + b"forged-before-dataset-snapshot"
        live_path.write_bytes(forged)

        def aba_dataset_validator(
            frozen_paths: list[Path],
            *_args: Any,
            **_kwargs: Any,
        ) -> None:
            live_path.write_bytes(legitimate)
            try:
                frozen_bytes = (frozen_paths[0] / name).read_bytes()
                if frozen_bytes != forged:
                    raise AssertionError(
                        "dataset validator received mutable live bytes")
                raise GateError(
                    "forged dataset bytes remain forged for validation")
            finally:
                live_path.write_bytes(forged)

        try:
            _expect_failure(
                label,
                lambda: verify_gate(
                    gate_dir,
                    commit,
                    git_check=no_git,
                    dataset_validator=aba_dataset_validator,
                    mat_validator=no_mat,
                ),
            )
        finally:
            live_path.write_bytes(legitimate)

    dataset_aba_probe(
        "state MAT ABA cannot alter private dataset snapshot",
        "0001.mat",
    )
    dataset_aba_probe(
        "file_digests MAT ABA cannot alter private dataset snapshot",
        "file_digests.mat",
    )

    verifier_module = ROOT / "contact_gate_path_safety.py"
    original_verifier_module = verifier_module.read_bytes()

    def mutate_transitive_verifier_source(
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        verifier_module.write_bytes(
            original_verifier_module
            + b"\n# intra-verification drift probe\n"
        )

    try:
        _expect_failure(
            "transitive verifier module changed during validation",
            lambda: verify_gate(
                gate_dir,
                commit,
                git_check=no_git,
                dataset_validator=no_dataset,
                mat_validator=mutate_transitive_verifier_source,
            ),
        )
    finally:
        verifier_module.write_bytes(original_verifier_module)
    assert receipt["checker_sha256"] == verifier_source_root()

    def inventory_mutation(
        label: str, path: Path, *, directory: bool = False,
    ) -> None:
        if directory:
            path.mkdir()
        else:
            path.write_bytes(b"foreign\n")
        try:
            _expect_failure(
                label,
                lambda: verify_gate(
                    gate_dir,
                    commit,
                    git_check=no_git,
                    dataset_validator=no_dataset,
                    mat_validator=no_mat,
                ),
            )
        finally:
            if directory:
                path.rmdir()
            else:
                path.unlink()

    inventory_mutation(
        "foreign top-level artifact", gate_dir / "foreign.tmp")
    inventory_mutation(
        "stale case artifact",
        gate_dir / "cases" / f"{EXPECTED_TOTAL_CASES + 1:04d}_case.json",
    )
    inventory_mutation(
        "case subdirectory", gate_dir / "cases" / "nested",
        directory=True,
    )
    inventory_mutation(
        "ambiguous final-plus-temp artifact",
        gate_dir / "gate_summary.json.tmp",
    )
    symlink_path = gate_dir / "cases" / "foreign_link"
    try:
        os.symlink(gate_dir / "GATE_STATUS.txt", symlink_path)
    except (OSError, NotImplementedError):
        print("[SKIP] symlink mutation unavailable on this Windows host")
    else:
        try:
            _expect_failure(
                "symlink case artifact",
                lambda: _validate_gate_inventory(gate_dir),
            )
        finally:
            symlink_path.unlink()

    summary_path = gate_dir / "gate_summary.json"
    original_summary = summary_path.read_bytes()

    def reject_summary(label: str, mutate: Any) -> None:
        summary = _strict_json_file(summary_path, "synthetic summary")
        mutate(summary)
        _write_canonical_json(summary_path, summary)
        try:
            _expect_failure(
                label,
                lambda: verify_gate(
                    gate_dir,
                    commit,
                    git_check=no_git,
                    dataset_validator=no_dataset,
                    mat_validator=no_mat,
                ),
            )
        finally:
            summary_path.write_bytes(original_summary)

    reject_summary(
        "summary extra field",
        lambda summary: summary.__setitem__("foreign", True),
    )
    reject_summary(
        "summary missing field",
        lambda summary: summary.pop("matlab_release"),
    )
    reject_summary(
        "summary MATLAB environment drift",
        lambda summary: summary.__setitem__(
            "matlab_environment_sha256", "0" * 64),
    )
    reject_summary(
        "summary gate-module execution root drift",
        lambda summary: summary.__setitem__(
            "gate_execution_root_sha256", "0" * 64),
    )
    reject_summary(
        "summary dataset descriptor drift",
        lambda summary: summary["datasets"][0].__setitem__(
            "gen_fingerprint", "0" * 64),
    )
    reject_summary(
        "closure host differs from qualification receipts",
        lambda summary: summary["closure_host_attestation"].__setitem__(
            "hostname", "different-physical-host"),
    )
    print("[PASS] top-level/summary/inventory mutation matrix")


def run_self_tests() -> None:
    """Run every self-test domain in order.

    Split by domain (P1-4): this driver used to be one 751-line function, which
    made it impossible to see which guarantee a given probe belonged to or to
    run one domain while developing it.  Each helper below owns exactly one
    domain and is named for the guarantee it defends; the shared synthetic gate
    is built once here and threaded explicitly, so no helper depends on
    fixture state established by an earlier one except where the signature says
    so.
    """
    _check_inventory_and_numerics()
    _check_strict_parsing_and_inventory_mutations()
    _check_source_contract()

    no_git = lambda _commit: None
    no_dataset = lambda *_args, **_kwargs: None
    no_mat = lambda *_args, **_kwargs: None
    with tempfile.TemporaryDirectory(prefix="contact_gate_selftest_") as raw:
        fixture = _build_synthetic_gate(Path(raw))
        _check_case_evidence(fixture, raw)
        receipt_path, receipt = _check_receipt_publication(
            fixture, raw, no_git, no_dataset, no_mat
        )
        _check_receipt_and_summary_mutations(
            fixture, raw, receipt_path, receipt, no_git, no_dataset, no_mat
        )

    print("CONTACT CLOSURE CHECKER SELF-TESTS: ALL PASS")

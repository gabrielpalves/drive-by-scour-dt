"""Static MATLAB execution contract for the contact-closure verifier.

The MATLAB study and gate are intentionally split into one-purpose functions.
This module owns the independent Python inventories and the small set of
source-level invariants that make those inventories auditable.  In particular,
``A02_Track`` must call the reviewed TrackProp function directly; dynamic
``run``/``eval`` dispatch is forbidden throughout the reachable contact chain.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

from contact_gate_path_safety import GateError


ROOT = Path(__file__).resolve().parent

# Exact MATLAB study executable closure. Keep synchronized with
# scour_MATLAB/contact_study_harness_files.m.
STUDY_HARNESS_FILES = (
    "contact_absolute_path.m",
    "contact_allclose.m",
    "contact_assert_file_snapshot_unchanged.m",
    "contact_assert_manifest_snapshot.m",
    "contact_assert_reviewed_bootstrap.m",
    "contact_assert_snapshot_set_unchanged.m",
    "contact_bytes_sha256.m",
    "contact_capture_file_snapshot.m",
    "contact_case_info_from_snapshot.m",
    "contact_case_state_count.m",
    "contact_case_text.m",
    "contact_channel_metric_table.m",
    "contact_channel_qoi_table.m",
    "contact_closure_common.m",
    "contact_closure_study.m",
    "contact_comparison_path.m",
    "contact_damage_descriptor.m",
    "contact_delete_file_entry_if_present.m",
    "contact_descriptor_summary.m",
    "contact_exact_manifest_text.m",
    "contact_file_bytes.m",
    "contact_file_observation.m",
    "contact_file_sha256.m",
    "contact_filesystem_identity.m",
    "contact_gate_pass.m",
    "contact_indexed_value.m",
    "contact_is_same_or_child.m",
    "contact_java_boolean_value.m",
    "contact_load_mat_bytes.m",
    "contact_load_study_dataset_snapshots.m",
    "contact_logical_scalar.m",
    "contact_maybe_write_report.m",
    "contact_named_file_snapshot.m",
    "contact_nearest_dt_index.m",
    "contact_nonnegative_scalar.m",
    "contact_nonnegative_vector.m",
    "contact_numeric_sha256.m",
    "contact_path_component_is_link_alias.m",
    "contact_positive_scalar.m",
    "contact_positive_vector.m",
    "contact_profile_descriptor.m",
    "contact_qualification_script_identity.m",
    "contact_regular_nonsymlink.m",
    "contact_regular_nonsymlink_directory.m",
    "contact_replace_unique_bytes.m",
    "contact_resolved_module_root.m",
    "contact_run_one.m",
    "contact_run_small_process.m",
    "contact_saved_baseline_comparison.m",
    "contact_select_passage.m",
    "contact_solver_modules.m",
    "contact_solver_source_manifest.m",
    "contact_stable_file_bytes.m",
    "contact_state_text.m",
    "contact_study_harness_files.m",
    "contact_study_harness_root.m",
    "contact_study_metrics.m",
    "contact_study_reconstruction.m",
    "contact_study_report.m",
    "contact_study_solver.m",
    "contact_text_scalar.m",
    "contact_text_sha256.m",
    "contact_unlinked_path_identity.m",
    "contact_utc_now.m",
    "contact_validate_case.m",
    "contact_validate_completion_marker_snapshot.m",
    "contact_validate_host_receipt.m",
    "contact_validate_r11_descriptor.m",
    "contact_verify_dataset_integrity.m",
    "contact_windows_file_identity.m",
    "contact_write_markdown.m",
    "current_matlab_environment.m",
    "generator_source_root.m",
    "matlab_environment_identity.m",
    "validate_dataset_digest_manifest.m",
)

# Exact MATLAB gate executable closure. Keep synchronized with
# scour_MATLAB/contact_gate_module_files.m.
GATE_SOURCE_FILES = (
    "contact_absolute_path.m",
    "contact_allclose.m",
    "contact_assert_closure_host_matches_datasets.m",
    "contact_assert_file_snapshot_unchanged.m",
    "contact_assert_manifest_snapshot.m",
    "contact_assert_numbered_state_inventory.m",
    "contact_assert_reviewed_bootstrap.m",
    "contact_assert_snapshot_set_unchanged.m",
    "contact_bytes_sha256.m",
    "contact_capture_file_snapshot.m",
    "contact_closure_common.m",
    "contact_closure_gate.m",
    "contact_closure_host_attestation.m",
    "contact_closure_host_text.m",
    "contact_comparison_path.m",
    "contact_contracts_to_finest.m",
    "contact_delete_file_entry_if_present.m",
    "contact_exact_manifest_text.m",
    "contact_file_bytes.m",
    "contact_file_observation.m",
    "contact_file_sha256.m",
    "contact_filesystem_identity.m",
    "contact_gate_accept_report.m",
    "contact_gate_acceptance.m",
    "contact_gate_build_selection.m",
    "contact_gate_case_artifact_root.m",
    "contact_gate_case_result_skeleton.m",
    "contact_gate_count_case_files.m",
    "contact_gate_execution_root.m",
    "contact_gate_module_files.m",
    "contact_gate_plain_report.m",
    "contact_gate_plain_table.m",
    "contact_gate_policy.m",
    "contact_gate_policy_definition.m",
    "contact_gate_policy_descriptor.m",
    "contact_gate_public_case.m",
    "contact_gate_publish_or_validate_summary_sidecars.m",
    "contact_gate_publish_summary_mat.m",
    "contact_gate_recover_interrupted_temps.m",
    "contact_gate_recover_one_temp.m",
    "contact_gate_save_atomic.m",
    "contact_gate_selection.m",
    "contact_gate_summary_skeleton.m",
    "contact_gate_validate_canonical_case.m",
    "contact_gate_validate_case_inventory.m",
    "contact_gate_validate_output_inventory.m",
    "contact_gate_validate_publication.m",
    "contact_gate_validate_report_binding.m",
    "contact_gate_validate_solver_execution_manifest.m",
    "contact_gate_verify_frozen_plan.m",
    "contact_gate_write_or_verify_text.m",
    "contact_gate_write_text_atomic.m",
    "contact_gci_bound.m",
    "contact_is_same_or_child.m",
    "contact_java_boolean_value.m",
    "contact_load_mat_bytes.m",
    "contact_logical_scalar.m",
    "contact_numeric_sha256.m",
    "contact_numeric_text.m",
    "contact_path_component_is_link_alias.m",
    "contact_qualification_script_identity.m",
    "contact_regular_nonsymlink.m",
    "contact_regular_nonsymlink_directory.m",
    "contact_replace_unique_bytes.m",
    "contact_resolved_module_root.m",
    "contact_run_small_process.m",
    "contact_selection_descriptor.m",
    "contact_snapshot_receipt.m",
    "contact_stable_file_bytes.m",
    "contact_text_scalar.m",
    "contact_text_sha256.m",
    "contact_unlinked_path_identity.m",
    "contact_utc_now.m",
    "contact_validate_completion_marker_snapshot.m",
    "contact_validate_host_receipt.m",
    "contact_validate_locked_matlab_environment.m",
    "contact_windows_file_identity.m",
    "validate_dataset_digest_manifest.m",
)

# Production solver functions in the exact root-hash order. Keep synchronized
# with scour_MATLAB/contact_solver_modules.m.
SOLVER_MODULES = (
    "A01_Train",
    "A02_Track",
    "A03_Bridge",
    "A04_Options",
    "B00_Calculations",
    "B01_ElementsAndCoordinates",
    "B02_BoundaryConditions",
    "B03_BeamMatrices",
    "B07_OptionsProcessing",
    "B08_VehFreq",
    "B09_BeamFrq",
    "B10_EndTime",
    "B11_TimeSpaceDiscretization",
    "B14_EqVertNodalForce",
    "B17_CalcUat",
    "B18_TrainVehEq",
    "B19_GenerateProfile",
    "B24_BeamDamping",
    "B25_WheelProfiles",
    "B31_BeamBM",
    "B33_BeamShear",
    "B43_ModelGeometry",
    "B47_VehStaticLoads",
    "B49_BeamDeformation",
    "B50_ElementNumOfForce",
    "B51_RailVariables",
    "B53_BeamAcceleration",
    "B54_ModelMatrices",
    "B54_TrackVectors",
    "B55_ModelBC",
    "B56_ModelFrq",
    "B58_ResultsBeamSections",
    "B64_Coupled_InitialStatic",
    "B65_DynamicCalcCoupledFaster",
    "B66_ContactForce",
    "TrackProp_Zhai_et_al_WithBallastOnBridge",
    "TrainProp_ObrienCalibrate",
)
SOLVER_SOURCE_FILES = tuple(f"{name}.m" for name in SOLVER_MODULES)

# Compatibility name retained for callers of the previous checker API.  The
# file now contains a direct function call, not dynamic dispatch.
DYNAMIC_DISPATCH_FILES = ("A02_Track.m",)
TRACK_DEFINITION_FILES = DYNAMIC_DISPATCH_FILES
REVIEWED_TRACKPROP_CALL = (
    "Track = TrackProp_Zhai_et_al_WithBallastOnBridge(Track);"
)
REVIEWED_TRACKPROP_DISPATCH = REVIEWED_TRACKPROP_CALL
FORBIDDEN_TRACKPROP_TARGETS = (
    "TrackProp_Zhai_et_al_NoBallastOnBridge",
)
_DYNAMIC_DISPATCH_PATTERNS = tuple(
    (name, re.compile(rf"(?<![\w.]){name}\s*\("))
    for name in ("run", "eval", "feval", "evalin", "assignin", "str2func")
)


def _matlab_statements(source: str) -> str:
    """Drop whole-line MATLAB comments, leaving executable statements."""
    kept: list[str] = []
    in_block = False
    for raw in source.splitlines():
        stripped = raw.strip()
        if in_block:
            if stripped == "%}":
                in_block = False
            continue
        if stripped == "%{":
            in_block = True
            continue
        if stripped.startswith("%"):
            continue
        kept.append(raw)
    return "\n".join(kept)


def _contact_source_set(
    names: Sequence[str],
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Read reviewed contact sources, with in-memory self-test overrides."""
    sources: dict[str, str] = {}
    for name in names:
        if overrides is not None and name in overrides:
            sources[name] = overrides[name]
            continue
        path = ROOT / "scour_MATLAB" / name
        try:
            sources[name] = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise GateError(
                f"contact source {name} is unreadable: {exc}"
            ) from exc
    return sources


def _validate_matlab_source_contract(
    study_sources: dict[str, str],
    gate_sources: dict[str, str],
    dispatch_sources: dict[str, str] | None = None,
    solver_sources: dict[str, str] | None = None,
) -> None:
    """Pin Paper-1 invariants in the one-purpose MATLAB files that own them."""
    seed_assignment = (
        "cfg.phase_seed = descriptor_contract.profile_phase_seed;"
    )
    window_assignment = (
        "x_hi_requested = 10 + double(case_info.L_bridge_m) + 18.30;"
    )
    seed_owner = "contact_profile_descriptor.m"
    window_owner = "contact_closure_study.m"
    required_identity_file = {
        "contact_file_observation.m": (
            "'basic:size,lastModifiedTime'",
            "confirmed_identity = contact_unlinked_path_identity(path);",
            "~isequal(confirmed_identity, identity)",
            "~contact_regular_nonsymlink(path)",
            "'file_key', identity.file_key",
        ),
        "contact_filesystem_identity.m": (
            "'basic:fileKey'",
            "if ~isempty(file_key)",
            "key_text = strtrim(char(file_key.toString()));",
            "if ~isempty(key_text)",
            "identity = ['nio|' key_text];",
            "identity = ['windows|' contact_windows_file_identity(path)];",
            "'unix:dev'",
            "'unix:ino'",
        ),
        "contact_java_boolean_value.m": (
            "if islogical(raw) && isscalar(raw)",
            "if isnumeric(raw) && isscalar(raw) && isfinite(raw)",
            "(raw == 0 || raw == 1)",
            "javaMethod('booleanValue', raw)",
            "islogical(converted) && isscalar(converted)",
        ),
        "contact_path_component_is_link_alias.m": (
            "contact_java_boolean_value(symbolic)",
            "contact_java_boolean_value(other)",
        ),
        "contact_regular_nonsymlink.m": (
            "system_directory = char(System.Environment.SystemDirectory);",
            "contact_run_small_process( ...",
            "{executable, 'hardlink', 'list'",
            "tf = exit_code == 0 && link_count == 1;",
        ),
        "contact_run_small_process.m": (
            "javaObject('java.lang.ProcessBuilder', native_arguments)",
            "javaMethod('destroyForcibly', process)",
            "'waitFor', process, int64(timeout_seconds), seconds",
            "if ~finished",
            "javaMethod('exitValue', process)",
            "reader.readLine()",
        ),
        "contact_unlinked_path_identity.m": (
            "absolute_nio = file_obj.toPath().toAbsolutePath().normalize();",
            "contact_path_component_is_link_alias(char(cursor.getPath()))",
            "canonical_native = char(file_obj.getCanonicalPath());",
            "if ~strcmp(absolute_comparison, canonical_comparison)",
            "file_key = contact_filesystem_identity(absolute_native);",
            "'basic:isDirectory'",
            "is_directory = contact_java_boolean_value(directory_value);",
            "confirmed_file_key = contact_filesystem_identity(absolute_native);",
            "~strcmp(file_key, confirmed_file_key)",
        ),
        "contact_windows_file_identity.m": (
            "system_directory = char(System.Environment.SystemDirectory);",
            "'getFileStore', 'java.nio.file.Files', absolute_nio",
            "volume_before_value = javaMethod( ...",
            "contact_run_small_process( ...",
            "{executable, 'file', 'queryfileid'",
            "if exit_code ~= 0 || numel(matches) ~= 1",
            "all(file_id == '0') || all(file_id == 'f')",
            "volume_after_value = javaMethod( ...",
            "if ~strcmp(volume_before, volume_after)",
            "identity = ['volume-vsn=' volume_before '|file-id=' file_id];",
        ),
    }
    required_study_file = {
        **required_identity_file,
        "contact_validate_r11_descriptor.m": (
            "StateNamedStreamSeedID",
            "state_stream_names",
            "'profile-phase'",
            "audit-2026-08-09-r12",
            "generation-rules-v8",
            "physical8_v1",
        ),
        "contact_run_one.m": (
            "which('ttbi.wheel_contact_kinematics')",
            "ttbi.wheel_contact_kinematics(",
            "wheelset_acceleration(1:2, :)",
            "wheelset_1_constrained_vertical_acceleration_proxy",
            "wheelset_2_constrained_vertical_acceleration_proxy",
        ),
        "contact_saved_baseline_comparison.m": (
            "AcelWheelsetPrimVag",
            "physical8_v1",
            "saved_wheelset(1:2, :)",
        ),
        "contact_validate_case.m": (
            "BadVehicleProps",
        ),
        window_owner: (
            "report.study_schema = 'contact-closure-v3';",
            "report.generation_behavior_version = recon.case_text(",
            "report.channel_schema_id = recon.case_text(",
            "direct_reconstruction_pass",
            "saved_contact_reconstruction_pass",
            "'contact_filesystem_identity';",
            "'contact_java_boolean_value';",
            "'contact_run_small_process';",
            "'contact_windows_file_identity'",
            "dataset_identity = contact_unlinked_path_identity(dataset_dir);",
            "dataset_dir = dataset_identity.canonical_path;",
            "dataset_snapshot = recon.load_dataset_snapshots(",
            "report.state_file_sha256 = dataset_snapshot.state_snapshot.sha256;",
            "contact_assert_snapshot_set_unchanged(publication_snapshots);",
        ),
        "contact_load_study_dataset_snapshots.m": (
            "contact_capture_file_snapshot(case_path);",
            "contact_load_mat_bytes(case_snapshot.bytes);",
            "'RetainSnapshots', true",
            "contact_named_file_snapshot(",
            "contact_assert_snapshot_set_unchanged(snapshots);",
        ),
        "contact_verify_dataset_integrity.m": (
            "contact_assert_manifest_snapshot(strict_manifest, case_snapshot);",
            "contact_assert_manifest_snapshot(strict_manifest, states_snapshot);",
            "contact_assert_manifest_snapshot(strict_manifest, state_file_snapshot);",
            "~isequal(snapshots, strict_manifest.retained_snapshots)",
            "contact_validate_completion_marker_snapshot(",
            "contact_validate_host_receipt(",
            "[snapshots; sidecar_snapshots]",
        ),
        "validate_dataset_digest_manifest.m": (
            "dataset_identity = contact_unlinked_path_identity(dataset_dir);",
            "'dataset_digest_manifest:LinkedDataset'",
            "dataset_dir = dataset_identity.canonical_path;",
            "contact_capture_file_snapshot(manifest_path);",
            "blob = contact_load_mat_bytes(manifest_snapshot.bytes);",
            "manifest.file_digests_snapshot = manifest_snapshot;",
            "manifest.retained_snapshots = retained_snapshots;",
            "contact_assert_file_snapshot_unchanged( ...",
            "dataset_identity_after = contact_unlinked_path_identity(dataset_dir);",
            "if ~isequal(dataset_identity_after, dataset_identity)",
        ),
        "contact_solver_modules.m": (
            "'TrackProp_Zhai_et_al_WithBallastOnBridge';",
        ),
    }
    required_gate_file = {
        **required_identity_file,
        "contact_gate_policy_definition.m": (
            "expected_cases = 420",
            "policy.stages = {'F40-S', 'F40-M', 'L99-S', 'L99-M'}",
            "actual-step-generalized-richardson-v1",
            "bounded-numerical-tension-engineering-v1",
            "physical8_v1",
        ),
        "contact_gate_accept_report.m": (
            "contact_gci_bound",
            "0/12/24-kN classification changed with dt",
            "policy.channel_schema_id",
        ),
        "contact_gate_case_artifact_root.m": (
            "function root = contact_gate_case_artifact_root",
        ),
        "contact_gate_recover_interrupted_temps.m": (
            "function contact_gate_recover_interrupted_temps",
        ),
        "contact_gate_plain_report.m": (
            "function plain = contact_gate_plain_report",
        ),
        "contact_closure_gate.m": (
            "'canonical_case'",
            "'contact_filesystem_identity';",
            "'contact_java_boolean_value';",
            "'contact_run_small_process';",
            "'contact_windows_file_identity'",
            "dataset_identity = contact_unlinked_path_identity(dataset_dirs{k});",
            "dataset_dirs{k} = dataset_identity.canonical_path;",
        ),
        "contact_gate_build_selection.m": (
            "strict_manifest.file_digests_snapshot",
            "contact_capture_file_snapshot(",
            "contact_load_mat_bytes(case_snapshot.bytes);",
            "contact_load_mat_bytes(states_snapshot.bytes);",
            "contact_load_mat_bytes(state_snapshot.bytes);",
            "contact_assert_manifest_snapshot(strict_manifest, case_snapshot);",
            "contact_assert_manifest_snapshot(strict_manifest, states_snapshot);",
            "contact_assert_manifest_snapshot(strict_manifest, state_snapshot);",
            "contact_validate_completion_marker_snapshot(",
            "state_sha = state_snapshot.sha256;",
            "datasets(stage_index).case_info_sha256 = "
            "case_snapshot.sha256;",
            "datasets(stage_index).damage_states_sha256 = "
            "states_snapshot.sha256;",
            "datasets(stage_index).file_digests_sha256 = "
            "manifest_snapshot.sha256;",
            "datasets(stage_index).completion_marker_sha256 = "
            "marker_snapshot.sha256;",
            "datasets(stage_index).qualification_host_receipt_sha256 = ...\n"
            "        receipt_snapshot.sha256;",
            "contact_snapshot_receipt(state_snapshot)",
            "all_dataset_snapshots{stage_index});",
            "contact_assert_snapshot_set_unchanged(",
        ),
    }
    forbidden = (
        "1e9 + damage_seed * 100000 + state_index",
        "cfg.phase_seed = state_seed",
        "cfg.phase_seed = descriptor_contract.profile_phase_seed +",
        "cfg.phase_seed = mod(",
        "+ 18.31",
    )
    if solver_sources is None:
        solver_sources = _contact_source_set(SOLVER_SOURCE_FILES)
    if dispatch_sources is None:
        dispatch_sources = {
            name: solver_sources[name]
            for name in TRACK_DEFINITION_FILES
            if name in solver_sources
        }
    if set(study_sources) != set(STUDY_HARNESS_FILES):
        raise GateError(
            "study source set is not the frozen study executable set"
        )
    if set(gate_sources) != set(GATE_SOURCE_FILES):
        raise GateError("gate source set is not the reviewed gate file set")
    study_code = {
        name: _matlab_statements(text)
        for name, text in study_sources.items()
    }
    gate_code = {
        name: _matlab_statements(text)
        for name, text in gate_sources.items()
    }
    seed_total = sum(
        code.count(seed_assignment) for code in study_code.values()
    )
    if seed_total != 1 or study_code[seed_owner].count(seed_assignment) != 1:
        raise GateError(
            "the frozen profile-phase seed must be assigned exactly once "
            f"across the study file set, in {seed_owner}"
        )
    window_total = sum(
        code.count(window_assignment) for code in study_code.values()
    )
    if (
        window_total != 1
        or study_code[window_owner].count(window_assignment) != 1
    ):
        raise GateError(
            "the registered comparison window (10 + L_bridge + 18.30 m) "
            f"must be built exactly once, in {window_owner}"
        )
    for name, tokens in required_study_file.items():
        for token in tokens:
            if token not in study_code[name]:
                raise GateError(
                    f"study source {name} lacks Paper-1 evidence guard {token!r}"
                )
    for name, tokens in required_gate_file.items():
        for token in tokens:
            if token not in gate_code[name]:
                raise GateError(
                    f"gate source {name} lacks Paper-1 evidence guard {token!r}"
                )
    gate_selection = gate_code["contact_gate_build_selection.m"]
    if gate_selection.count("contact_assert_snapshot_set_unchanged(") != 2:
        raise GateError(
            "contact_gate_build_selection.m must reassert each stage and "
            "the complete four-dataset snapshot exactly once"
        )
    if study_code[window_owner].count(
        "contact_assert_snapshot_set_unchanged(publication_snapshots);"
    ) != 4:
        raise GateError(
            "contact_closure_study.m must reassert before and after both "
            "dry-run and completed-report publication"
        )
    retired_gate_snapshot_reads = (
        "load(",
        "file_sha256(",
    )
    for token in retired_gate_snapshot_reads:
        if token in gate_selection:
            raise GateError(
                "contact_gate_build_selection.m reintroduces a pathname "
                f"parse/hash after authentication: {token!r}"
            )
    for name in (
        "contact_load_study_dataset_snapshots.m",
        "contact_validate_r11_descriptor.m",
        "contact_verify_dataset_integrity.m",
    ):
        for token in ("load(", "fileread(", "file_sha256("):
            if token in study_code[name]:
                raise GateError(
                    f"study source {name} reopens authenticated evidence "
                    f"by pathname: {token!r}"
                )
    for name, code in (*study_code.items(), *gate_code.items()):
        for token in forbidden:
            if token in code:
                raise GateError(
                    f"contact source {name} reintroduces retired construct "
                    f"{token!r}"
                )
    _validate_dynamic_dispatch(
        study_code,
        gate_code,
        solver_sources,
        dispatch_sources,
    )


def _validate_dynamic_dispatch(
    study_code: dict[str, str],
    gate_code: dict[str, str],
    solver_sources: dict[str, str],
    dispatch_sources: dict[str, str] | None = None,
) -> None:
    """Require direct TrackProp dispatch and a fully static solver closure."""
    if set(solver_sources) != set(SOLVER_SOURCE_FILES):
        raise GateError(
            "solver source set is not the frozen reachable solver closure"
        )
    if dispatch_sources is None:
        dispatch_sources = {
            name: solver_sources[name] for name in TRACK_DEFINITION_FILES
        }
    if set(dispatch_sources) != set(TRACK_DEFINITION_FILES):
        raise GateError(
            "track-definition source set is not the reviewed file set"
        )
    for name in TRACK_DEFINITION_FILES:
        if dispatch_sources[name] != solver_sources[name]:
            raise GateError(
                f"{name} has divergent track-definition and solver-closure "
                "source views"
            )
    solver_code = {
        name: _matlab_statements(text)
        for name, text in solver_sources.items()
    }
    all_code = (*study_code.items(), *gate_code.items(), *solver_code.items())
    for name, code in all_code:
        for label, pattern in _DYNAMIC_DISPATCH_PATTERNS:
            found = pattern.search(code)
            if found is not None:
                raise GateError(
                    f"contact source {name} introduces dynamic dispatch "
                    f"({label}) at offset {found.start()}; the executed-module "
                    "closure would no longer be statically auditable"
                )
    track = solver_code["A02_Track.m"]
    if track.count(REVIEWED_TRACKPROP_CALL) != 1:
        raise GateError(
            "A02_Track.m must call the reviewed track function exactly once "
            f"as {REVIEWED_TRACKPROP_CALL!r}"
        )
    for retired in FORBIDDEN_TRACKPROP_TARGETS:
        if retired in track:
            raise GateError(
                f"A02_Track.m activates the unreviewed track model {retired!r}"
            )
    if SOLVER_MODULES.count(
        "TrackProp_Zhai_et_al_WithBallastOnBridge"
    ) != 1:
        raise GateError(
            "the static solver closure must contain the reviewed TrackProp "
            "function exactly once"
        )

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
import os
from functools import lru_cache
from pathlib import Path
import re
import types

from core.campaign_contract import (
    EXPECTED_CHANNEL_SCHEMA_ID,
    EXPECTED_GENERATION_BEHAVIOR_VERSION,
    EXPECTED_GEN_SCHEMA,
    EXPECTED_PROTOCOL_SCHEMA_TAG,
)
from core.generation_state_contract import (
    STATE_DATA_FIELDS,
    STATE_TOP_LEVEL_FIELDS,
)


ROOT = Path(__file__).resolve().parent
A00_PATH = ROOT / "scour_MATLAB" / "A00_Run.m"
TTBI_PKG = ROOT / "scour_MATLAB" / "+ttbi"
# Injected only into the in-memory reviewed text, never on disk.
TTBI_MARK = "\n%%%TTBI-PACKAGE-FUNCTION:"


def ttbi_block(text: str, name: str) -> str:
    """Slice one +ttbi function out of the SUPPLIED generator text.

    Deliberately not a disk read.  The mutation harness proves each guard by
    rewriting the generator text in memory and requiring the guard to reject
    it; a guard that re-read the file from disk would be blind to those
    mutations and would pass unconditionally.  That is exactly what happened
    when these blocks first moved to the package, and the harness caught it.
    """
    header = f"{TTBI_MARK}{name}\n"
    start = text.index(header) + len(header)
    nxt = text.find(TTBI_MARK, start)
    return text[start:] if nxt < 0 else text[start:nxt]


def ttbi_source(name: str) -> str:
    """Return one +ttbi package function's exact source.

    The seed/UID derivation and the qualification-script canonicalisation used
    to be local functions inside A00_Run.m, so these guards sliced their bodies
    out of the A00 text by locating the next `function` header.  They now live
    one function per file, which makes the slice unnecessary: the whole file IS
    the function, so a guard can no longer read past its end or be confused by
    a reordering of A00's locals.
    """
    path = TTBI_PKG / f"{name}.m"
    if not path.is_file():
        raise ContractError(
            f"reviewed package function ttbi.{name} is missing: {path}"
        )
    return path.read_text(encoding="utf-8")
SAVE_PROGRESS_PATH = ROOT / "scour_MATLAB" / "save_progress.m"
ENV_IDENTITY_PATH = ROOT / "scour_MATLAB" / "matlab_environment_identity.m"
CURRENT_ENV_PATH = ROOT / "scour_MATLAB" / "current_matlab_environment.m"
SOURCE_ROOT_PATH = ROOT / "scour_MATLAB" / "generator_source_root.m"
CRN_SMOKE_PATH = ROOT / "scour_MATLAB" / "smoke_crn_state_design.m"
DAMAGE_PHYSICS_PATH = ROOT / "scour_MATLAB" / "B02_BoundaryConditions.m"
DAMAGE_MODAL_PATH = ROOT / "scour_MATLAB" / "B09_BeamFrq.m"
DAMAGE_DAMPING_PATH = ROOT / "scour_MATLAB" / "B24_BeamDamping.m"
DAMAGE_SMOKE_PATH = ROOT / "scour_MATLAB" / "smoke_damage_toggles.m"
MESH_OPTIONS_PATH = ROOT / "scour_MATLAB" / "A04_Options.m"
MESH_SELECTOR_PATH = (
    ROOT / "scour_MATLAB" / "bridge_mesh_elements_per_sleeper.m"
)
MESH_ALIGNMENT_SMOKE_PATH = (
    ROOT / "scour_MATLAB" / "smoke_bridge_mesh_alignment.m"
)
BALLAST_MODEL_PATH = ROOT / "scour_MATLAB" / "B54_ModelMatrices.m"
BALLAST_MIRROR_PATH = ROOT / "TTBI_2D" / "b54_model_matrices.py"
NUMERICAL_VV_PREFLIGHT_PATH = (
    ROOT / "scour_MATLAB" / "numerical_vv_coupled_mesh_preflight.m"
)
NUMERICAL_VV_SMOKE_PATH = (
    ROOT / "scour_MATLAB" / "smoke_numerical_vv_harness.m"
)
PROVENANCE_SMOKE_PATH = (
    ROOT / "scour_MATLAB" / "smoke_r11_provenance_serialization.m"
)
GENERATION_WORKER_SMOKE_PATH = (
    ROOT / "scour_MATLAB" / "smoke_generation_worker.m"
)
MAKE_MICRO_PATH = ROOT / "make_micro_smoke.py"
DATASET_PATH = ROOT / "core" / "dataset.py"
DRIVER_PATH = ROOT / "comprehensive_ablation_multidamage.py"
ENVIRONMENT_PATH = ROOT / "environment" / "campaign-py313-cu128.json"
SOURCE_MANIFEST_PATH = ROOT / "bundle_source_files.txt"

EXPECTED_SCHEMA = EXPECTED_GEN_SCHEMA
EXPECTED_BEHAVIOR_VERSION = EXPECTED_GENERATION_BEHAVIOR_VERSION
EXPECTED_STUDY_TAG = EXPECTED_PROTOCOL_SCHEMA_TAG
EXPECTED_CHANNEL_SCHEMA = EXPECTED_CHANNEL_SCHEMA_ID
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
DAMAGE_PHYSICS_SOURCE = DAMAGE_PHYSICS_PATH.read_text(encoding="utf-8")
DAMAGE_MODAL_SOURCE = DAMAGE_MODAL_PATH.read_text(encoding="utf-8")
DAMAGE_DAMPING_SOURCE = DAMAGE_DAMPING_PATH.read_text(encoding="utf-8")
DAMAGE_SMOKE_SOURCE = DAMAGE_SMOKE_PATH.read_text(encoding="utf-8")
MESH_OPTIONS_SOURCE = MESH_OPTIONS_PATH.read_text(encoding="utf-8")
MESH_SELECTOR_SOURCE = MESH_SELECTOR_PATH.read_text(encoding="utf-8")
MESH_ALIGNMENT_SMOKE_SOURCE = MESH_ALIGNMENT_SMOKE_PATH.read_text(
    encoding="utf-8"
)
BALLAST_MODEL_SOURCE = BALLAST_MODEL_PATH.read_text(encoding="utf-8")
BALLAST_MIRROR_SOURCE = BALLAST_MIRROR_PATH.read_text(encoding="utf-8")
NUMERICAL_VV_PREFLIGHT_SOURCE = NUMERICAL_VV_PREFLIGHT_PATH.read_text(
    encoding="utf-8"
)
NUMERICAL_VV_SMOKE_SOURCE = NUMERICAL_VV_SMOKE_PATH.read_text(encoding="utf-8")
PROVENANCE_SMOKE_SOURCE = PROVENANCE_SMOKE_PATH.read_text(encoding="utf-8")
GENERATION_WORKER_SMOKE_SOURCE = GENERATION_WORKER_SMOKE_PATH.read_text(
    encoding="utf-8"
)
MAKE_MICRO_SOURCE = MAKE_MICRO_PATH.read_text(encoding="utf-8")

MATLAB_EXECUTABLE_SUFFIX_RE = re.compile(
    r"\.(?:m|p|mlx|mex[^/]*)$", re.IGNORECASE
)


class ContractError(AssertionError):
    """A generation-identity invariant is absent or ambiguous."""


def _validate_damage_physics_contract(
    b02: str,
    b09: str,
    b24: str,
    smoke: str,
) -> None:
    """Bind the production support equations to an independent MATLAB oracle."""
    for token in (
        "DOF_Original_value = 344e6;",
        "retained_stiffness = 1.0 - ...\n"
        "    Damage.scour_rates(positive_vertical_supports);",
        "vert_stiff_values = retained_stiffness * DOF_Original_value;",
        "rot_stiff_values(positive_rotational_supports == 1) = ...\n"
        "    Damage.bearing_left;",
        "rot_stiff_values(positive_rotational_supports == "
        "Beam.BC.supp_num) = ...\n"
        "    Damage.bearing_right;",
        "any(Damage.scour_rates(:) < 0 | Damage.scour_rates(:) > 1)",
        "numel(Damage.scour_rates) ~= Beam.BC.supp_num",
        "Damage.bearing_left < 0",
        "Damage.bearing_right < 0",
        "Damage.scour_rates = Damage.scour_rates(:)';",
        "Beam.BC.loc = Beam.BC.loc(:)';",
        "rigid_vertical_nodes = Beam.BC.loc_ind(Beam.BC.vert_stiff == -1);",
        "rigid_rotation_constrained = any(Beam.BC.rot_stiff == -1);",
        "if numel(unique(rigid_vertical_nodes)) >= 2",
        "Beam.Modal.num_rigid_modes = 2 - rigid_constraint_rank;",
    ):
        _once(b02, token, f"B02 damage-physics invariant {token}")
    for token in (
        "[lambda,~] = local_sorted_eigenvalues( ...\n"
        "        lambda, Beam.Modal.num_rigid_modes);",
        "[lambda,k] = local_sorted_eigenvalues( ...\n"
        "        diag(lambda), Beam.Modal.num_rigid_modes);",
        "tol = 128 * eps(max(scale));",
        "if any(lambda < -tol)",
        "if sum(near_zero) ~= expected_rigid_modes",
        "lambda(near_zero) = 0;",
        "[lambda, order] = sort(lambda, 'ascend');",
    ):
        _once(b09, token, f"B09 modal-order invariant {token}")
    for token in (
        "ref_modes = (1:2) + Beam.Modal.num_rigid_modes;",
        "if numel(Beam.Modal.w) < ref_modes(end)",
        "if ~isreal(wr) || any(~isfinite(wr)) || any(wr <= 0)",
        "Beam.Damping.reference_mode_indices = ref_modes;",
        "Beam.Damping.rayleigh_alpha = aux1(1);",
        "Beam.Damping.rayleigh_beta = aux1(2);",
    ):
        _once(b24, token, f"B24 damping invariant {token}")
    for token in (
        "expected_dofs4 = [1 2 3 5 7 8];",
        "expected_k4 = [344e6 1.25e9 0.70*344e6 0.40*344e6 344e6 3.75e9];",
        "assert(max(abs(zeta12 - Beam.Damping.per/100)) < 1e-10, ...",
        "assert(issorted(Beam.Modal.w), ...",
        "assert(isequal(Beam.Damping.reference_mode_indices, [1 2]), ...",
        "'B02:ScourSupportCountMismatch'",
        "'B02:InvalidScourRates'",
        "'B02:InvalidBearingStiffness'",
        "assert(Beam4.Modal.num_rigid_modes == 0, ...",
        "assert(isempty(Rail4.BC.DOF_with_values), ...",
        "assert(Rail4.Modal.num_rigid_modes == 2, ...",
        "assert(OneSpring4.Modal.num_rigid_modes == 1, ...",
    ):
        if token not in smoke:
            raise ContractError(
                f"MATLAB analytic damage oracle missing: {token}"
            )


def _must_reject_damage_mapping(
    name: str,
    b02: str,
    smoke: str,
    b09: str = DAMAGE_MODAL_SOURCE,
    b24: str = DAMAGE_DAMPING_SOURCE,
) -> None:
    try:
        _validate_damage_physics_contract(b02, b09, b24, smoke)
    except ContractError:
        print(f"  [PASS] damage-physics mutation rejected: {name}")
        return
    raise AssertionError(
        f"mutation escaped damage-physics guard: {name}"
    )


def _validate_bridge_mesh_alignment_contract(
    options: str,
    selector: str,
    b02: str,
    smoke: str,
    b54: str,
    py_b54: str,
    vv_preflight: str,
    vv_smoke: str,
) -> None:
    """Bind production bridge meshing to exact equal-span support nodes."""
    for token in (
        "Beam.Mesh.Ele.num_per_spacing = "
        "bridge_mesh_elements_per_sleeper( ...\n"
        "    Beam.Prop.L, Beam.Prop.num_spans, Track.Sleeper.spacing, 2);",
        "Track.Rail.Mesh.Ele.num_per_spacing = 2;",
    ):
        _once(options, token, f"A04 aligned-mesh invariant {token}")
    for token in (
        "maximum_elements_per_bay = 64;",
        "span_length_m = bridge_length_m / num_spans;",
        "for candidate = minimum_elements_per_bay:maximum_elements_per_bay",
        "bridge_element_count = bridge_length_m / element_length_m;",
        "span_element_count = span_length_m / element_length_m;",
        "abs(bridge_element_count-round(bridge_element_count)) <= tolerance",
        "abs(span_element_count-round(span_element_count)) <= tolerance",
        "'bridge_mesh:NoSupportAlignedDensity'",
    ):
        _once(selector, token, f"bridge mesh selector invariant {token}")
    for token in (
        "Beam.BC.loc_realized = Beam.Mesh.Nodes.acum(Beam.BC.loc_ind);",
        "Beam.BC.loc_offset = Beam.BC.loc_realized - Beam.BC.loc;",
        "mesh_element_count = max(numel(Beam.Mesh.Nodes.acum) - 1, 0);",
        "summation_roundoff_factor = max(256, 2*mesh_element_count);",
        "Beam.BC.loc_tolerance = summation_roundoff_factor * eps(alignment_scale);",
        "abs(Beam.BC.loc_offset) > Beam.BC.loc_tolerance",
        "if ~isempty(misaligned_spring_supports)",
        "'B02:SupportNotOnNode'",
    ):
        _once(b02, token, f"B02 support-alignment invariant {token}")
    for token in (
        "'expected_elements_per_bay', {3, 2, 3}",
        "Beam.Mesh.Ele.num = 200;",
        "'B02:SupportNotOnNode'",
        "'bridge_mesh:NoSupportAlignedDensity'",
    ):
        if token not in smoke:
            raise ContractError(
                f"aligned bridge-mesh oracle missing: {token}"
            )
    for token in (
        "Zhai et al. (2004), Eq. (5) and Table 1",
        "Model.Mesh.Mg = funAdd1(Model.Mesh.Mg,...\n"
        "    Model.Mesh.DOF.beam_vert_under_sleeper,...\n"
        "    funDiag(Track.Sleeper.num_onbeam,"
        "Track.BallastOnBeam.Prop.m));",
    ):
        _once(b54, token, f"B54 support-point ballast-mass invariant {token}")
    if b54.count("Track.BallastOnBeam.Prop.m") != 2:
        raise ContractError(
            "MATLAB B54 must reference the on-bridge ballast mass exactly "
            "once in its pad-substitution branch and once in the canonical "
            "support-point mass addition"
        )
    for forbidden in (
        "Track.BallastOnBeam.Prop.m/Beam.Mesh.Ele.num_per_spacing",
        "Track.BallastOnBeam.Prop.m / Beam.Mesh.Ele.num_per_spacing",
    ):
        if forbidden in b54:
            raise ContractError(
                "MATLAB B54 reintroduced mesh-density-scaled ballast mass"
            )
    for token in (
        "Zhai et al. (2004), Eq. (5) and Table 1",
        "Model.Mesh.DOF.beam_vert_under_sleeper,\n"
        "        funDiag(Track.Sleeper.num_onbeam, "
        "Track.BallastOnBeam.Prop.m),",
    ):
        _once(
            py_b54,
            token,
            f"Python B54 support-point ballast-mass invariant {token}",
        )
    if py_b54.count("Track.BallastOnBeam.Prop.m") != 2:
        raise ContractError(
            "Python B54 must reference the on-bridge ballast mass exactly "
            "once in its pad-substitution branch and once in the canonical "
            "support-point mass addition"
        )
    if (
        "Track.BallastOnBeam.Prop.m / Beam.Mesh.Ele.num_per_spacing"
        in py_b54
    ):
        raise ContractError(
            "Python B54 reintroduced mesh-density-scaled ballast mass"
        )
    for token in (
        "addParameter(parser, 'Redux', 0, @local_logical_scalar);",
        "Calc.Options.redux = redux;",
        "summation_roundoff_factor = max(256, 2*bridge_element_count);",
        "alignment_tolerance_m = summation_roundoff_factor * ...",
        "assembled_ballast_mass = model_bridge_mass-Beam.Mesh.Mg;",
        "'numerical_vv:BallastMassAssemblyMismatch'",
    ):
        _once(
            vv_preflight,
            token,
            f"production-path ballast assembly oracle {token}",
        )
    for token in (
        "assert(aligned_L60.redux == 0);",
        "assert(aligned_L99.redux == 0);",
        "fine_L99 = numerical_vv_coupled_mesh_preflight(99.6, 4, 4, 4, ...\n"
        "    'Assemble', true, 'Redux', 1);",
        "for density = [12, 8; 24, 16]'",
        "for density = [8, 8; 16, 16]'",
    ):
        _once(vv_smoke, token, f"ballast assembly smoke invariant {token}")


def _must_reject_bridge_mesh_alignment(
    name: str,
    options: str = MESH_OPTIONS_SOURCE,
    selector: str = MESH_SELECTOR_SOURCE,
    b02: str = DAMAGE_PHYSICS_SOURCE,
    smoke: str = MESH_ALIGNMENT_SMOKE_SOURCE,
    b54: str = BALLAST_MODEL_SOURCE,
    py_b54: str = BALLAST_MIRROR_SOURCE,
    vv_preflight: str = NUMERICAL_VV_PREFLIGHT_SOURCE,
    vv_smoke: str = NUMERICAL_VV_SMOKE_SOURCE,
) -> None:
    try:
        _validate_bridge_mesh_alignment_contract(
            options,
            selector,
            b02,
            smoke,
            b54,
            py_b54,
            vv_preflight,
            vv_smoke,
        )
    except ContractError:
        print(f"  [PASS] bridge-mesh mutation rejected: {name}")
        return
    raise AssertionError(
        f"mutation escaped bridge-mesh alignment guard: {name}"
    )


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


def _is_matlab_executable(path: Path) -> bool:
    """Mirror the executable suffix boundary used by the MATLAB inventory."""

    return MATLAB_EXECUTABLE_SUFFIX_RE.search(path.name) is not None


def _validate_matlab_layout() -> None:
    """Validate the thin driver, one-function package, and manifest parity.

    Qualification is tied to semantic owners and exact manifest/disk parity,
    never to source line counts or helper counts.  Those numeric anchors made
    harmless comments and newly reviewed one-function helpers look like
    scientific drift while failing to say which invariant had changed.
    """

    a00_source = A00_PATH.read_text(encoding="utf-8")
    if re.search(r"(?m)^\s*function\b", a00_source):
        raise ContractError("A00_Run.m regained local function definitions")
    for token in (
        "campaign_config = ttbi.campaign_setup(campaign_setup_inputs);",
        "state_design = ttbi.build_state_design(campaign_config);",
        "identity = ttbi.build_generation_identity( ...",
        "ttbi.run_generation_states( ...",
        "ttbi.publish_generation_completion(run_folder, execution_context);",
    ):
        _once(a00_source, token, f"thin A00 orchestration owner {token}")

    package_paths = sorted(TTBI_PKG.glob("*.m"))
    contact_paths = sorted((ROOT / "scour_MATLAB").glob("contact_*.m"))
    required_package_owners = {
        "campaign_setup",
        "build_state_design",
        "state_uid",
        "build_generation_identity",
        "build_case_info",
        "execute_generation_state",
        "run_generation_states",
        "publish_generation_completion",
    }
    actual_package_owners = {path.stem for path in package_paths}
    if not required_package_owners <= actual_package_owners:
        raise ContractError(
            "reviewed +ttbi semantic owners are missing: "
            f"{sorted(required_package_owners - actual_package_owners)!r}"
        )
    for path in (*package_paths, *contact_paths):
        source = path.read_text(encoding="utf-8")
        header_matches = list(re.finditer(r"(?m)^\s*function\b", source))
        if not header_matches:
            raise ContractError(
                f"{path.name} does not contain its public MATLAB function"
            )
        signature_lines: list[str] = []
        for line in source[header_matches[0].start():].splitlines():
            signature_lines.append(line)
            if not line.rstrip().endswith("..."):
                break
        signature = "\n".join(signature_lines)
        if re.search(rf"\b{re.escape(path.stem)}\s*\(", signature) is None:
            raise ContractError(
                f"{path.name} does not declare its file-matched function"
            )

    manifest_entries = [
        line
        for line in SOURCE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if len(manifest_entries) != len(set(manifest_entries)):
        raise ContractError("bundle source inventory contains duplicates")
    manifest_executables = sorted(
        entry
        for entry in manifest_entries
        if entry.startswith("scour_MATLAB/")
        and MATLAB_EXECUTABLE_SUFFIX_RE.search(entry) is not None
    )
    matlab_root = ROOT / "scour_MATLAB"
    disk_executables: list[str] = []
    generated_roots = {"results", "results_sensitivity"}
    for directory, child_dirs, file_names in os.walk(matlab_root):
        directory_path = Path(directory)
        if directory_path == matlab_root:
            child_dirs[:] = [
                name
                for name in child_dirs
                if name.casefold() not in generated_roots
            ]
        for file_name in file_names:
            path = directory_path / file_name
            if not _is_matlab_executable(path):
                continue
            inside = path.relative_to(matlab_root)
            if len(inside.parts) == 1 and path.stem.startswith("micro_A00_"):
                continue
            disk_executables.append(path.relative_to(ROOT).as_posix())
    disk_executables.sort()

    if manifest_executables != disk_executables:
        manifest_set = set(manifest_executables)
        disk_set = set(disk_executables)
        raise ContractError(
            "bundle/disk MATLAB executable inventory diverged; "
            f"unmanifested={sorted(disk_set - manifest_set)!r}, "
            f"missing={sorted(manifest_set - disk_set)!r}"
        )


def _validate_helpers(a00: str, source_root: str) -> None:
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
        "check_contact_closure_gate.py",
        "check_dispatch_authorization.py",
        "check_numerical_vv_package.py",
        "check_qualification_receipt_inventory.py",
        "check_source_provenance.py",
        "core/source_provenance.py",
        "dispatch_authorization.py",
        "docs/damage_model_reference.md",
        "docs/dry_ballast_stiffness_sign_sensitivity.md",
        "docs/numerical_vv_protocol.md",
        "docs/shm_reviewer_readiness_plan.md",
        "qualification_receipt_inventory.py",
        "scour_MATLAB/+ttbi/assert_no_shadow_matlab_sources.m",
        "scour_MATLAB/+ttbi/assert_generation_output_directory.m",
        "scour_MATLAB/+ttbi/assert_reviewed_matlab_resolution.m",
        "scour_MATLAB/+ttbi/authenticate_generation_worker.m",
        "scour_MATLAB/+ttbi/build_generator_source_attestation.m",
        "scour_MATLAB/+ttbi/delete_file_entry_if_present.m",
        "scour_MATLAB/+ttbi/delete_generation_pool.m",
        "scour_MATLAB/+ttbi/directory_observation.m",
        "scour_MATLAB/+ttbi/dry_ballast_stiffness_arm.m",
        "scour_MATLAB/+ttbi/execute_generation_state.m",
        "scour_MATLAB/+ttbi/ensure_generation_output_directory.m",
        "scour_MATLAB/+ttbi/filesystem_identity.m",
        "scour_MATLAB/+ttbi/generation_publication_credential_names.m",
        "scour_MATLAB/+ttbi/hash_reviewed_source_entries.m",
        "scour_MATLAB/+ttbi/java_boolean_value.m",
        "scour_MATLAB/+ttbi/list_matlab_executable_files.m",
        "scour_MATLAB/+ttbi/publish_generation_completion.m",
        "scour_MATLAB/+ttbi/require_generation_worker_attestation.m",
        "scour_MATLAB/+ttbi/reviewed_source_entries.m",
        "scour_MATLAB/+ttbi/revoke_generation_publication.m",
        "scour_MATLAB/+ttbi/run_generation_states.m",
        "scour_MATLAB/+ttbi/run_small_process.m",
        "scour_MATLAB/+ttbi/smoke_generation_output_boundary.m",
        "scour_MATLAB/+ttbi/stable_file_sha256.m",
        "scour_MATLAB/+ttbi/validate_repository_relative_path.m",
        "scour_MATLAB/+ttbi/windows_file_identity.m",
        "scour_MATLAB/+ttbi/write_generation_marker_temp.m",
        "scour_MATLAB/contact_closure_gate.m",
        "scour_MATLAB/contact_closure_study.m",
        "scour_MATLAB/bridge_mesh_elements_per_sleeper.m",
        "scour_MATLAB/current_matlab_environment.m",
        "scour_MATLAB/generator_source_root.m",
        "scour_MATLAB/matlab_environment_identity.m",
        "scour_MATLAB/numerical_vv_bridge_fixture.m",
        "scour_MATLAB/numerical_vv_coupled_mesh_preflight.m",
        "scour_MATLAB/numerical_vv_file_sha256.m",
        "scour_MATLAB/numerical_vv_micro_run.m",
        "scour_MATLAB/numerical_vv_protocol_definition.m",
        "scour_MATLAB/numerical_vv_scalar_convergence.m",
        "scour_MATLAB/numerical_vv_sha256_bytes.m",
        "scour_MATLAB/numerical_vv_support_alignment.m",
        "scour_MATLAB/numerical_vv_validate_package.m",
        "scour_MATLAB/numerical_vv_waveform_metrics.m",
        "scour_MATLAB/response_signature_damage_catalog.m",
        "scour_MATLAB/response_signature_metrics.m",
        "scour_MATLAB/response_signature_run_one.m",
        "scour_MATLAB/response_signature_study.m",
        "scour_MATLAB/smoke_bridge_mesh_alignment.m",
        "scour_MATLAB/smoke_crn_state_design.m",
        "scour_MATLAB/smoke_contact_closure.m",
        "scour_MATLAB/smoke_dry_ballast_sign_sensitivity.m",
        "scour_MATLAB/smoke_generation_worker.m",
        "scour_MATLAB/smoke_numerical_vv_harness.m",
        "scour_MATLAB/smoke_r11_provenance_serialization.m",
        "scour_MATLAB/smoke_response_signature_metrics.m",
        "scour_MATLAB/smoke_response_signature_pair.m",
        "scour_MATLAB/smoke_structural_oracle.m",
        "scour_MATLAB/validate_dataset_digest_manifest.m",
        "scour_MATLAB/vv_euler_bernoulli_reference.m",
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

    reviewed_entries = ttbi_block(a00, "reviewed_source_entries")
    path_validator = ttbi_block(a00, "validate_repository_relative_path")
    source_hasher = ttbi_block(a00, "hash_reviewed_source_entries")
    stable_hasher = ttbi_block(a00, "stable_file_sha256")
    file_observer = ttbi_block(a00, "file_observation")
    filesystem_identity = ttbi_block(a00, "filesystem_identity")
    java_boolean = ttbi_block(a00, "java_boolean_value")
    path_alias = ttbi_block(a00, "path_component_is_link_alias")
    process_runner = ttbi_block(a00, "run_small_process")
    windows_file_identity = ttbi_block(a00, "windows_file_identity")
    windows_hardlink_count = ttbi_block(a00, "windows_hardlink_count")
    shadow_guard = ttbi_block(a00, "assert_no_shadow_matlab_sources")
    resolution_guard = ttbi_block(a00, "assert_reviewed_matlab_resolution")
    executable_inventory = ttbi_block(a00, "list_matlab_executable_files")
    directory_observer = ttbi_block(a00, "directory_observation")

    # Bind the public source-root helper to each modular owner.  All package
    # sources come from the supplied in-memory generator text, so the mutation
    # harness cannot accidentally validate an unchanged file from disk.
    for token in (
        "[repository_root, entries] = ttbi.reviewed_source_entries();",
        "selected = sort(entries(startsWith(entries, 'scour_MATLAB/')));",
        "first_lines = ttbi.hash_reviewed_source_entries( ...",
        "[confirmed_root, confirmed_entries] = ttbi.reviewed_source_entries();",
        "confirmed_lines = ttbi.hash_reviewed_source_entries( ...",
        "if ~strcmp(first_lines, confirmed_lines)",
        "root_sha256 = ttbi.sha256(digest_lines);",
        "file_count = numel(selected);",
    ):
        _once(source_root, token, f"generator source-root binding {token}")
    if source_root.count("ttbi.assert_no_shadow_matlab_sources( ...") != 2:
        raise ContractError("source root does not repeat the shadow inventory")
    if source_root.count("ttbi.assert_reviewed_matlab_resolution( ...") != 2:
        raise ContractError("source root does not repeat MATLAB resolution checks")
    if source_root.count("ttbi.hash_reviewed_source_entries( ...") != 2:
        raise ContractError("source root does not take two complete byte snapshots")

    for token in (
        "'bundle_source_files.txt'",
        "if ~ttbi.regular_nonsymlink_file(manifest_path)",
        "before_sha = ttbi.stable_file_sha256(manifest_path);",
        "manifest_text = fileread(manifest_path);",
        "after_sha = ttbi.stable_file_sha256(manifest_path);",
        "ttbi.validate_repository_relative_path(entry);",
        "if ~ttbi.regular_nonsymlink_file(absolute_path)",
        "numel(unique(entries)) ~= numel(entries)",
        "numel(unique(lower_entries)) ~= numel(entries)",
        "if ~isequal(entries, sort(entries))",
    ):
        _once(reviewed_entries, token, f"reviewed source inventory {token}")
    for token in (
        "contains(entry, '\\')",
        "startsWith(entry, '/')",
        "contains(entry, '//')",
        "strcmp(components, '.')",
        "strcmp(components, '..')",
        "endsWith(component, '.')",
        "'^(COM|LPT)[1-9]$'",
    ):
        _once(path_validator, token, f"unsafe manifest-path guard {token}")
    for token in (
        "digest = ttbi.stable_file_sha256(absolute_path);",
        "sprintf('%s:%s', relative_name, digest);",
        "digest_lines = strjoin(sort(lines), newline);",
    ):
        _once(source_hasher, token, f"reviewed source hasher {token}")
    for token in (
        "before = ttbi.file_observation(path);",
        "digest = ttbi.file_sha256(path);",
        "after = ttbi.file_observation(path);",
        "if ~isequal(before, after)",
    ):
        _once(stable_hasher, token, f"stable source-file hasher {token}")
    for token in (
        "if ~ttbi.regular_nonsymlink_file(path)",
        "file_identity = ttbi.filesystem_identity(path);",
        "'basic:size,lastModifiedTime'",
        "confirmed_identity = ttbi.filesystem_identity(path);",
        "if ~strcmp(file_identity, confirmed_identity)",
        "'file_key', file_identity",
    ):
        _once(file_observer, token, f"source-file observation {token}")
    for token in (
        "options = ttbi.nofollow_link_options();",
        "attributes.get('fileKey')",
        "if ~isempty(file_key)",
        "key_text = strtrim(char(file_key.toString()));",
        "if ~isempty(key_text)",
        "identity = ['nio|' key_text];",
        "if ispc",
        "identity = ['windows|' ttbi.windows_file_identity(path)];",
        "if isunix",
        "'unix:dev'",
        "'unix:ino'",
    ):
        _once(filesystem_identity, token, f"filesystem identity {token}")
    for token in (
        "if islogical(raw) && isscalar(raw)",
        "if isnumeric(raw) && isscalar(raw) && isfinite(raw)",
        "(raw == 0 || raw == 1)",
        "javaMethod('booleanValue', raw)",
        "islogical(converted) && isscalar(converted)",
    ):
        _once(java_boolean, token, f"Java Boolean boundary {token}")
    for token in (
        "ttbi.java_boolean_value(symbolic)",
        "ttbi.java_boolean_value(other)",
    ):
        _once(path_alias, token, f"path-component Boolean boundary {token}")
    for token in (
        "system_directory = char(System.Environment.SystemDirectory);",
        "executable = fullfile(system_directory, 'fsutil.exe');",
        "store = javaMethod( ...",
        "volume_before_value = javaMethod( ...",
        "ttbi.run_small_process( ...",
        "{executable, 'file', 'queryfileid'",
        "(?i)(?<![0-9a-f])0x(?:[0-9a-f]{32}|[0-9a-f]{16})(?![0-9a-f])",
        "if exit_code ~= 0 || numel(matches) ~= 1",
        "all(file_id == '0') || all(file_id == 'f')",
        "store_after = javaMethod( ...",
        "volume_after_value = javaMethod( ...",
        "if ~strcmp(volume_before, volume_after)",
        "identity = ['volume-vsn=' volume_before '|file-id=' file_id];",
    ):
        _once(
            windows_file_identity,
            token,
            f"Windows native file identity {token}",
        )
    for token in (
        "javaObject('java.lang.ProcessBuilder', native_arguments)",
        "javaMethod('destroyForcibly', process)",
        "'waitFor', process, int64(timeout_seconds), seconds",
        "if ~finished",
        "javaMethod('exitValue', process)",
        "reader.readLine()",
    ):
        _once(process_runner, token, f"bounded native process {token}")
    for token in (
        "system_directory = char(System.Environment.SystemDirectory);",
        "ttbi.run_small_process( ...",
        "{executable, 'hardlink', 'list'",
        "if exit_code ~= 0 || isempty(lines)",
        "count = numel(lines);",
    ):
        _once(
            windows_hardlink_count,
            token,
            f"bounded Windows hard-link query {token}",
        )
    for token in (
        "ttbi.assert_results_not_on_matlab_path(matlab_root);",
        "listed = ttbi.list_matlab_executable_files(matlab_root);",
        "if ~ttbi.regular_nonsymlink_file(absolute_path)",
        "unexpected = actual(~ismember(actual_folded, manifest_folded));",
    ):
        _once(shadow_guard, token, f"MATLAB shadow-inventory guard {token}")
    for token in (
        "resolved = which(symbol);",
        "if isempty(resolved)",
        "resolved_nio = javaObject( ...",
        "expected_nio = javaObject( ...",
        "resolved_absolute = ttbi.comparison_path(char(resolved_nio.toString()));",
        "expected_absolute = ttbi.comparison_path(char(expected_nio.toString()));",
        "~ttbi.regular_nonsymlink_file(resolved)",
    ):
        _once(resolution_guard, token, f"MATLAB resolution guard {token}")
    for token in (
        "root_observation = ttbi.directory_observation(matlab_root);",
        "next_folder = 1;",
        "while next_folder <= numel(pending_paths)",
        "folder = pending_paths{next_folder};",
        "next_folder = next_folder + 1;",
        "visited_paths = {root_observation.canonical_path};",
        "visited_keys = {root_observation.file_key};",
        "observation = ttbi.directory_observation(absolute_path);",
        "any(strcmp(visited_paths, observation.canonical_path))",
        "any(strcmp(visited_keys, observation.file_key))",
        "startsWith(extension, '.mex', 'IgnoreCase', true)",
        "if ~isequal(ttbi.directory_observation(folder), folder_observation)",
    ):
        _once(executable_inventory, token, f"MATLAB disk inventory {token}")
    for token in (
        "absolute_nio = file.toPath().toAbsolutePath().normalize();",
        "if ttbi.path_component_is_link_alias(char(cursor.getPath()))",
        "canonical = ttbi.comparison_path(char(file.getCanonicalPath()));",
        "if ~strcmp(absolute, canonical)",
        "'basic:isDirectory'",
        "~ttbi.java_boolean_value(is_directory)",
        "file_identity = ttbi.filesystem_identity(path);",
        "confirmed_identity = ttbi.filesystem_identity(path);",
        "if ttbi.path_component_is_link_alias( ...",
        "if ~strcmp(file_identity, confirmed_identity) || ...",
        "'file_key', file_identity",
    ):
        _once(directory_observer, token, f"source-directory identity {token}")


def _semantic_state_uids(stage: str) -> tuple[list[str], list[str]]:
    """Independently reconstruct one reviewed four-block UID inventory."""

    if stage in ("F40-S", "F40-M"):
        bridge_length, spans, targets = 40.0, 2, (2,)
    elif stage in ("L99-S", "L99-M"):
        bridge_length, spans, targets = 99.6, 4, (2, 3, 4)
    else:  # pragma: no cover - fixed private fixture
        raise ContractError(f"unknown semantic UID fixture {stage!r}")
    geometry = "".join(f"{target:02d}" for target in targets)

    def uid(
        family: str, target: int, level: int, replica: int
    ) -> str:
        return (
            f"ttbi-state-v2|Lmm={round(1000 * bridge_length):06d}|"
            f"spans={spans}|scour={geometry}|family={family}|"
            f"target={target:02d}|level={level:04d}|rep={replica:03d}"
        )

    uids: list[str] = []
    families: list[str] = []
    if stage == "F40-S":
        for severity_percent in range(61):
            for replica in range(1, 6):
                if severity_percent == 0:
                    family, target, level = "target_healthy", 0, 0
                else:
                    family, target, level = (
                        "scour_only", targets[0], severity_percent
                    )
                uids.append(uid(family, target, level, replica))
                families.append(family)
        return uids, families

    for replica in range(1, 51):
        uids.append(uid("target_healthy", 0, 0, replica))
        families.append("target_healthy")
    for target in targets:
        for replica in range(1, 6):
            for level in (12, 24, 36, 48, 60):
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
    """Exercise four exact inventories and every named seed namespace."""

    schedule = "uid-named-substreams-v2"
    state_names = (
        "operations",
        "crack",
        "profile-state",
        "track",
        "profile-phase",
    )
    passage_names = ("profile-passage", "oor-passage")
    inventories: dict[str, list[str]] = {}
    for stage, expected in (
        ("F40-S", 305),
        ("F40-M", 425),
        ("L99-S", 475),
        ("L99-M", 475),
    ):
        uids, families = _semantic_state_uids(stage)
        inventories[stage] = uids
        if len(uids) != expected or len(set(uids)) != expected:
            raise ContractError(
                f"semantic UID fixture for {stage} is not {expected} unique states"
            )
        expected_joint = 0 if stage == "F40-S" else 250
        if families.count("joint") != expected_joint:
            raise ContractError(f"{stage} joint UID population drifted")
        roots = [
            _seed32(f"ttbi-state-seed-v1|damage_seed=1|{uid}")
            for uid in uids
        ]
        if stage == "F40-S":
            expected_first_state = (
                1471267274,
                2445595845,
                4221784991,
                2506555902,
                166414681,
            )
            expected_first_passage = (1508343681, 3254353867)
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
                roots[0] != 1955233256
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
                f"named seed fixture for {stage} has zero/collision"
            )
    if len(set(inventories["F40-S"]) & set(inventories["F40-M"])) != 30:
        raise ContractError("F40 semantic inventories do not match exactly 30 states")
    if inventories["L99-S"] != inventories["L99-M"]:
        raise ContractError("L99 semantic inventories are not completely paired")


def validate_contract(
    a00: str,
    dataset: str,
    driver: str,
    source_root: str = SOURCE_ROOT_SOURCE,
    environment: str = ENVIRONMENT_SOURCE,
    save_progress: str = SAVE_PROGRESS_SOURCE,
    make_micro: str = MAKE_MICRO_SOURCE,
    crn_smoke: str = CRN_SMOKE_SOURCE,
    provenance_smoke: str = PROVENANCE_SMOKE_SOURCE,
) -> None:
    """Raise ContractError unless all R11 generation invariants hold."""
    try:
        a00_script = a00[:a00.index(TTBI_MARK)]
    except ValueError as exc:
        raise ContractError("reviewed +ttbi source inventory is missing") from exc
    lock = _validate_environment(environment)
    _validate_helpers(a00, source_root)
    _validate_crn_numeric_fixture()
    campaign_setup = ttbi_block(a00, "campaign_setup")
    state_design_owner = ttbi_block(a00, "build_state_design")
    track_sampler = ttbi_block(a00, "sample_track_damage")
    dry_arm_resolver = ttbi_block(a00, "dry_ballast_stiffness_arm")

    matlab_schema = _one(
        r"^\s*gen_schema\s*=\s*'([^']+)';", a00, "MATLAB gen_schema"
    )
    matlab_channel_schema = _one(
        r"^\s*channel_schema_id\s*=\s*'([^']+)';",
        a00,
        "MATLAB channel_schema_id",
    )
    _once(
        dataset,
        "_EXPECTED_GEN_SCHEMA = EXPECTED_GEN_SCHEMA",
        "Python expected gen_schema derivation",
    )
    _once(
        dataset,
        "_EXPECTED_CHANNEL_SCHEMA_ID = EXPECTED_CHANNEL_SCHEMA_ID",
        "Python expected channel_schema_id derivation",
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
        a00_script,
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
    for token in (
        "'TTBI_DRY_BALLAST_STIFFNESS_ARM'",
        "assert(~qualification_run, ...\n"
        "        'A00:DryBallastSensitivityQualification'",
        "if ~campaign_config.use_track_eov",
        "'A00:DryBallastSensitivityDeferred'",
        "campaign_config.ballast_dry_stiffness_arm = ...\n"
        "        ttbi.dry_ballast_stiffness_arm(dry_ballast_sign_request_);",
    ):
        _once(a00_script, token, f"dry-ballast dispatch guard {token}")
    for token in (
        "arm = 'retained-stiffening';",
        "allowed = {'retained-stiffening', 'reciprocal-softening'};",
        "if ~any(strcmp(value, allowed))",
        "'ttbi:dry_ballast_stiffness_arm:UnsupportedArm'",
    ):
        _once(dry_arm_resolver, token, f"dry-ballast arm resolver {token}")
    for token in (
        "dry_stiffness_arm = ttbi.dry_ballast_stiffness_arm(config);",
        "config_has_arm = isfield(config, 'ballast_dry_stiffness_arm');",
        "state_has_arm = isfield(track_state, 'ballast_dry_stiffness_arm');",
        "'ttbi:sample_track_damage:SensitivityArmMismatch'",
        "if strcmp(dry_stiffness_arm, 'reciprocal-softening')",
        "if ~isfinite(eta_k_base) || eta_k_base <= 1",
        "eta_k = 1 / eta_k_base;",
        "eta_k = eta_k_base;",
        "track_state.ballast_dry_stiffness_arm = dry_stiffness_arm;",
    ):
        _once(track_sampler, token, f"dry-ballast sampler contract {token}")
    for token in (
        "sleeper_ratio = track_window / sleeper_spacing;",
        "floor(sleeper_ratio + 10 * eps(sleeper_ratio)) * sleeper_spacing;",
        "latest_group_start = last_sleeper_location - ...",
    ):
        _once(track_sampler, token, f"hanging-group lattice boundary {token}")
    crn_integer_literals = {
        "n_states_multi": 0,
        "n_anchor_levels": 60,
        "n_anchor_reps": 5,
        "n_healthy_states": 5,
        "n_nuisance_states": 0,
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
    dense_count = (
        crn_integer_literals["n_healthy_states"]
        + crn_integer_literals["n_anchor_levels"]
        * crn_integer_literals["n_anchor_reps"]
    )
    if dense_count != 305:
        raise ContractError(f"F40-S dense state count {dense_count} != 305")
    for token in (
        "case 'F40-S'",
        "case 'F40-M'",
        "case 'L99-S'",
        "case 'L99-M'",
        "expected_counts = [0 5 60 5 0];",
        "expected_counts = [250 50 5 5 50];",
        "use_track_eov = false;",
        "use_oor_eov   = false;",
    ):
        if token not in campaign_setup:
            raise ContractError(f"four-block MATLAB setup missing: {token}")
    for token in (
        "allowed = [required; {'qualification_run'}];",
        "qualification_run = false;",
        "if isfield(inputs, 'qualification_run')",
        "if ~islogical(qualification_run) || ~isscalar(qualification_run)",
        "if ~qualification_run && ~isequal(actual_counts, expected_counts)",
        "if inputs.Npass > 5 || any(actual_counts > [16 8 8 8 16])",
        "state_design_kind = 'qualification-five-family-v1';",
        "config.qualification_run = qualification_run;",
    ):
        if token not in campaign_setup:
            raise ContractError(
                f"qualification-only campaign setup contract missing: {token}"
            )
    stream_schedule = _one(
        r"^\s*random_stream_schedule_version\s*=\s*'([^']+)';",
        state_design_owner,
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
                campaign_setup,
                f"reviewed EOV literal {name}",
            )
        )
        if value != expected:
            raise ContractError(
                f"{name}={value} differs from reviewed value {expected}"
            )
    group_literal = _one(
        r"^\s*hang_group_size\s*=\s*(\[\s*[0-9]+\s+[0-9]+\s*\]);",
        campaign_setup,
        "reviewed hanging-sleeper group-size bounds",
    )
    group_bounds = tuple(int(value) for value in re.findall(r"[0-9]+", group_literal))
    if group_bounds != (1, 5):
        raise ContractError(
            f"hang_group_size={group_bounds!r} differs from reviewed [1, 5] bounds"
        )
    mean_group_size = sum(group_bounds) / 2
    expected_unsupported_share = (
        reviewed_eov_literals["hang_rate_100m"] * mean_group_size / 167
    )
    if not (0.0538 < expected_unsupported_share < 0.0540):
        raise ContractError("analytic unsupported-sleeper share moved from 5.4%")
    unsupported_share_boundary = (
        "author-chosen stress prior. Its arithmetic 3*3/167 = 5.4% share "
        "assumes a\n% mean group size of three and is not field prevalence."
    )
    if unsupported_share_boundary not in campaign_setup:
        raise ContractError(
            "conditional 5.4% unsupported-sleeper derivation/evidence boundary "
            "is absent from ttbi.campaign_setup"
        )
    for token, label in (
        (
            "group_count = poissrnd(config.hang_rate_100m * track_window / 100);",
            "window-scaled hanging-sleeper Poisson draw",
        ),
        (
            "group_size = randi(config.hang_group_size);",
            "reviewed hanging-sleeper group-size draw",
        ),
    ):
        _once(track_sampler, token, label)
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
    if matlab_channel_schema != EXPECTED_CHANNEL_SCHEMA:
        raise ContractError(
            "MATLAB channel schema differs from campaign_contract"
        )
    if behavior_version != EXPECTED_BEHAVIOR_VERSION:
        raise ContractError(
            "MATLAB behavior version differs from campaign_contract"
        )
    if study_tag != EXPECTED_STUDY_TAG or not study_tag.endswith("r12"):
        raise ContractError("study tag does not identify the R12 protocol")
    if qualification_literal != "false" or qualification_source != "PRODUCTION":
        raise ContractError("production A00 qualification literals are not immutable")
    _once(
        a00_script,
        "'qualification_run', qualification_run, ...",
        "A00 explicit campaign-setup qualification marker",
    )
    if "A00_RELEASE_QUALIFICATION" in a00:
        raise ContractError("environment-variable qualification bypass returned")
    if "validated_matlab_releases" in a00:
        raise ContractError("coarse validated-release allowlist returned")
    qualification_size_gate = (
        "if qualification_run && ...\n"
        "        (state_design.n_states > 64 || campaign_config.Npass > 5)"
    )
    if qualification_size_gate not in a00_script:
        raise ContractError("qualification mode is not structurally micro-only")
    qualification_start = a00.index("if qualification_run")
    qualification_end = a00.index(
        "elseif ~strcmp(actual_matlab_environment_sha256", qualification_start
    )
    qualification_block = a00[qualification_start:qualification_end]
    for token in (
        "qualification_script_path_ = mfilename('fullpath');",
        "ttbi.qualification_script_identity(qualification_script_path_, ...",
        "if ~strcmp(executed_qualification_source_sha256_, ...",
        "qualification_source_sha256)",
        "qualification_sha_placeholder_ = ...",
        "qualification_folder_placeholder_ = ...",
    ):
        if token not in qualification_block:
            raise ContractError(
                f"qualification executable self-authentication missing: {token}"
            )
    _once(
        a00_script,
        "ttbi.preserve_qualification_evidence( ...\n"
        "    run_folder, qualification, provenance, run_folder_observation);",
        "A00-to-qualification evidence owner binding",
    )
    qualification_evidence_block = ttbi_block(
        a00, "preserve_qualification_evidence"
    )
    for token in (
        "function preserve_qualification_evidence( ...\n"
        "        run_folder, qualification, provenance, run_folder_observation)",
        "if ~provenance.release_qualification_run",
        "ttbi.assert_generation_output_directory( ...\n"
        "    run_folder, run_folder_observation);",
        "evidence_path = fullfile(run_folder, 'qualification_executed.m');",
        "qualification.executed_file_sha256",
        "[copied, copy_message] = copyfile( ...\n"
        "        qualification.script_path, evidence_path, 'f');",
        "receipt = ttbi.qualification_host_receipt( ...",
        "ttbi.write_qualification_host_receipt( ...",
    ):
        if token not in qualification_evidence_block:
            raise ContractError(
                f"qualification executable evidence copy missing: {token}"
            )
    if qualification_evidence_block.count(
        "ttbi.file_sha256(evidence_path)"
    ) != 2:
        raise ContractError(
            "qualification evidence bytes are not checked before/after copy"
        )
    # The canonicalisation rule spans two package functions: the identity
    # function reads and hashes the bytes, and replace_unique_bytes enforces
    # that the substitution it performs matched exactly once.  Binding the
    # guard to exactly those two is tighter than the previous slice, which ran
    # from the function header to END OF FILE and therefore also swept in every
    # unrelated local that happened to follow it in A00.
    local_function_block = (
        ttbi_block(a00, "qualification_script_identity")
        + ttbi_block(a00, "replace_unique_bytes")
    )
    for token in (
        "fopen(fpath, 'rb')",
        "raw_sha = ttbi.sha256_bytes(bytes);",
        "canonical = ttbi.replace_unique_bytes(bytes, ...",
        "canonical_sha = ttbi.sha256_bytes(canonical);",
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

    # Strong common-random-number state design (generation-rules-v8). The
    # complete scientific catalogue now has one explicit package owner. Slice
    # it from the supplied in-memory source so mutations cannot be hidden by a
    # disk re-read and so unrelated A00 publication code cannot satisfy it.
    design_block = ttbi_block(a00, "build_state_design")
    for token in (
        "n_latent_bear = 2;",
        "state_identity_version = 'semantic-state-v2';",
        "joint_lhs_design = 'not-applicable-dense-scour';",
        "joint_lhs_design = 'master-scour-plus-two-bearing-v2';",
        "elseif strcmp(state_design_kind, 'qualification-five-family-v1')",
        "~isequal(config.qualification_run, true)",
        "'qualification-five-family-v1 requires the explicit true '",
        "joint_lhs_design = 'qualification-master-lhs-v1';",
        "random_stream_schedule_version = 'uid-named-substreams-v2';",
        "state_stream_names = {'operations','crack','profile-state','track','profile-phase'};",
        "passage_stream_names = {'profile-passage','oor-passage'};",
        "~isequal(reshape(Dano, 1, []), (0:60)/100)",
        "severity_pct_ = 0:60",
        "level_codes_s = round(100*levels_s);",
        "k_ref_bear  = 4 * Beam_probe.Prop.E * Beam_probe.Prop.I / (L_bridge / num_spans);",
        "fix2k       = @(phi) k_ref_bear .* phi ./ (1 - phi);",
        "n_nuis_here = n_nuisance_states;",
        "levels_b = linspace(bearing_fixity_max / n_anchor_levels, ...",
        "lhs = lhsdesign(n_states_multi, n_tgt + n_latent_bear);",
        "joint_s(:, scour_supports) = lhs(:, 1:n_tgt) * dano_max;",
        "joint_bf = lhs(:, n_tgt+1:n_tgt+n_latent_bear) * bearing_fixity_max;",
        "LatentBearingFixity = [anchors_bf; joint_bf];",
        "StateUID     = [uid_; joint_uid_];",
        "expected_states_ = n_healthy_states + ...",
        "StateSeedID = ttbi.state_seed_ids(StateUID, damage_seed);",
        "ttbi.named_stream_seed_ids(StateSeedID, StateUID, Npass, ...",
        "LatentCrackOn(strcmp(StateFamily, 'nuisance_only')) = true;",
        "CrackOn = logical(use_crack_eov) & LatentCrackOn;",
        "BearingFixity = LatentBearingFixity;",
        "BearingStates = fix2k(BearingFixity);",
        "design.state_design_kind = state_design_kind;",
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

    uid_block = ttbi_block(a00, "state_uid")
    for token in (
        "ttbi-state-v2|Lmm=%06d|spans=%d|scour=%s|",
        "family=%s|target=%02d|level=%04d|rep=%03d",
        "round(1000 * L_bridge), num_spans, sprintf('%02d', scour_supports), ...",
    ):
        if token not in uid_block:
            raise ContractError(f"semantic UID grammar missing: {token}")

    seed_block = ttbi_block(a00, "state_seed_ids")
    for token in (
        "'ttbi-state-seed-v1|damage_seed=%.0f|%s'",
        "damage_seed, state_uids{k}));",
        "ids(k) = uint32(hex2dec(h(1:8)));",
        "if any(ids == 0) || numel(unique(ids)) ~= numel(ids)",
    ):
        if token not in seed_block:
            raise ContractError(f"stable root StateSeedID guard missing: {token}")

    # Substream derivation = the key grammar (named_stream_seed_ids) plus the
    # 32-bit reduction it delegates to (seed32).  Same reasoning as the
    # canonicalisation block above: previously one slice happened to cover both
    # because seed32 followed in A00's local list.
    named_block = (
        ttbi_block(a00, "named_stream_seed_ids") + ttbi_block(a00, "seed32")
    )
    for token in (
        "schedule_version, state_seed_ids(i_), state_uids{i_}, ...",
        "passage_names{stream_}, pass_",
        "all_ids_ = [state_seed_ids(:); state_seeds(:); passage_seeds(:)];",
        "if any(all_ids_ == 0) || numel(unique(all_ids_)) ~= numel(all_ids_)",
        "seed = uint32(hex2dec(h(1:8)));",
    ):
        if token not in named_block:
            raise ContractError(f"named RNG substream guard missing: {token}")

    operations_block = ttbi_block(a00, "sample_state_operations")
    crack_block = ttbi_block(a00, "sample_crack_damage")
    profile_block = ttbi_block(a00, "build_profile_config")
    track_block = ttbi_block(a00, "sample_track_damage")
    oor_block = ttbi_block(a00, "sample_wheel_oor")
    generation_runner = ttbi_block(a00, "run_generation_states")
    state_executor = ttbi_block(a00, "execute_generation_state")
    completion_publisher = ttbi_block(a00, "publish_generation_completion")
    publication_revoker = ttbi_block(a00, "revoke_generation_publication")
    credential_deleter = ttbi_block(a00, "delete_file_entry_if_present")
    credential_names_owner = ttbi_block(
        a00, "generation_publication_credential_names"
    )
    marker_writer = ttbi_block(a00, "write_generation_marker_temp")
    worker_authenticator = ttbi_block(a00, "authenticate_generation_worker")
    worker_attestation_builder = ttbi_block(
        a00, "build_generator_source_attestation"
    )
    worker_attestation_validator = ttbi_block(
        a00, "require_generation_worker_attestation"
    )
    pool_deleter = ttbi_block(a00, "delete_generation_pool")
    output_directory_assertion = ttbi_block(
        a00, "assert_generation_output_directory"
    )
    output_directory_creator = ttbi_block(
        a00, "ensure_generation_output_directory"
    )
    execution_context_builder = ttbi_block(a00, "build_execution_context")
    case_info_builder = ttbi_block(a00, "build_case_info")
    sidecar_validator = ttbi_block(a00, "validate_generation_sidecars")
    generated_path_guard = ttbi_block(
        a00, "assert_results_not_on_matlab_path"
    )
    executable_inventory = ttbi_block(a00, "list_matlab_executable_files")
    output_boundary_smoke = ttbi_block(a00, "smoke_generation_output_boundary")
    generation_rng = (
        operations_block
        + crack_block
        + profile_block
        + track_block
        + oor_block
        + state_executor
    )
    for token in (
        "generated_root_names = {'Results', 'Results_sensitivity'};",
        "for root_index = 1:numel(generated_root_names)",
        "ttbi.path_is_same_or_child(entry, results_root)",
    ):
        _once(
            generated_path_guard,
            token,
            f"generated-result path exclusion {token}",
        )
    _once(
        executable_inventory,
        "{'Results', 'Results_sensitivity'}",
        "generated-result source-inventory exclusion",
    )
    for token in (
        "if isfield(campaign, 'ballast_dry_stiffness_arm')",
        "context.track.ballast_dry_stiffness_arm = ...\n"
        "        ttbi.dry_ballast_stiffness_arm(campaign);",
        "'channel_schema_id', derived_identity.channel_schema_id, ...",
        "'state_design_kind', derived_identity.state_design_kind, ...",
        "if ~isequal(context.identity, derived_identity)",
        "'ttbi:ExecutionIdentityProjection'",
    ):
        _once(
            execution_context_builder,
            token,
            f"dry-ballast execution-context binding {token}",
        )
    for token in (
        "if isfield(campaign, 'ballast_dry_stiffness_arm')",
        "case_info.ballast_dry_stiffness_arm = ...\n"
        "        ttbi.dry_ballast_stiffness_arm(campaign);",
        "case_info.analysis_scope = 'dry-ballast-stiffness-sign-sensitivity';",
    ):
        _once(
            case_info_builder,
            token,
            f"dry-ballast case-info binding {token}",
        )
    for token in (
        "rng(double(operations_seed_id), 'twister');",
        "rng(double(crack_seed_id), 'twister');",
        "rng(double(passage_profile_seed_id), 'twister');",
        "rng(double(state_profile_seed_id), 'twister');",
        "profile_config.phase_seed = double(state_phase_seed_id);",
        "rng(double(track_seed_id), 'twister');",
        "rng(double(passage_seed_id), 'twister');",
        "state.StateNamedStreamSeedID(state_index, 1)",
        "state.StateNamedStreamSeedID(state_index, 2)",
        "state.StateNamedStreamSeedID(state_index, 3)",
        "state.StateNamedStreamSeedID(state_index, 4)",
        "state.StateNamedStreamSeedID(state_index, 5)",
        "state.PassageNamedStreamSeedID(state_index, passage_index, 1)",
        "state.PassageNamedStreamSeedID(state_index, passage_index, 2)",
        "lhs_matrix = lhsdesign(Npass, 2)';",
        "if ~isequal(size(lhs_matrix), [2, Npass])",
        "observed_strata = sort(floor(lhs_matrix * Npass), 2);",
        "expected_strata = repmat(0:Npass-1, 2, 1);",
        "if ~isequal(observed_strata, expected_strata)",
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
        "assert(f40s.n_states == 305);",
        "assert(f40m.n_states == 425);",
        "assert(l99s.n_states == 475 && l99m.n_states == 475);",
        "assert(numel(shared_uid) == 30);",
        "[12 24 36 48 60]",
        "assert(isequal(l99s.StateUID,l99m.StateUID));",
        "assert(isequal(l99s.StateSeedID,l99m.StateSeedID));",
        "assert(isequal(l99s.DamageStates,l99m.DamageStates));",
        "assert(isequal(l99s.LatentBearingFixity,l99m.LatentBearingFixity));",
        "assert(isequal(l99s.LatentCrackOn,l99m.LatentCrackOn));",
        "uint32(1955233256)",
        "1471267274 2445595845 4221784991 2506555902 166414681",
        "1508343681 3254353867",
        "local_namespace_draws(state_seed_row,pass_seed_row,1000)",
        "assert(isequal(base.track,mutated.track));",
        "assert(isequal(base.oor,mutated.oor));",
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
        "(16 mutations rejected; temp cleaned)",
    ):
        if token not in provenance_smoke:
            raise ContractError(
                f"R11 state-serialization smoke missing CRN guard: {token}"
            )

    for token in (
        '"F40-S"',
        '"F40-M"',
        '"L99-S"',
        '"L99-M"',
        'stage: str = "F40-S"',
        '"qualification_run = true;  % RELEASE-QUALIFICATION ONLY"',
        '"n_states_multi   = 10;     % MICRO-SMOKE"',
        '"n_healthy_states  = 3;     % MICRO-SMOKE"',
        '"n_anchor_levels  = 2;      % MICRO-SMOKE"',
        '"n_anchor_reps     = 2;     % MICRO-SMOKE"',
        '"n_nuisance_states = 6;     % MICRO-SMOKE"',
        '"no-argument micro generation is retired; use --qualification "',
        '"--dryrun is retired; use --qualification --stage with one of "',
        '"the legacy toy dry-run is retired; use four-stage release "',
    ):
        if token not in make_micro:
            raise ContractError(
                f"four-block qualification generator missing: {token}"
            )
    if make_micro.count('stage: str = "F40-S"') != 3:
        raise ContractError(
            "qualification generator must expose exactly three F40-S defaults"
        )
    if make_micro.count("if not qualification:") != 3:
        raise ContractError(
            "every unmarked micro rendering/writing entry must fail closed"
        )
    for retired_stage in (
        "s0_scour", "s11_bear", "s12_crack", "s13_bearcrack",
        "s14_prof", "s15_track", "s16_all", "s21_scour4",
        "s22_bearcrack4", "s23_all4",
    ):
        if retired_stage in make_micro:
            raise ContractError(
                f"qualification generator still exposes retired stage {retired_stage}"
            )

    for token in (
        "current_working_dir_ = ttbi.canonical_execution_path(pwd);",
        "expected_working_dir_ = ttbi.canonical_execution_path(script_dir_);",
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
    ):
        if token not in a00:
            raise ContractError(f"MATLAB environment/source gate missing: {token}")

    host_receipt_block = ttbi_block(a00, "qualification_host_receipt")
    receipt_writer_block = ttbi_block(a00, "write_qualification_host_receipt")
    for token in (
        "declared_host_id_ = strtrim(getenv('TTBI_QUALIFICATION_HOST_ID'));",
        "if isempty(regexp(declared_host_id_, ...",
        "'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'",
        "'schema=ttbi-matlab-qualification-host-v1'",
        "'host_diagnostic_sha256', ttbi.sha256(descriptor_)",
    ):
        if token not in host_receipt_block:
            raise ContractError(f"qualification host identity gate missing: {token}")
    for token in (
        "if ~isequal(observed_bytes_, expected_bytes_)",
        "error('A00:QualificationHostReceiptCollision', ...",
        "[moved_, move_message_] = movefile(temp_path_, path, 'f');",
        "ttbi.file_sha256(path)",
        "ttbi.sha256_bytes(expected_bytes_)",
    ):
        if token not in receipt_writer_block:
            raise ContractError(f"qualification receipt writer gate missing: {token}")

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

    identity_builder = ttbi_block(a00, "build_generation_identity")
    for token in (
        "if isfield(campaign, 'ballast_dry_stiffness_arm')",
        "config.ballast_dry_stiffness_arm = ...\n"
        "        ttbi.dry_ballast_stiffness_arm(campaign);",
    ):
        _once(
            identity_builder,
            token,
            f"dry-ballast fingerprint binding {token}",
        )
    definition_at = identity_builder.index(f"{BEHAVIOR_KEY} =")
    fp_start = identity_builder.index("config = struct(")
    fp_end = identity_builder.index(
        "% The realized state matrices", fp_start
    )
    fp_block = identity_builder[fp_start:fp_end]
    fp_binding = f"'{BEHAVIOR_KEY}', {BEHAVIOR_KEY}, ..."
    if definition_at > fp_start:
        raise ContractError("behavior version is defined after fingerprint config")
    for binding in (
        fp_binding,
        "'schema', gen_schema, ...",
        "'channel_schema_id', channel_schema_id, ...",
        "'campaign_matlab_release', provenance.campaign_matlab_release, ...",
        "'campaign_matlab_environment_sha256', ...",
        "provenance.campaign_matlab_environment_sha256, ...",
        "'generator_source_root_sha256', ...\n"
        "        provenance.generator_source_root_sha256, ...",
        "'qualification_source_sha256', ...\n"
        "        provenance.qualification_source_sha256, ...",
        "'max_parfor_workers', campaign.max_parfor_workers, ...",
        "'state_design_kind', state.state_design_kind, ...",
        "'state_identity_version', state.state_identity_version, ...",
        "'joint_lhs_design', state.joint_lhs_design, ...",
        "'n_latent_bearing_dims', state.n_latent_bear, ...",
        "'random_stream_schedule_version', ...\n"
        "        state.random_stream_schedule_version, ...",
        "'state_stream_names', {state.state_stream_names}, ...",
        "'passage_stream_names', {state.passage_stream_names}, ...",
    ):
        if fp_block.count(binding) != 1:
            raise ContractError(
                f"fingerprint binding missing/ambiguous: {binding}"
            )
    for forbidden in (
        "actual_matlab_environment_sha256",
        "actual_matlab_environment_descriptor",
        "provenance.matlab_release",
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

    fp_assignment_end = identity_builder.index(
        "generation_config_json = jsonencode(config);", fp_end
    )
    fp_assignment_block = identity_builder[fp_end:fp_assignment_end]
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
            f"config.{field}",
            f"fingerprinted CRN field {field}",
        )

    _once(
        a00_script,
        "identity = ttbi.build_generation_identity( ...\n"
        "    campaign_config, state_design, provenance);",
        "A00-to-generation identity binding",
    )
    _once(
        a00_script,
        "case_info = ttbi.build_case_info( ...\n"
        "    campaign_config, state_design, descriptor, identity, provenance, ...",
        "A00-to-case manifest builder binding",
    )
    _once(
        a00_script,
        "ttbi.write_run_manifest( ...\n"
        "    run_folder, case_info, campaign_config, state_design, ...\n"
        "    run_folder_observation);",
        "A00-to-run manifest writer binding",
    )
    manifest_block = ttbi_block(a00, "build_case_info")
    for binding in (
        "'generation_behavior_version', ...\n"
        "        identity.generation_behavior_version, ...",
        "'channel_schema_id', identity.channel_schema_id, ...",
        "'state_design_kind', identity.state_design_kind, ...",
        "'matlab_release', provenance.matlab_release, ...",
        "'campaign_matlab_release', provenance.campaign_matlab_release, ...",
        "'actual_matlab_environment_descriptor', ...",
        "provenance.actual_matlab_environment_descriptor, ...",
        "'actual_matlab_environment_sha256', ...",
        "provenance.actual_matlab_environment_sha256, ...",
        "'campaign_matlab_environment_descriptor', ...",
        "provenance.campaign_matlab_environment_descriptor, ...",
        "'campaign_matlab_environment_sha256', ...",
        "provenance.campaign_matlab_environment_sha256, ...",
        "'generator_source_root_sha256', ...\n"
        "        provenance.generator_source_root_sha256, ...",
        "'generator_source_digest_lines', ...\n"
        "        provenance.generator_source_digest_lines, ...",
        "'generator_source_file_count', ...\n"
        "        provenance.generator_source_file_count, ...",
        "'qualification_source_sha256', ...\n"
        "        provenance.qualification_source_sha256, ...",
        "'release_qualification_run', ...\n"
        "        provenance.release_qualification_run, ...",
        "'max_parfor_workers', campaign.max_parfor_workers, ...",
        "'state_identity_version', state.state_identity_version, ...",
        "'joint_lhs_design', state.joint_lhs_design, ...",
        "'n_latent_bearing_dims', state.n_latent_bear, ...",
        "'random_stream_schedule_version', ...\n"
        "        state.random_stream_schedule_version, ...",
        "'state_stream_names', strjoin(state.state_stream_names, ','), ...",
        "'passage_stream_names', strjoin(state.passage_stream_names, ','), ...",
    ):
        if manifest_block.count(binding) != 1:
            raise ContractError(f"case_info binding missing/ambiguous: {binding}")

    for token in (
        "'generation_behavior_version', 'channel_schema_id', ...",
        "'state_design_kind', ...",
        "context.identity.channel_schema_id",
        "case_info.state_design_kind, ...\n"
        "        context.identity.state_design_kind",
    ):
        if token not in sidecar_validator:
            raise ContractError(f"case-info schema/design guard missing: {token}")

    manifest_writer = ttbi_block(a00, "write_run_manifest")
    _once(
        manifest_writer,
        "function write_run_manifest( ...\n"
        "        run_folder, case_info, campaign, state, run_folder_observation)",
        "run-manifest writer public signature",
    )
    if manifest_writer.count(
        "ttbi.assert_generation_output_directory( ...\n"
        "    run_folder, run_folder_observation);"
    ) < 7:
        raise ContractError(
            "run-manifest writes/renames are not fenced by output identity"
        )
    damage_save_start = manifest_writer.index("save(damage_temp, ...")
    damage_save_end = manifest_writer.index(");", damage_save_start) + 2
    damage_save_block = manifest_writer[damage_save_start:damage_save_end]
    for field in (
        "DamageStates",
        "BearingStates",
        "BearingFixity",
        "LatentBearingFixity",
        "k_ref_bear",
        "scour_supports",
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

    _once(
        a00_script,
        "ttbi.assert_resume_case_identity( ...\n"
        "        run_folder, previous_manifest.case_info, identity, provenance);",
        "A00-to-resume identity binding",
    )
    resume_block = ttbi_block(a00, "assert_resume_case_identity")
    for token in (
        "~strcmp(previous.gen_schema, identity.gen_schema)",
        "~strcmp(previous.gen_fingerprint, identity.gen_fingerprint)",
        "strcmp(previous.matlab_release, provenance.matlab_release)",
        "strcmp(previous.campaign_matlab_release, ...\n"
        "        provenance.campaign_matlab_release)",
        "previous.release_qualification_run == ...\n"
        "        provenance.release_qualification_run",
        "strcmp(previous.actual_matlab_environment_descriptor, ...\n"
        "        provenance.actual_matlab_environment_descriptor)",
        "strcmp(previous.actual_matlab_environment_sha256, ...\n"
        "        provenance.actual_matlab_environment_sha256)",
        "strcmp(previous.campaign_matlab_environment_descriptor, ...\n"
        "        provenance.campaign_matlab_environment_descriptor)",
        "strcmp(previous.campaign_matlab_environment_sha256, ...\n"
        "        provenance.campaign_matlab_environment_sha256)",
        "strcmp(previous.generator_source_root_sha256, ...\n"
        "        provenance.generator_source_root_sha256)",
        "strcmp(previous.generator_source_digest_lines, ...\n"
        "        provenance.generator_source_digest_lines)",
        "previous.generator_source_file_count == ...\n"
        "        provenance.generator_source_file_count",
        "strcmp(previous.qualification_source_sha256, ...\n"
        "        provenance.qualification_source_sha256)",
    ):
        if token not in resume_block:
            raise ContractError(f"case_info resume invariant missing: {token}")

    _once(
        a00_script,
        "completed = ttbi.validate_resume_states( ...\n"
        "    run_folder, execution_context);",
        "A00-to-state resume validator binding",
    )
    state_resume = ttbi_block(a00, "validate_resume_states")
    state_file_inventory = ttbi_block(a00, "state_file_fields")
    state_payload_inventory = ttbi_block(a00, "state_payload_fields")
    observed_top_fields = frozenset(re.findall(
        r"^\s*'([^']+)'", state_file_inventory, flags=re.MULTILINE
    ))
    observed_data_fields = frozenset(re.findall(
        r"^\s*'([^']+)'", state_payload_inventory, flags=re.MULTILINE
    ))
    if observed_top_fields != STATE_TOP_LEVEL_FIELDS:
        raise ContractError("MATLAB/Python state top-level inventories diverge")
    if observed_data_fields != STATE_DATA_FIELDS:
        raise ContractError("MATLAB/Python state payload inventories diverge")
    for token in (
        "expected_variables = sort(ttbi.state_file_fields());",
        "variable_info = whos('-file', state_path);",
        "if ~isequal(variable_names, expected_variables)",
        "stamps = load(state_path, stamp_fields{:});",
        "isequal(stamps.file_gen_schema, provenance.gen_schema)",
        "isequal(stamps.file_gen_fingerprint, ...\n"
        "            provenance.gen_fingerprint)",
        "isequal(stamps.file_state_uid, state.StateUID{state_index})",
        "isequal(stamps.file_state_seed_id, ...\n"
        "            state.StateSeedID(state_index))",
        "isequal(stamps.file_random_stream_schedule_version, ...\n"
        "            state.random_stream_schedule_version)",
        "isequal(stamps.file_matlab_release, provenance.matlab_release)",
        "isequal(stamps.file_campaign_matlab_release, ...\n"
        "            provenance.campaign_matlab_release)",
        "isequal(stamps.file_release_qualification_run, ...\n"
        "            provenance.release_qualification_run)",
        "isequal(stamps.file_actual_matlab_environment_sha256, ...\n"
        "            provenance.actual_matlab_environment_sha256)",
        "isequal(stamps.file_campaign_matlab_environment_sha256, ...\n"
        "            provenance.campaign_matlab_environment_sha256)",
        "isequal(stamps.file_generator_source_root_sha256, ...\n"
        "            provenance.generator_source_root_sha256)",
        "isequal(stamps.file_qualification_source_sha256, ...\n"
        "            provenance.qualification_source_sha256)",
        "ttbi.validate_resumed_state_payload( ...",
    ):
        if token not in state_resume:
            raise ContractError(f"per-state resume invariant missing: {token}")

    payload_validator = ttbi_block(a00, "validate_resumed_state_payload")
    for token in (
        "expected_fields = sort(ttbi.state_payload_fields());",
        "if ~isequal(observed_fields, expected_fields)",
        "ttbi.validate_state_provenance(data, state_index, file_name, context);",
        "ttbi.validate_state_identity(data, state_index, file_name, context);",
        "ttbi.validate_state_metadata(data, state_index, file_name, context);",
        "ttbi.validate_state_raw_metadata(data, state_index, file_name, context);",
        "ttbi.validate_state_signals(data, state_index, file_name, context.Npass);",
        "ttbi.validate_state_contact(data, state_index, file_name, context);",
    ):
        if token not in payload_validator:
            raise ContractError(f"payload-validator chain missing: {token}")

    state_identity_validator = ttbi_block(a00, "validate_state_identity")
    state_provenance_validator = ttbi_block(a00, "validate_state_provenance")
    for token in (
        "state.PassageNamedStreamSeedID(state_index, :, :)",
        "isequal(data.state_uid, state.StateUID{state_index})",
        "isequal(data.state_seed_id, state.StateSeedID(state_index))",
        "isequal(data.state_named_stream_seed_id, ...",
        "isequal(data.passage_named_stream_seed_id, expected_passage_seeds)",
        "isequal(data.latent_bearing_fixity, ...",
        "isequal(data.latent_crack_on, state.LatentCrackOn(state_index))",
        "isequal(data.crack_on, state.CrackOn(state_index))",
    ):
        if token not in state_identity_validator:
            raise ContractError(f"per-state CRN identity guard missing: {token}")
    for token in (
        "isequal(data.channel_schema_id, context.identity.channel_schema_id)",
        "isequal(data.actual_matlab_environment_descriptor, ...",
        "isequal(data.actual_matlab_environment_sha256, ...",
        "isequal(data.campaign_matlab_environment_descriptor, ...",
        "isequal(data.campaign_matlab_environment_sha256, ...",
        "isequal(data.generator_source_root_sha256, ...",
        "isequal(data.generator_source_digest_lines, ...",
        "isequal(data.generator_source_file_count, ...",
        "isequal(data.qualification_source_sha256, ...",
        "ttbi.sha256_bytes(unicode2native( ...",
    ):
        if token not in state_provenance_validator:
            raise ContractError(f"per-state provenance guard missing: {token}")

    state_signal_validator = ttbi_block(a00, "validate_state_signals")
    for token in (
        "'AcelWheelsetPrimVag'",
        "expected_rows = [3, 4, 4, 3];",
    ):
        if token not in state_signal_validator:
            raise ContractError(f"physical8_v1 signal guard missing: {token}")

    _once(
        state_executor,
        "data2save.channel_schema_id = context.identity.channel_schema_id;",
        "per-state physical8_v1 schema assignment",
    )

    state_write_start = state_executor.index("data2save.gen_schema")
    state_end = state_executor.index(
        "save_progress(data2save", state_write_start
    )
    state_block = state_executor[state_write_start:state_end]
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
    full_state_write_start = state_executor.index("data2save.state_family")
    full_state_block = state_executor[full_state_write_start:state_end]
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
    call_end = state_executor.index(");", state_end) + 2
    call_block = state_executor[state_end:call_end]
    for token in (
        "provenance.matlab_release",
        "provenance.campaign_matlab_release",
        "provenance.release_qualification_run",
        "provenance.actual_matlab_environment_sha256",
        "provenance.campaign_matlab_environment_sha256",
        "provenance.generator_source_root_sha256",
        "provenance.qualification_source_sha256",
        "context.run_folder_observation",
    ):
        _once(call_block, token, f"save_progress call argument {token}")

    for token in (
        "narginchk(12, 13);",
        "if nargin < 13",
        "run_folder_observation = ttbi.directory_observation(run_path);",
        "ttbi.assert_generation_output_directory( ...\n"
        "        run_path, run_folder_observation);",
        "required_payload = {",
        "'gen_schema', 'gen_fingerprint', 'channel_schema_id', ...",
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
    for token in (
        "data.channel_schema_id, 'data.channel_schema_id'",
        "strcmp(channel_schema_id, 'physical8_v1')",
    ):
        if token not in save_progress:
            raise ContractError(f"save_progress channel-schema guard missing: {token}")
    if save_progress.count(
        "ttbi.assert_generation_output_directory( ...\n"
        "        run_path, run_folder_observation);"
    ) != 4:
        raise ContractError(
            "save_progress does not reassert output identity around state publication"
        )
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

    # Authenticate the generation output directory before every filesystem
    # boundary. Canonicalising first would erase evidence of a junction/symlink.
    for token in (
        "function assert_generation_output_directory(path, expected)",
        "required = sort({'canonical_path'; 'file_key'});",
        "current = ttbi.directory_observation(path);",
        "if ~isequal(current, expected)",
        "error('ttbi:GenerationOutputChanged', ...",
    ):
        _once(
            output_directory_assertion,
            token,
            f"generation output-directory boundary {token}",
        )
    for token in (
        "function observation = ensure_generation_output_directory( ...",
        "javaObject('java.io.File', run_folder).isAbsolute()",
        "any(ismember(parts, {'.', '..'}))",
        "ttbi.assert_generation_output_directory( ...\n"
        "    results_root, results_root_observation);",
        "if ttbi.path_entry_exists(child)",
        "child_observation = ttbi.directory_observation(child);",
        "[made, message] = mkdir(child);",
        "ttbi.assert_generation_output_directory(run_folder, observation);",
    ):
        if token not in output_directory_creator:
            raise ContractError(
                f"safe generation output creation missing: {token}"
            )
    if not (
        output_directory_creator.index(
            "ttbi.assert_generation_output_directory(cursor, cursor_observation);"
        )
        < output_directory_creator.index("[made, message] = mkdir(child);")
        < output_directory_creator.index(
            "child_observation = ttbi.directory_observation(child);",
            output_directory_creator.index("[made, message] = mkdir(child);")
        )
    ):
        raise ContractError(
            "output child creation is not parent-fenced then authenticated"
        )
    for token in (
        "context.run_folder = runtime.run_folder;",
        "context.run_folder_observation = runtime.run_folder_observation;",
        "ttbi.assert_generation_output_directory( ...\n"
        "    context.run_folder, context.run_folder_observation);",
    ):
        _once(
            execution_context_builder,
            token,
            f"execution-context output binding {token}",
        )
    for token in (
        "relative_run_folder = case_name;",
        "results_root = 'Results_sensitivity';",
        "relative_run_folder = fullfile('dry_ballast_stiffness_sign', ...\n"
        "        campaign_config.ballast_dry_stiffness_arm, case_name);",
        "run_folder = fullfile(results_root, relative_run_folder);",
        "results_root_observation = ttbi.directory_observation(results_root);",
        "run_folder_observation = ttbi.ensure_generation_output_directory( ...\n"
        "    run_folder, results_root, results_root_observation);",
        "ttbi.assert_generation_output_directory( ...\n"
        "    run_folder, run_folder_observation);",
        "if ~ttbi.regular_nonsymlink_file(case_info_path)",
        "runtime_context.run_folder = run_folder_observation.canonical_path;",
        "runtime_context.run_folder_observation = run_folder_observation;",
        "ttbi.revoke_generation_publication( ...\n"
        "    run_folder, run_folder_observation);",
    ):
        if token not in a00_script:
            raise ContractError(f"A00 output-directory boundary missing: {token}")
    output_capture_at = a00_script.index(
        "run_folder_observation = ttbi.ensure_generation_output_directory( ..."
    )
    case_path_at = a00_script.index(
        "case_info_path = fullfile(run_folder, 'case_info.mat');"
    )
    first_output_assert_at = a00_script.index(
        "ttbi.assert_generation_output_directory( ...",
        case_path_at,
    )
    resume_load_at = a00_script.index(
        "previous_manifest = load(case_info_path, 'case_info');"
    )
    revoke_output_at = a00_script.index(
        "ttbi.revoke_generation_publication( ..."
    )
    preserve_output_at = a00_script.index(
        "ttbi.preserve_qualification_evidence( ..."
    )
    manifest_output_at = a00_script.index("ttbi.write_run_manifest( ...")
    if not (
        a00_script.index("results_root = 'Results';")
        < a00_script.index(
            "results_root_observation = ttbi.directory_observation(results_root);"
        )
        < a00_script.index(
            "run_folder = fullfile(results_root, relative_run_folder);"
        )
        < output_capture_at < case_path_at < first_output_assert_at
        < resume_load_at < revoke_output_at < preserve_output_at
        < manifest_output_at
    ):
        raise ContractError(
            "A00 reads or writes the run folder outside its pinned identity"
        )
    if "delete(marker_path);" in a00_script:
        raise ContractError("A00 bypasses authenticated credential revocation")

    # Authenticate the modular execution chain. A00 owns orchestration only;
    # the bounded pool owns scheduling; the state executor owns one state.
    run_call = (
        "ttbi.run_generation_states( ...\n"
        "    execution_context, completed, "
        "campaign_config.max_parfor_workers);"
    )
    publish_call = (
        "ttbi.publish_generation_completion(run_folder, execution_context);"
    )
    _once(a00_script, run_call, "A00-to-generation runner binding")
    _once(a00_script, publish_call, "A00-to-completion publisher binding")
    if "ttbi.execute_generation_state(" in a00_script:
        raise ContractError("A00 bypasses the bounded generation runner")
    if a00_script.index(run_call) >= a00_script.index(publish_call):
        raise ContractError("A00 publishes completion before state generation")

    client_source_fences = [
        match.start()
        for match in re.finditer(
            re.escape("ttbi.assert_generator_source_unchanged(provenance);"),
            a00_script,
        )
    ]
    if len(client_source_fences) != 2 or not (
        client_source_fences[0]
        < a00_script.index(run_call)
        < client_source_fences[1]
        < a00_script.index(publish_call)
    ):
        raise ContractError(
            "A00 must fence the live source immediately around the runner"
        )

    _once(
        generation_runner,
        "function run_generation_states(context, completed, max_workers)",
        "generation runner public signature",
    )
    _once(
        generation_runner,
        "ttbi.execute_generation_state( ...\n"
        "        state_index, context, worker_attestation);",
        "generation runner-to-state executor binding",
    )
    _once(
        state_executor,
        "function execute_generation_state(state_index, context, worker_attestation)",
        "state executor public signature",
    )

    # Preserve the fail-closed, fresh-process worker boundary in its owner.
    skip_delimiter = "end\n\nexisting_pool = gcp('nocreate');"
    _once(
        generation_runner,
        "if ~isempty(getCurrentTask())",
        "generation runner client-only guard",
    )
    _once(generation_runner, "if all(completed)", "all-complete resume guard")
    _once(generation_runner, skip_delimiter, "all-complete pool bypass")
    all_complete_branch = generation_runner[
        generation_runner.index("if all(completed)"):
        generation_runner.index(skip_delimiter)
    ]
    if not re.search(r"(?m)^\s*return\s*;?\s*$", all_complete_branch):
        raise ContractError("all-complete resume branch does not return")
    if any(token in all_complete_branch for token in (
        "gcp(", "parcluster(", "parpool(", "parallel.pool.Constant", "parfor "
    )):
        raise ContractError("all-complete resume branch touches parallel pool")
    for token in (
        "existing_pool = gcp('nocreate');",
        "delete(existing_pool);",
        "if ~isempty(gcp('nocreate'))",
        "cluster = parcluster('Processes');",
        "pool_workers = min(max_workers, cluster.NumWorkers);",
        "pool = parpool(cluster, pool_workers);",
        "pool_cleanup = onCleanup(@() ttbi.delete_generation_pool(pool));",
        "if ~isa(pool, 'parallel.ProcessPool') || pool.NumWorkers ~= pool_workers",
        "worker_source = parallel.pool.Constant( ...\n"
        "    @() ttbi.authenticate_generation_worker(provenance));",
        "attestation_cleanup = onCleanup(@() delete(worker_source));",
        "parfor (state_index = 1:n_states, pool_workers)",
        "worker_attestation = worker_source.Value;",
        "ttbi.require_generation_worker_attestation( ...\n"
        "        worker_attestation, provenance);",
        "ttbi.execute_generation_state( ...\n"
        "        state_index, context, worker_attestation);",
        "clear attestation_cleanup",
        "clear pool_cleanup",
        "error('ttbi:GenerationPoolLeak', ...",
    ):
        if token not in generation_runner:
            raise ContractError(f"fresh process-pool invariant missing: {token}")
    if generation_runner.count("~isa(pool, 'parallel.ProcessPool')") != 1:
        raise ContractError("fresh generation pool type is not checked exactly once")
    if generation_runner.count("parcluster(") != 1:
        raise ContractError("generation cluster selection is ambiguous")
    existing_at = generation_runner.index("existing_pool = gcp('nocreate');")
    delete_existing_at = generation_runner.index("delete(existing_pool);")
    cluster_at = generation_runner.index("cluster = parcluster('Processes');")
    pool_at = generation_runner.index("pool = parpool(cluster, pool_workers);")
    constant_at = generation_runner.index(
        "worker_source = parallel.pool.Constant( ..."
    )
    loop_at = generation_runner.index(
        "parfor (state_index = 1:n_states, pool_workers)"
    )
    value_at = generation_runner.index("worker_attestation = worker_source.Value;")
    require_at = generation_runner.index(
        "ttbi.require_generation_worker_attestation( ..."
    )
    completed_at = generation_runner.index("if completed(state_index)")
    execute_at = generation_runner.index("ttbi.execute_generation_state( ...")
    clear_attestation_at = generation_runner.index("clear attestation_cleanup")
    clear_pool_at = generation_runner.index("clear pool_cleanup")
    leak_check_at = generation_runner.rindex("if ~isempty(gcp('nocreate'))")
    if not (
        existing_at < delete_existing_at < cluster_at < pool_at < constant_at
        < loop_at < value_at < require_at < completed_at < execute_at
        < clear_attestation_at < clear_pool_at < leak_check_at
    ):
        raise ContractError(
            "worker authentication, execution, or teardown order is wrong"
        )

    for token in (
        "task = getCurrentTask();",
        "if isempty(task)",
        "attestation = ttbi.build_generator_source_attestation(provenance);",
        "attestation.worker_context_authenticated = true;",
    ):
        _once(worker_authenticator, token, f"worker authenticator {token}")
    for token in (
        "[source_root, source_lines, source_count] = generator_source_root();",
        "provenance.generator_source_root_sha256",
        "provenance.generator_source_digest_lines",
        "provenance.generator_source_file_count",
        "'generator_source_root_sha256', source_root",
        "'generator_source_digest_lines', source_lines",
        "'generator_source_file_count', source_count",
    ):
        if token not in worker_attestation_builder:
            raise ContractError(f"worker source attestation missing: {token}")
    for token in (
        "if isempty(getCurrentTask())",
        "'generator_source_root_sha256'; ...",
        "'generator_source_digest_lines'; ...",
        "'generator_source_file_count'; ...",
        "line_count = sum(attestation.generator_source_digest_lines == newline) + 1;",
        "strcmp(attestation.gen_schema, provenance.gen_schema)",
        "strcmp(attestation.gen_fingerprint, provenance.gen_fingerprint)",
        "strcmp(attestation.generator_source_root_sha256, ...\n"
        "        provenance.generator_source_root_sha256)",
        "strcmp(attestation.generator_source_digest_lines, ...\n"
        "        provenance.generator_source_digest_lines)",
        "isequal(attestation.generator_source_file_count, ...\n"
        "        provenance.generator_source_file_count)",
        "strcmp(ttbi.sha256(attestation.generator_source_digest_lines), ...",
        "line_count == attestation.generator_source_file_count;",
    ):
        _once(
            worker_attestation_validator,
            token,
            f"worker attestation validation {token}",
        )
    executor_first_statement = next(
        (
            line.strip()
            for line in state_executor.splitlines()[1:]
            if line.strip() and not line.lstrip().startswith("%")
        ),
        "",
    )
    if executor_first_statement != (
        "ttbi.require_generation_worker_attestation( ..."
    ):
        raise ContractError(
            "state executor does not revalidate worker attestation first"
        )
    _once(
        state_executor,
        "ttbi.require_generation_worker_attestation( ...\n"
        "    worker_attestation, context.provenance);",
        "state executor worker-attestation recheck",
    )
    _once(
        state_executor,
        "ttbi.assert_generation_output_directory( ...\n"
        "    context.run_folder, context.run_folder_observation);",
        "state executor initial output-directory recheck",
    )
    if not (
        state_executor.index("ttbi.require_generation_worker_attestation( ...")
        < state_executor.index("ttbi.assert_generation_output_directory( ...")
        < state_executor.index("state = context.state;")
    ):
        raise ContractError(
            "state executor does not authenticate worker then output directory"
        )
    for token in (
        "if ~isempty(pool) && isvalid(pool)",
        "delete(pool);",
    ):
        _once(pool_deleter, token, f"generation pool teardown {token}")
    for token in (
        "ttbi.smoke_generation_output_boundary(scratch, context);",
        "ttbi.smoke_reject_unauthenticated_generation_pool(context);",
        "ttbi.run_generation_states(context, false, 1);",
        "assert(isempty(gcp('nocreate')), ...",
        "ttbi.publish_generation_completion(publication_folder, context);",
        "manifest = validate_dataset_digest_manifest(publication_folder, 1);",
    ):
        _once(
            GENERATION_WORKER_SMOKE_SOURCE,
            token,
            f"real generation-worker smoke {token}",
        )
    if "ttbi.execute_generation_state(" in GENERATION_WORKER_SMOKE_SOURCE:
        raise ContractError("generation-worker smoke bypasses the real runner")
    for token in (
        "ttbi.seed_stale_generation_credentials(target_folder);",
        "ttbi.create_directory_alias(alias_folder, target_folder);",
        "linked_context.run_folder = alias_folder;",
        "ttbi.directory_observation(target_folder);",
        "ttbi.publish_generation_completion(alias_folder, linked_context);",
        "'ttbi:DirectoryObservationLinked'",
        "credential_names = ttbi.generation_publication_credential_names();",
        "ttbi.path_entry_exists( ...\n"
        "        fullfile(target_folder, credential_names{name_index}))",
    ):
        _once(
            output_boundary_smoke,
            token,
            f"linked output-root smoke {token}",
        )

    # Authenticate the modular completion boundary and all three source fences.
    _once(
        completion_publisher,
        "function publish_generation_completion(run_folder, context)",
        "completion publisher public signature",
    )
    for token in (
        "run_folder_observation = context.run_folder_observation;",
        "ttbi.revoke_generation_publication( ...\n"
        "    run_folder, run_folder_observation);",
        "state_paths = ttbi.inspect_numbered_state_inventory(run_folder, n_states);",
        "if present ~= n_states",
        "if ~all(completed)",
        "if ~strcmp(digest_lines_before, digest_lines) || ...",
        "if ~strcmp(digest_lines_after, digest_lines) || ...",
        "save(temporary_digest, 'file_digests');",
        "movefile(temporary_digest, digest_path, 'f');",
        "ttbi.write_generation_marker_temp( ...",
        "movefile(temporary_marker, marker_path, 'f');",
        "catch publication_error",
        "publication_error = addCause(publication_error, cleanup_error);",
        "rethrow(publication_error);",
    ):
        if token not in completion_publisher:
            raise ContractError(f"completion publication invariant missing: {token}")
    for token, count, label in (
        (
            "ttbi.assert_generator_source_unchanged(provenance);",
            3,
            "source-stability fences",
        ),
        (
            "ttbi.validate_generation_sidecars(run_folder, context);",
            2,
            "sidecar semantic fences",
        ),
        (
            "ttbi.validate_resume_states(run_folder, context);",
            2,
            "state semantic fences",
        ),
        (
            "ttbi.generation_artifact_digests(run_folder, n_states);",
            3,
            "artifact snapshot fences",
        ),
    ):
        if completion_publisher.count(token) != count:
            raise ContractError(
                f"completion publisher requires {count} {label}"
            )
    revoke_call = completion_publisher.index(
        "ttbi.revoke_generation_publication( ..."
    )
    inventory_call = completion_publisher.index(
        "state_paths = ttbi.inspect_numbered_state_inventory(run_folder, n_states);"
    )
    first_early_exit = min(
        completion_publisher.index("return;"),
        completion_publisher.index("error("),
    )
    if not revoke_call < inventory_call < first_early_exit:
        raise ContractError(
            "publication credentials are not revoked before validation can exit"
        )

    expected_credentials = [
        "_GENERATION_COMPLETE",
        "file_digests.mat",
        "._GENERATION_COMPLETE.tmp",
        ".file_digests.mat.tmp",
    ]
    observed_credentials = re.findall(r"'([^']+)'", credential_names_owner)
    if observed_credentials != expected_credentials:
        raise ContractError(
            "central publication credential inventory is not the exact "
            "two-final/two-temporary set"
        )
    for token in (
        "function names = generation_publication_credential_names()",
        "names = { ...",
    ):
        _once(
            credential_names_owner,
            token,
            f"publication credential inventory {token}",
        )
    for token in (
        "function revoke_generation_publication(run_folder, run_folder_observation)",
        "credential_names = ttbi.generation_publication_credential_names();",
        "first_error = cell(0, 1);",
        "for name_index = 1:numel(credential_names)",
        "ttbi.delete_file_entry_if_present( ...",
        "fullfile(run_folder, credential_names{name_index}));",
        "if isempty(first_error)",
        "first_error = {cleanup_error};",
        "if ~isempty(first_error)",
        "rethrow(first_error{1});",
    ):
        _once(publication_revoker, token, f"publication revocation invariant {token}")
    if publication_revoker.count(
        "ttbi.assert_generation_output_directory( ..."
    ) != 3:
        raise ContractError(
            "credential revocation must fence output identity around every deletion"
        )
    if publication_revoker.index("rethrow(first_error{1});") < (
        publication_revoker.index("for name_index = 1:numel(credential_names)")
    ):
        raise ContractError("publication revocation does not finish all credentials")
    for token in (
        "options = ttbi.nofollow_link_options();",
        "javaMethod('exists', 'java.nio.file.Files', nio_path, options);",
        "javaMethod( ...\n    'isDirectory', 'java.nio.file.Files', nio_path, options);",
        "if is_directory",
        "javaMethod('delete', 'java.nio.file.Files', nio_path);",
    ):
        _once(credential_deleter, token, f"safe credential deletion {token}")
    for token in (
        "marker_text = [schema newline fingerprint newline digest_root newline];",
        "expected_bytes = reshape(unicode2native(marker_text, 'UTF-8'), 1, []);",
        "expected_sha256 = ttbi.sha256_bytes(expected_bytes);",
        "file_id = fopen(marker_path, 'wb');",
        "wrote = fwrite(file_id, expected_bytes, 'uint8');",
        "close_status = fclose(file_id);",
        "if wrote ~= numel(expected_bytes) || close_status ~= 0",
        "if ~ttbi.regular_nonsymlink_file(marker_path) || ...",
        "~strcmp(ttbi.stable_file_sha256(marker_path), expected_sha256)",
    ):
        _once(marker_writer, token, f"completion marker writer {token}")
    if not (
        marker_writer.index("wrote = fwrite(file_id, expected_bytes, 'uint8');")
        < marker_writer.index("close_status = fclose(file_id);")
        < marker_writer.index(
            "if wrote ~= numel(expected_bytes) || close_status ~= 0"
        )
        < marker_writer.index("ttbi.stable_file_sha256(marker_path)")
    ):
        raise ContractError("completion marker write/close/hash order is wrong")
    source_fences = [
        match.start()
        for match in re.finditer(
            re.escape("ttbi.assert_generator_source_unchanged(provenance);"),
            completion_publisher,
        )
    ]
    digest_publish = completion_publisher.index(
        "movefile(temporary_digest, digest_path, 'f');"
    )
    marker_publish = completion_publisher.index(
        "movefile(temporary_marker, marker_path, 'f');"
    )
    first_sidecar_fence = completion_publisher.index(
        "ttbi.validate_generation_sidecars(run_folder, context);"
    )
    second_sidecar_fence = completion_publisher.index(
        "ttbi.validate_generation_sidecars(run_folder, context);",
        first_sidecar_fence + 1,
    )
    marker_write = completion_publisher.index(
        "ttbi.write_generation_marker_temp( ..."
    )
    output_fences = [
        match.start()
        for match in re.finditer(
            re.escape("ttbi.assert_generation_output_directory( ..."),
            completion_publisher,
        )
    ]
    if len(output_fences) != 6:
        raise ContractError(
            "completion publisher requires six output-directory identity fences"
        )
    digest_save = completion_publisher.index(
        "save(temporary_digest, 'file_digests');"
    )
    if not (
        source_fences[0] < first_sidecar_fence < digest_publish
        < source_fences[1] < second_sidecar_fence < marker_write
        < source_fences[2] < output_fences[4] < marker_publish
        < output_fences[5]
    ):
        raise ContractError(
            "three source fences do not close validation/digest/final-marker order"
        )
    if not (
        output_fences[0] < revoke_call < inventory_call < output_fences[1]
        < output_fences[2] < digest_save < output_fences[3] < digest_publish
    ):
        raise ContractError(
            "completion publication writes outside the pinned output identity"
        )
    if completion_publisher.count(
        "ttbi.revoke_generation_publication( ..."
    ) != 2:
        raise ContractError(
            "completion publication must revoke credentials before validation "
            "and again on every caught failure"
        )
    if any(token in completion_publisher for token in (
        "fopen(temporary_marker", "fprintf(file_id", "fwrite(file_id"
    )):
        raise ContractError("completion publisher bypasses the marker writer")

    f1_block = state_executor
    for token in (
        "beam_f1 = Beam.Modal.f(1);",
        "if ~isfinite(beam_f1) || beam_f1 < 0.2 || beam_f1 > 15",
        "if strcmp(state.StateFamily{state_index}, 'target_healthy')",
        "healthy_f1_bounds = [3, 6];",
        "healthy_f1_bounds = [2, 4];",
        "if beam_f1 < healthy_f1_bounds(1) || ...",
        "beam_f1 > healthy_f1_bounds(2)",
        "error(['A00: healthy deck f1 %.6g Hz outside [%g,%g] Hz ' ...",
    ):
        if token not in f1_block:
            raise ContractError(f"deck-f1 sanity gate missing: {token}")
    if "data2save.beam_f1_Hz = beam_f1;" not in state_executor:
        raise ContractError("deck-f1 attestation is not saved per state")

    config_at = identity_builder.index(
        "generation_config_json = jsonencode(config);"
    )
    hash_at = identity_builder.index(
        "'gen_fingerprint', ttbi.sha256(generation_config_json)"
    )
    identity_call_at = a00_script.index(
        "identity = ttbi.build_generation_identity( ..."
    )
    case_call_at = a00_script.index("case_info = ttbi.build_case_info( ...")
    writer_call_at = a00_script.index("ttbi.write_run_manifest( ...")
    if not (
        fp_end < config_at < hash_at
        and identity_call_at < case_call_at < writer_call_at
    ):
        raise ContractError(
            "canonical identity is not hashed before manifest publication"
        )


def _must_reject(
    name: str,
    a00: str,
    dataset: str,
    driver: str,
    source_root: str = SOURCE_ROOT_SOURCE,
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
            source_root,
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


def _replace_nth(
    text: str,
    old: str,
    new: str,
    *,
    occurrence: int,
    expected_count: int,
) -> str:
    """Mutate one selected in-memory occurrence after checking multiplicity."""
    if occurrence < 1 or occurrence > expected_count:
        raise AssertionError("mutation occurrence is outside the expected range")
    if text.count(old) != expected_count:
        raise AssertionError(
            f"mutation fixture expected {expected_count} {old!r}, "
            f"found {text.count(old)}"
        )
    start = -1
    for _ in range(occurrence):
        start = text.index(old, start + 1)
    return text[:start] + new + text[start + len(old):]


def _replace_ttbi_once(
    text: str, owner: str, old: str, new: str
) -> str:
    """Mutate one exact token inside one supplied +ttbi owner block."""
    header = f"{TTBI_MARK}{owner}\n"
    owner_start = text.index(header) + len(header)
    owner_end = text.find(TTBI_MARK, owner_start)
    if owner_end < 0:
        owner_end = len(text)
    block = text[owner_start:owner_end]
    if block.count(old) != 1:
        raise AssertionError(
            f"mutation fixture expected one {old!r} in ttbi.{owner}, "
            f"found {block.count(old)}"
        )
    return (
        text[:owner_start]
        + block.replace(old, new, 1)
        + text[owner_end:]
    )


def _replace_ttbi_nth(
    text: str,
    owner: str,
    old: str,
    new: str,
    *,
    occurrence: int,
    expected_count: int,
) -> str:
    """Mutate one selected occurrence inside one +ttbi owner only."""

    header = f"{TTBI_MARK}{owner}\n"
    owner_start = text.index(header) + len(header)
    owner_end = text.find(TTBI_MARK, owner_start)
    if owner_end < 0:
        owner_end = len(text)
    block = text[owner_start:owner_end]
    mutated = _replace_nth(
        block,
        old,
        new,
        occurrence=occurrence,
        expected_count=expected_count,
    )
    return text[:owner_start] + mutated + text[owner_end:]


def _replace_a00_nth(
    text: str,
    old: str,
    new: str,
    *,
    occurrence: int,
    expected_count: int,
) -> str:
    """Mutate the A00 prefix without counting package-owner occurrences."""

    a00_end = text.index(TTBI_MARK)
    prefix = text[:a00_end]
    mutated = _replace_nth(
        prefix,
        old,
        new,
        occurrence=occurrence,
        expected_count=expected_count,
    )
    return mutated + text[a00_end:]


def main() -> None:
    # The reviewed generator source is A00_Run.m PLUS the +ttbi package, which
    # holds the state-identity, seeding, hashing and qualification-receipt
    # helpers that used to be local functions inside A00.  Guards asserting
    # "the generator implements this rule" must see the whole of it, or moving
    # a helper into its own file would silently disarm them.
    #
    # A00's text is kept as the PREFIX so the positional guards below (which
    # assert one statement precedes another via .index()) keep comparing
    # offsets inside A00 alone, and the retired-construct absence checks still
    # cover exactly the same code - only its file changed.  Guards that must
    # stay bound to ONE function slice it back out with ttbi_block().
    _validate_matlab_layout()
    a00 = A00_PATH.read_text(encoding="utf-8") + "".join(
        f"{TTBI_MARK}{path.stem}\n" + ttbi_source(path.stem)
        for path in sorted(TTBI_PKG.glob("*.m"))
    )
    dataset = DATASET_PATH.read_text(encoding="utf-8")
    driver = DRIVER_PATH.read_text(encoding="utf-8")
    validate_contract(a00, dataset, driver)
    print("GENERATION CONTRACT CHECKS")
    print("  [PASS] live MATLAB/Python R11 environment/source contract")

    mutations: list[
        tuple[str, str, str, str, str, str, str, str, str, str]
    ] = []

    def add(
        name: str,
        *,
        ma: str = a00,
        py: str = dataset,
        dr: str = driver,
        source: str = SOURCE_ROOT_SOURCE,
        env: str = ENVIRONMENT_SOURCE,
        save: str = SAVE_PROGRESS_SOURCE,
        micro: str = MAKE_MICRO_SOURCE,
        smoke: str = CRN_SMOKE_SOURCE,
        provenance_smoke: str = PROVENANCE_SMOKE_SOURCE,
    ) -> None:
        mutations.append(
            (
                name, ma, py, dr, source, env, save, micro, smoke,
                provenance_smoke,
            )
        )

    def mutate_fp(old: str, new: str) -> str:
        owner = f"{TTBI_MARK}build_generation_identity\n"
        owner_start = a00.index(owner) + len(owner)
        start = a00.index("config = struct(", owner_start)
        end = a00.index("% The realized state matrices", start)
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
        owner = f"{TTBI_MARK}write_run_manifest\n"
        owner_start = a00.index(owner) + len(owner)
        start = a00.index("save(damage_temp, ...", owner_start)
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
            "_EXPECTED_GENERATION_BEHAVIOR_VERSION = 'generation-rules-v7'",
        ),
    )
    add("behavior removed from fingerprint", ma=mutate_fp(
        f"'{BEHAVIOR_KEY}', {BEHAVIOR_KEY}, ...",
        "'unrelated_config_key', 1, ...",
    ))
    add("behavior removed from case_info", ma=_replace_once(
        a00,
        "'generation_behavior_version', ...\n"
        "        identity.generation_behavior_version, ...",
        "'kept_in_fingerprint_only', 1, ...",
    ))
    add("channel schema removed from fingerprint", ma=mutate_fp(
        "'channel_schema_id', channel_schema_id, ...",
        "'unrelated_channel_schema', channel_schema_id, ...",
    ))
    add("channel schema removed from case_info", ma=_replace_ttbi_once(
        a00,
        "build_case_info",
        "'channel_schema_id', identity.channel_schema_id, ...",
        "'unrelated_channel_schema', identity.channel_schema_id, ...",
    ))
    add("state design kind removed from fingerprint", ma=mutate_fp(
        "'state_design_kind', state.state_design_kind, ...",
        "'unrelated_state_design', state.state_design_kind, ...",
    ))
    add("per-state channel schema assignment removed", ma=_replace_ttbi_once(
        a00,
        "execute_generation_state",
        "data2save.channel_schema_id = context.identity.channel_schema_id;",
        "data2save.channel_schema_id = 'legacy_virtual8'; % MUTANT",
    ))
    add("channel schema omitted from worker context", ma=_replace_ttbi_once(
        a00,
        "build_execution_context",
        "'channel_schema_id', derived_identity.channel_schema_id, ...",
        "'unrelated_channel_schema', derived_identity.channel_schema_id, ...",
    ))
    add("state design kind omitted from worker identity", ma=_replace_ttbi_once(
        a00,
        "build_execution_context",
        "'state_design_kind', derived_identity.state_design_kind, ...",
        "'unrelated_design_kind', derived_identity.state_design_kind, ...",
    ))
    add(
        "Python channel schema stopped deriving from campaign contract",
        py=_replace_once(
            dataset,
            "_EXPECTED_CHANNEL_SCHEMA_ID = EXPECTED_CHANNEL_SCHEMA_ID",
            "_EXPECTED_CHANNEL_SCHEMA_ID = 'legacy_virtual8'",
        ),
    )
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
        ("n_states_multi   = 0;", "n_states_multi   = 1;", "joint states"),
        ("Npass = 50;", "Npass = 49;", "passages"),
        ("n_anchor_levels  = 60;", "n_anchor_levels  = 59;", "anchor levels"),
        ("n_anchor_reps     = 5;", "n_anchor_reps     = 4;", "anchor replicas"),
        ("n_healthy_states  = 5;", "n_healthy_states  = 4;", "healthy states"),
        ("n_nuisance_states = 0;", "n_nuisance_states = 1;", "nuisance states"),
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
        "bearing reference stiffness omitted the 4EI/L factor",
        ma=_replace_once(
            a00,
            "k_ref_bear  = 4 * Beam_probe.Prop.E * Beam_probe.Prop.I / "
            "(L_bridge / num_spans);",
            "k_ref_bear  = Beam_probe.Prop.E * Beam_probe.Prop.I / "
            "(L_bridge / num_spans);",
        ),
    )
    add(
        "bearing fixity-to-stiffness transform made linear",
        ma=_replace_once(
            a00,
            "fix2k       = @(phi) k_ref_bear .* phi ./ (1 - phi);",
            "fix2k       = @(phi) k_ref_bear .* phi;",
        ),
    )
    add(
        "latent bearing fixity bypassed before structural solve",
        ma=_replace_once(
            a00,
            "BearingStates = fix2k(BearingFixity);",
            "BearingStates = BearingFixity;",
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
            "ttbi.state_uniform(StateUID{state_}, ...\n"
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
        ma=_replace_ttbi_once(
            a00,
            "build_state_design",
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
            "'state_stream_names', {state.state_stream_names}",
            "'state_stream_names', state.state_stream_names",
        ),
    )
    add(
        "passage stream names expand fingerprint struct",
        ma=_replace_once(
            a00,
            "'passage_stream_names', {state.passage_stream_names}",
            "'passage_stream_names', state.passage_stream_names",
        ),
    )
    add(
        "operational LHS orientation transposed",
        ma=_replace_once(
            a00,
            "lhs_matrix = lhsdesign(Npass, 2)';",
            "lhs_matrix = lhsdesign(2, Npass);",
        ),
    )
    add(
        "operational LHS marginal-strata gate removed",
        ma=_replace_ttbi_once(
            a00,
            "sample_state_operations",
            "if ~isequal(observed_strata, expected_strata)",
            "if false",
        ),
    )
    add(
        "operational LHS shape gate removed",
        ma=_replace_once(
            a00,
            "if ~isequal(size(lhs_matrix), [2, Npass])",
            "if false",
        ),
    )
    add(
        "operations draw uses crack namespace",
        ma=_replace_ttbi_once(
            a00,
            "execute_generation_state",
            "state.StateNamedStreamSeedID(state_index, 1)",
            "state.StateNamedStreamSeedID(state_index, 2)",
        ),
    )
    add(
        "profile phase uses profile-class namespace",
        ma=_replace_ttbi_once(
            a00,
            "execute_generation_state",
            "state.StateNamedStreamSeedID(state_index, 5)",
            "state.StateNamedStreamSeedID(state_index, 3)",
        ),
    )
    add(
        "per-passage OOR uses profile-passage namespace",
        ma=_replace_ttbi_once(
            a00,
            "execute_generation_state",
            "state.PassageNamedStreamSeedID(state_index, passage_index, 2)",
            "state.PassageNamedStreamSeedID(state_index, passage_index, 1)",
        ),
    )
    add(
        "semantic UID removed from comparison fingerprint",
        ma=_replace_once(
            a00,
            "config.StateUID = state.StateUID;",
            "config.UnrelatedUID = state.StateUID;",
        ),
    )
    add(
        "named passage seeds removed from damage-state manifest",
        ma=mutate_damage_save(
            "'PassageNamedStreamSeedID', 'PassageNamedStreamSeedIDFlat', ...",
            "'UnrelatedPassageSeedID', 'PassageNamedStreamSeedIDFlat', ...",
        ),
    )
    add(
        "state UID removed from per-state payload",
        ma=_replace_once(
            a00,
            "data2save.state_uid = state.StateUID{state_index};",
            "data2save.unrelated_uid = state.StateUID{state_index};",
        ),
    )
    add(
        "cheap state-identity resume inventory removed",
        ma=_replace_once(
            a00,
            "'file_state_uid'};",
            "'unrelated_state_uid'};",
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
    add("production qualification enabled", ma=_replace_a00_nth(
        a00,
        "qualification_run = false;",
        "qualification_run = true;",
        occurrence=1,
        expected_count=1,
    ))
    add("A00 omits explicit qualification marker", ma=_replace_a00_nth(
        a00,
        "'qualification_run', qualification_run, ...",
        "'unrelated_qualification', qualification_run, ...",
        occurrence=1,
        expected_count=1,
    ))
    add("campaign setup accepts a nonlogical qualification marker",
        ma=_replace_ttbi_once(
            a00,
            "campaign_setup",
            "if ~islogical(qualification_run) || ~isscalar(qualification_run)",
            "if false % MUTANT",
        ))
    add("production stage count contract became qualification-conditional",
        ma=_replace_ttbi_once(
            a00,
            "campaign_setup",
            "if ~qualification_run && ~isequal(actual_counts, expected_counts)",
            "if false % MUTANT",
        ))
    add("qualification micro tuple ceiling removed", ma=_replace_ttbi_once(
        a00,
        "campaign_setup",
        "if inputs.Npass > 5 || any(actual_counts > [16 8 8 8 16])",
        "if false % MUTANT",
    ))
    add("qualification reused a production state-design identity",
        ma=_replace_ttbi_once(
            a00,
            "campaign_setup",
            "state_design_kind = 'qualification-five-family-v1';",
            "state_design_kind = 'five-family-multidamage-v2'; % MUTANT",
        ))
    add("qualification state design accepted without its marker",
        ma=_replace_ttbi_once(
            a00,
            "build_state_design",
            "~isequal(config.qualification_run, true)",
            "false % MUTANT",
        ))
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
    add("A00 qualification-evidence delegation removed", ma=_replace_once(
        a00,
        "ttbi.preserve_qualification_evidence( ...\n"
        "    run_folder, qualification, provenance, run_folder_observation);",
        "preserve_qualification_evidence( ...\n"
        "    run_folder, qualification, provenance, run_folder_observation);",
    ))
    add("qualification evidence copies another file", ma=_replace_once(
        a00,
        "qualification.script_path, evidence_path, 'f');",
        "provenance.environment_lock_path, evidence_path, 'f');",
    ))
    add("qualification host-ID gate removed", ma=_replace_once(
        a00,
        "if isempty(regexp(declared_host_id_, ...",
        "if false && isempty(regexp(declared_host_id_, ...",
    ))
    add("qualification host receipt not persisted", ma=_replace_once(
        a00,
        "ttbi.write_qualification_host_receipt( ...",
        "local_ignore_qualification_host_receipt( ...",
    ))
    add("qualification host collision guard removed", ma=_replace_once(
        a00,
        "if ~isequal(observed_bytes_, expected_bytes_)",
        "if false",
    ))
    add("qualification host digest unbound", ma=_replace_once(
        a00,
        "'host_diagnostic_sha256', ttbi.sha256(descriptor_)",
        "'host_diagnostic_sha256', ttbi.sha256(declared_host_id_)",
    ))
    add("dry-ballast sensitivity allowed during qualification",
        ma=_replace_a00_nth(
            a00,
            "assert(~qualification_run, ...\n"
            "        'A00:DryBallastSensitivityQualification'",
            "assert(true, ...\n"
            "        'A00:DryBallastSensitivityQualification'",
            occurrence=1,
            expected_count=1,
        ))
    add("dry-ballast sensitivity allowed on a track-inactive stage",
        ma=_replace_a00_nth(
            a00,
            "if ~campaign_config.use_track_eov",
            "if false % MUTANT",
            occurrence=1,
            expected_count=1,
        ))
    add("dry-ballast reciprocal law disabled", ma=_replace_ttbi_once(
        a00,
        "sample_track_damage",
        "eta_k = 1 / eta_k_base;",
        "eta_k = eta_k_base; % MUTANT",
    ))
    add("dry-ballast persistent-state arm mismatch ignored",
        ma=_replace_ttbi_once(
            a00,
            "sample_track_damage",
            "config_has_arm = isfield(config, 'ballast_dry_stiffness_arm');",
            "config_has_arm = false; % MUTANT",
        ))
    add("dry-ballast arm omitted from generation fingerprint",
        ma=_replace_ttbi_once(
            a00,
            "build_generation_identity",
            "config.ballast_dry_stiffness_arm = ...\n"
            "        ttbi.dry_ballast_stiffness_arm(campaign);",
            "config.unbound_dry_arm = ...\n"
            "        ttbi.dry_ballast_stiffness_arm(campaign); % MUTANT",
        ))
    add("dry-ballast arm omitted from worker context",
        ma=_replace_ttbi_once(
            a00,
            "build_execution_context",
            "context.track.ballast_dry_stiffness_arm = ...\n"
            "        ttbi.dry_ballast_stiffness_arm(campaign);",
            "context.track.unbound_dry_arm = ...\n"
            "        ttbi.dry_ballast_stiffness_arm(campaign); % MUTANT",
        ))
    add("dry-ballast arm omitted from case info", ma=_replace_ttbi_once(
        a00,
        "build_case_info",
        "case_info.ballast_dry_stiffness_arm = ...\n"
        "        ttbi.dry_ballast_stiffness_arm(campaign);",
        "case_info.unbound_dry_arm = ...\n"
        "        ttbi.dry_ballast_stiffness_arm(campaign); % MUTANT",
    ))
    add("profile jitter re-enabled", ma=_replace_once(
        a00, "profile_jitter_sd_mm = 0;", "profile_jitter_sd_mm = 0.5;"))
    add("hanging-sleeper rate drift", ma=_replace_once(
        a00, "hang_rate_100m    = 3.0;", "hang_rate_100m    = 2.5;"))
    add("hanging-sleeper group-size bounds drift", ma=_replace_once(
        a00, "hang_group_size   = [1 5];", "hang_group_size   = [1 4];"))
    add("hanging-sleeper count lost window scaling", ma=_replace_ttbi_once(
        a00,
        "sample_track_damage",
        "group_count = poissrnd(config.hang_rate_100m * track_window / 100);",
        "group_count = poissrnd(config.hang_rate_100m); % MUTANT",
    ))
    add("hanging-sleeper size stopped using reviewed bounds", ma=_replace_ttbi_once(
        a00,
        "sample_track_damage",
        "group_size = randi(config.hang_group_size);",
        "group_size = 1; % MUTANT",
    ))
    add("hanging-sleeper exit bound ignored the realized lattice", ma=_replace_ttbi_once(
        a00,
        "sample_track_damage",
        "latest_group_start = last_sleeper_location - ...",
        "latest_group_start = track_window - ... % MUTANT",
    ))
    add("ballast-patch rate drift", ma=_replace_once(
        a00, "ballast_rate_100m = 1.2;", "ballast_rate_100m = 2.0;"))
    add("pad-failure prevalence drift", ma=_replace_once(
        a00, "pad_p_fail        = 0.02;", "pad_p_fail        = 0.10;"))
    add("analytic unsupported-share statement removed", ma=_replace_once(
        a00,
        "author-chosen stress prior. Its arithmetic 3*3/167 = 5.4% share "
        "assumes a\n% mean group size of three and is not field prevalence.",
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
        "'generator_source_root_sha256', ...\n"
        "        provenance.generator_source_root_sha256, ...",
        "'unrelated_source', ...\n"
        "        provenance.generator_source_root_sha256, ...",
    ))
    add("source root lost confirmed manifest snapshot", source=_replace_once(
        SOURCE_ROOT_SOURCE,
        "[confirmed_root, confirmed_entries] = ttbi.reviewed_source_entries();",
        "confirmed_root = repository_root; confirmed_entries = entries;",
    ))
    add("source root lost second shadow inventory", source=_replace_nth(
        SOURCE_ROOT_SOURCE,
        "ttbi.assert_no_shadow_matlab_sources( ...",
        "% MUTANT: second shadow inventory removed\n% ",
        occurrence=2,
        expected_count=2,
    ))
    add("reviewed manifest stopped validating paths", ma=_replace_ttbi_once(
        a00,
        "reviewed_source_entries",
        "ttbi.validate_repository_relative_path(entry);",
        "% MUTANT: manifest path accepted without validation",
    ))
    add("manifest traversal component accepted", ma=_replace_ttbi_once(
        a00,
        "validate_repository_relative_path",
        "any(strcmp(components, '..'))",
        "false",
    ))
    add("reviewed source hasher lost stable read", ma=_replace_ttbi_once(
        a00,
        "hash_reviewed_source_entries",
        "digest = ttbi.stable_file_sha256(absolute_path);",
        "digest = ttbi.file_sha256(absolute_path);",
    ))
    add("stable source hash became self-comparison", ma=_replace_ttbi_once(
        a00,
        "stable_file_sha256",
        "if ~isequal(before, after)",
        "if ~isequal(before, before)",
    ))
    add("unmanifested MATLAB shadows accepted", ma=_replace_ttbi_once(
        a00,
        "assert_no_shadow_matlab_sources",
        "unexpected = actual(~ismember(actual_folded, manifest_folded));",
        "unexpected = cell(0, 1);",
    ))
    add("MATLAB resolver binding became self-comparison", ma=_replace_ttbi_once(
        a00,
        "assert_reviewed_matlab_resolution",
        "expected_absolute = ttbi.comparison_path(char(expected_nio.toString()));",
        "expected_absolute = ttbi.comparison_path(char(resolved_nio.toString()));",
    ))
    add("campaign environment removed from fingerprint", ma=mutate_fp(
        "'campaign_matlab_environment_sha256', ...",
        "'unrelated_environment_sha256', ...",
    ))
    add("qualification source removed from fingerprint", ma=mutate_fp(
        "'qualification_source_sha256', ...\n"
        "        provenance.qualification_source_sha256, ...",
        "'unrelated_qualification_source', ...\n"
        "        provenance.qualification_source_sha256, ...",
    ))
    add("actual environment improperly added to comparison fingerprint",
        ma=mutate_fp(
            "'campaign_matlab_environment_sha256', ...",
            "'actual_matlab_environment_sha256', ...",
        ))

    add("manifest actual environment self-comparison", ma=_replace_once(
        a00,
        "strcmp(previous.actual_matlab_environment_sha256, ...\n"
        "        provenance.actual_matlab_environment_sha256)",
        "strcmp(previous.actual_matlab_environment_sha256, ...\n"
        "        previous.actual_matlab_environment_sha256)",
    ))
    add("manifest campaign environment self-comparison", ma=_replace_once(
        a00,
        "strcmp(previous.campaign_matlab_environment_sha256, ...\n"
        "        provenance.campaign_matlab_environment_sha256)",
        "strcmp(previous.campaign_matlab_environment_sha256, ...\n"
        "        previous.campaign_matlab_environment_sha256)",
    ))
    add("manifest source root self-comparison", ma=_replace_once(
        a00,
        "strcmp(previous.generator_source_root_sha256, ...\n"
        "        provenance.generator_source_root_sha256)",
        "strcmp(previous.generator_source_root_sha256, ...\n"
        "        previous.generator_source_root_sha256)",
    ))
    add("manifest mode self-comparison", ma=_replace_once(
        a00,
        "previous.release_qualification_run == ...\n"
        "        provenance.release_qualification_run",
        "previous.release_qualification_run == ...\n"
        "        previous.release_qualification_run",
    ))
    add("manifest release self-comparison", ma=_replace_once(
        a00,
        "strcmp(previous.matlab_release, provenance.matlab_release)",
        "strcmp(previous.matlab_release, previous.matlab_release)",
    ))

    add("state actual environment self-comparison", ma=_replace_once(
        a00,
        "isequal(stamps.file_actual_matlab_environment_sha256, ...\n"
        "            provenance.actual_matlab_environment_sha256)",
        "isequal(stamps.file_actual_matlab_environment_sha256, ...\n"
        "            stamps.file_actual_matlab_environment_sha256)",
    ))
    add("state campaign environment self-comparison", ma=_replace_once(
        a00,
        "isequal(stamps.file_campaign_matlab_environment_sha256, ...\n"
        "            provenance.campaign_matlab_environment_sha256)",
        "isequal(stamps.file_campaign_matlab_environment_sha256, ...\n"
        "            stamps.file_campaign_matlab_environment_sha256)",
    ))
    add("state source root self-comparison", ma=_replace_once(
        a00,
        "isequal(stamps.file_generator_source_root_sha256, ...\n"
        "            provenance.generator_source_root_sha256)",
        "isequal(stamps.file_generator_source_root_sha256, ...\n"
        "            stamps.file_generator_source_root_sha256)",
    ))
    add("state mode self-comparison", ma=_replace_once(
        a00,
        "isequal(stamps.file_release_qualification_run, ...\n"
        "            provenance.release_qualification_run)",
        "isequal(stamps.file_release_qualification_run, ...\n"
        "            stamps.file_release_qualification_run)",
    ))
    add("state release self-comparison", ma=_replace_once(
        a00,
        "isequal(stamps.file_matlab_release, provenance.matlab_release)",
        "isequal(stamps.file_matlab_release, stamps.file_matlab_release)",
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
    add("save_progress accepted fewer than twelve identity arguments",
        save=_replace_once(
            SAVE_PROGRESS_SOURCE,
            "narginchk(12, 13);",
            "narginchk(11, 13);",
        ))
    add("save_progress skipped initial output-directory fence",
        save=_replace_nth(
            SAVE_PROGRESS_SOURCE,
            "ttbi.assert_generation_output_directory( ...\n"
            "        run_path, run_folder_observation);",
            "% MUTANT: output directory not authenticated",
            occurrence=1,
            expected_count=4,
        ))
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
        "provenance.generator_source_root_sha256, ...\n"
        "    provenance.qualification_source_sha256, ...\n"
        "    context.run_folder_observation);",
        "provenance.qualification_source_sha256, ...\n"
        "    provenance.qualification_source_sha256, ...\n"
        "    context.run_folder_observation);",
    ))

    add("unsafe automatic worker count restored", ma=_replace_ttbi_once(
        a00,
        "campaign_setup",
        "max_parfor_workers = 4;",
        "max_parfor_workers = 16;",
    ))
    add("A00 generation-runner delegation removed", ma=_replace_once(
        a00,
        "ttbi.run_generation_states( ...\n"
        "    execution_context, completed, "
        "campaign_config.max_parfor_workers);",
        "run_generation_states( ...\n"
        "    execution_context, completed, "
        "campaign_config.max_parfor_workers);",
    ))
    add("generation runner bypasses state executor namespace", ma=_replace_once(
        a00,
        "ttbi.execute_generation_state( ...\n"
        "        state_index, context, worker_attestation);",
        "execute_generation_state( ...\n"
        "        state_index, context, worker_attestation);",
    ))
    add("A00 completion-publisher delegation removed", ma=_replace_once(
        a00,
        "ttbi.publish_generation_completion(run_folder, execution_context);",
        "publish_generation_completion(run_folder, execution_context);",
    ))
    add("dry-ballast sensitivity nested root bypassed authenticated parent",
        ma=_replace_a00_nth(
            a00,
            "results_root = 'Results_sensitivity';",
            "results_root = fullfile('Results_sensitivity', "
            "'dry_ballast_stiffness_sign');",
            occurrence=1,
            expected_count=1,
        ))
    add("dry-ballast sensitivity namespace removed", ma=_replace_a00_nth(
        a00,
        "relative_run_folder = fullfile('dry_ballast_stiffness_sign', ...\n"
        "        campaign_config.ballast_dry_stiffness_arm, case_name);",
        "relative_run_folder = case_name;",
        occurrence=1,
        expected_count=1,
    ))
    add("A00 bypassed safe output-directory creation", ma=_replace_a00_nth(
        a00,
        "run_folder_observation = ttbi.ensure_generation_output_directory( ...\n"
        "    run_folder, results_root, results_root_observation);",
        "run_folder_observation = ttbi.directory_observation(run_folder);",
        occurrence=1,
        expected_count=1,
    ))
    add("Windows filesystem-identity fallback removed",
        ma=_replace_ttbi_once(
            a00,
            "filesystem_identity",
            "identity = ['windows|' ttbi.windows_file_identity(path)];",
            "identity = ['windows|' path]; % MUTANT",
        ))
    add("Windows file identity no longer bound to its volume",
        ma=_replace_ttbi_once(
            a00,
            "windows_file_identity",
            "identity = ['volume-vsn=' volume_before '|file-id=' file_id];",
            "identity = ['file-id=' file_id]; % MUTANT",
        ))
    add("file observation bypassed stable filesystem identity",
        ma=_replace_ttbi_once(
            a00,
            "file_observation",
            "file_identity = ttbi.filesystem_identity(path);",
            "file_identity = path; % MUTANT",
        ))
    add("directory observation bypassed stable filesystem identity",
        ma=_replace_ttbi_once(
            a00,
            "directory_observation",
            "file_identity = ttbi.filesystem_identity(path);",
            "file_identity = path; % MUTANT",
        ))
    add("filesystem identity followed symbolic links",
        ma=_replace_ttbi_once(
            a00,
            "filesystem_identity",
            "options = ttbi.nofollow_link_options();",
            "options = javaArray('java.nio.file.LinkOption', 0); % MUTANT",
        ))
    add("filesystem identity ignored a valid Java fileKey",
        ma=_replace_ttbi_once(
            a00,
            "filesystem_identity",
            "if ~isempty(file_key)",
            "if false",
        ))
    add("filesystem identity accepted an empty Java fileKey string",
        ma=_replace_ttbi_once(
            a00,
            "filesystem_identity",
            "if ~isempty(key_text)",
            "if true",
        ))
    add("Java Boolean boundary accepted arbitrary numeric metadata",
        ma=_replace_ttbi_once(
            a00,
            "java_boolean_value",
            "(raw == 0 || raw == 1)",
            "true",
        ))
    add("path alias guard bypassed the Java Boolean boundary",
        ma=_replace_ttbi_once(
            a00,
            "path_component_is_link_alias",
            "ttbi.java_boolean_value(symbolic)",
            "logical(symbolic)",
        ))
    add("native metadata process lost its finite timeout",
        ma=_replace_ttbi_once(
            a00,
            "run_small_process",
            "if ~finished",
            "if false",
        ))
    add("Windows file identity bypassed the bounded process runner",
        ma=_replace_ttbi_once(
            a00,
            "windows_file_identity",
            "[lines, exit_code] = ttbi.run_small_process( ...",
            "[exit_code, lines] = system( ... % MUTANT",
        ))
    add("Windows file identity trusted an environment-selected executable",
        ma=_replace_ttbi_once(
            a00,
            "windows_file_identity",
            "system_directory = char(System.Environment.SystemDirectory);",
            "system_directory = getenv('SystemRoot'); % MUTANT",
        ))
    add("Windows hard-link query bypassed the bounded process runner",
        ma=_replace_ttbi_once(
            a00,
            "windows_hardlink_count",
            "[raw_lines, exit_code] = ttbi.run_small_process( ...",
            "[exit_code, raw_lines] = system( ... % MUTANT",
        ))
    add("native Windows identity accepted ambiguous fsutil output",
        ma=_replace_ttbi_once(
            a00,
            "windows_file_identity",
            "if exit_code ~= 0 || numel(matches) ~= 1",
            "if exit_code ~= 0 || isempty(matches)",
        ))
    add("native Windows identity accepted sentinel file IDs",
        ma=_replace_ttbi_once(
            a00,
            "windows_file_identity",
            "if all(file_id == '0') || all(file_id == 'f')",
            "if false",
        ))
    add("native Windows identity reused stale volume observation",
        ma=_replace_ttbi_once(
            a00,
            "windows_file_identity",
            "store_after = javaMethod( ...\n"
            "        'getFileStore', 'java.nio.file.Files', absolute_nio);",
            "store_after = store; % MUTANT",
        ))
    add("file observation dropped its second identity fence",
        ma=_replace_ttbi_once(
            a00,
            "file_observation",
            "confirmed_identity = ttbi.filesystem_identity(path);",
            "confirmed_identity = file_identity; % MUTANT",
        ))
    add("directory observation dropped its second identity fence",
        ma=_replace_ttbi_once(
            a00,
            "directory_observation",
            "confirmed_identity = ttbi.filesystem_identity(path);",
            "confirmed_identity = file_identity; % MUTANT",
        ))
    add("directory observation dropped its final alias recheck",
        ma=_replace_ttbi_once(
            a00,
            "directory_observation",
            "if ttbi.path_component_is_link_alias( ...\n"
            "                char(confirmed_cursor.getPath()))",
            "if false",
        ))
    add("MATLAB source traversal reverted to a destructive queue pop",
        ma=_replace_ttbi_once(
            a00,
            "list_matlab_executable_files",
            "folder = pending_paths{next_folder};",
            "folder = pending_paths{1}; % MUTANT",
        ))
    add("sensitivity results returned to MATLAB source traversal",
        ma=_replace_ttbi_once(
            a00,
            "list_matlab_executable_files",
            "{'Results', 'Results_sensitivity'}",
            "{'Results'}; % MUTANT",
        ))
    add("sensitivity results allowed on the MATLAB path",
        ma=_replace_ttbi_once(
            a00,
            "assert_results_not_on_matlab_path",
            "generated_root_names = {'Results', 'Results_sensitivity'};",
            "generated_root_names = {'Results'}; % MUTANT",
        ))
    add("output-directory assertion became a self-comparison",
        ma=_replace_ttbi_once(
            a00,
            "assert_generation_output_directory",
            "if ~isequal(current, expected)",
            "if ~isequal(current, current)",
        ))
    add("output creator accepted parent traversal", ma=_replace_ttbi_once(
        a00,
        "ensure_generation_output_directory",
        "any(ismember(parts, {'.', '..'}))",
        "false",
    ))
    add("state executor skipped output-directory authentication",
        ma=_replace_ttbi_once(
            a00,
            "execute_generation_state",
            "ttbi.assert_generation_output_directory( ...\n"
            "    context.run_folder, context.run_folder_observation);",
            "% MUTANT: worker output path not authenticated",
        ))
    add("credential revoker skipped output-directory authentication",
        ma=_replace_ttbi_nth(
            a00,
            "revoke_generation_publication",
            "ttbi.assert_generation_output_directory( ...",
            "% MUTANT: credential target not authenticated",
            occurrence=1,
            expected_count=3,
        ))
    add("publisher skipped final output-directory fence",
        ma=_replace_ttbi_nth(
            a00,
            "publish_generation_completion",
            "ttbi.assert_generation_output_directory( ...",
            "% MUTANT: final output fence removed",
            occurrence=5,
            expected_count=6,
        ))
    add("publication no longer revokes stale credentials first",
        ma=_replace_ttbi_once(
            a00,
            "publish_generation_completion",
            "ttbi.revoke_generation_publication( ...\n"
            "    run_folder, run_folder_observation);",
            "% MUTANT: stale publication credentials retained",
        ))
    add("central credential inventory omitted completion marker",
        ma=_replace_ttbi_once(
        a00,
        "generation_publication_credential_names",
        "'_GENERATION_COMPLETE';",
        "'_UNRELATED_CREDENTIAL';",
    ))
    add("publication revoker bypassed safe entry deletion",
        ma=_replace_ttbi_once(
            a00,
            "revoke_generation_publication",
            "ttbi.delete_file_entry_if_present( ...",
            "delete( ...",
        ))
    add("credential deleter accepted directories", ma=_replace_ttbi_once(
        a00,
        "delete_file_entry_if_present",
        "if is_directory",
        "if false",
    ))
    add("publication catch skipped complete credential revocation",
        ma=_replace_ttbi_once(
            a00,
            "publish_generation_completion",
            "ttbi.revoke_generation_publication( ...\n"
            "            run_folder, run_folder_observation);",
            "% MUTANT: publication failure retained credentials",
        ))
    add("parfor worker cap bypassed", ma=_replace_once(
        a00,
        "parfor (state_index = 1:n_states, pool_workers)",
        "parfor state_index = 1:n_states",
    ))
    add("default cluster profile restored", ma=_replace_once(
        a00, "cluster = parcluster('Processes');", "cluster = parcluster();"))
    add("thread-pool profile requested", ma=_replace_once(
        a00,
        "cluster = parcluster('Processes');",
        "cluster = parcluster('Threads');",
    ))
    add("unsafe existing pool not deleted", ma=_replace_once(
        a00, "    delete(existing_pool);\n", ""))
    add("all-complete resume guard removed", ma=_replace_once(
        a00, "if all(completed)", "if false"))
    add("generation worker Constant removed", ma=_replace_ttbi_once(
        a00,
        "run_generation_states",
        "worker_source = parallel.pool.Constant( ...\n"
        "    @() ttbi.authenticate_generation_worker(provenance));",
        "worker_source = struct('Value', struct()); % MUTANT",
    ))
    add("generation loop stopped consuming worker Constant",
        ma=_replace_ttbi_once(
            a00,
            "run_generation_states",
            "worker_attestation = worker_source.Value;",
            "worker_attestation = struct(); % MUTANT",
        ))
    add("generation loop skipped worker attestation validation",
        ma=_replace_ttbi_once(
            a00,
            "run_generation_states",
            "ttbi.require_generation_worker_attestation( ...\n"
            "        worker_attestation, provenance);",
            "% MUTANT: cached worker proof not validated",
        ))
    add("state executor skipped defense-in-depth attestation",
        ma=_replace_ttbi_once(
            a00,
            "execute_generation_state",
            "ttbi.require_generation_worker_attestation( ...\n"
            "    worker_attestation, context.provenance);",
            "% MUTANT: executor trusts caller",
        ))
    add("worker authenticator allowed client construction",
        ma=_replace_ttbi_once(
            a00,
            "authenticate_generation_worker",
            "if isempty(task)",
            "if false",
        ))
    add("worker attestation dropped digest-line binding",
        ma=_replace_ttbi_once(
            a00,
            "require_generation_worker_attestation",
            "strcmp(attestation.generator_source_digest_lines, ...\n"
            "        provenance.generator_source_digest_lines) && ...",
            "true && ... % MUTANT",
        ))
    add("worker attestation dropped schema binding",
        ma=_replace_ttbi_once(
            a00,
            "require_generation_worker_attestation",
            "strcmp(attestation.gen_schema, provenance.gen_schema) && ...",
            "true && ... % MUTANT",
        ))
    add("worker attestation dropped fingerprint binding",
        ma=_replace_ttbi_once(
            a00,
            "require_generation_worker_attestation",
            "strcmp(attestation.gen_fingerprint, provenance.gen_fingerprint) && ...",
            "true && ... % MUTANT",
        ))
    add("worker attestation dropped source-root binding",
        ma=_replace_ttbi_once(
            a00,
            "require_generation_worker_attestation",
            "strcmp(attestation.generator_source_root_sha256, ...\n"
            "        provenance.generator_source_root_sha256) && ...",
            "true && ... % MUTANT",
        ))
    add("worker attestation dropped source-count binding",
        ma=_replace_ttbi_once(
            a00,
            "require_generation_worker_attestation",
            "isequal(attestation.generator_source_file_count, ...\n"
            "        provenance.generator_source_file_count) && ...",
            "true && ... % MUTANT",
        ))
    add("generation pool cleanup disabled", ma=_replace_ttbi_once(
        a00,
        "run_generation_states",
        "pool_cleanup = onCleanup(@() ttbi.delete_generation_pool(pool));",
        "pool_cleanup = onCleanup(@() []); % MUTANT",
    ))
    add("A00 pre-run source fence removed", ma=_replace_a00_nth(
        a00,
        "ttbi.assert_generator_source_unchanged(provenance);",
        "% MUTANT: pre-run source gate removed",
        occurrence=1,
        expected_count=2,
    ))
    add("A00 post-run source fence removed", ma=_replace_a00_nth(
        a00,
        "ttbi.assert_generator_source_unchanged(provenance);",
        "% MUTANT: post-run source gate removed",
        occurrence=2,
        expected_count=2,
    ))
    add("publisher initial source fence removed", ma=_replace_ttbi_nth(
        a00,
        "publish_generation_completion",
        "ttbi.assert_generator_source_unchanged(provenance);",
        "% MUTANT: initial publication source gate removed",
        occurrence=1,
        expected_count=3,
    ))
    add("publisher post-digest source fence removed", ma=_replace_ttbi_nth(
        a00,
        "publish_generation_completion",
        "ttbi.assert_generator_source_unchanged(provenance);",
        "% MUTANT: post-digest source gate removed",
        occurrence=2,
        expected_count=3,
    ))
    add("publisher final pre-marker source fence removed",
        ma=_replace_ttbi_nth(
            a00,
            "publish_generation_completion",
            "ttbi.assert_generator_source_unchanged(provenance);",
            "% MUTANT: final pre-marker source gate removed",
            occurrence=3,
            expected_count=3,
        ))
    add("pre-marker sidecar semantic gate removed", ma=_replace_ttbi_nth(
        a00,
        "publish_generation_completion",
        "ttbi.validate_generation_sidecars(run_folder, context);",
        "% MUTANT: pre-marker sidecar gate removed",
        occurrence=2,
        expected_count=2,
    ))
    add("global deck-f1 gate removed", ma=_replace_once(
        a00,
        "if ~isfinite(beam_f1) || beam_f1 < 0.2 || beam_f1 > 15",
        "if false",
    ))
    add("healthy deck-f1 family gate laundered", ma=_replace_once(
        a00,
        "if strcmp(state.StateFamily{state_index}, 'target_healthy')",
        "if true",
    ))
    add("healthy deck-f1 bounds removed", ma=_replace_once(
        a00,
        "if beam_f1 < healthy_f1_bounds(1) || ...\n"
        "                    beam_f1 > healthy_f1_bounds(2)",
        "if false",
    ))

    add(
        "qualification default reverted to a retired stage",
        micro=_replace_nth(
            MAKE_MICRO_SOURCE,
            'stage: str = "F40-S"',
            'stage: str = "s0_scour"',
            occurrence=1,
            expected_count=3,
        ),
    )
    add(
        "unmarked micro API guard removed",
        micro=_replace_nth(
            MAKE_MICRO_SOURCE,
            "if not qualification:",
            "if qualification:",
            occurrence=1,
            expected_count=3,
        ),
    )
    add(
        "no-argument micro CLI retirement removed",
        micro=_replace_once(
            MAKE_MICRO_SOURCE,
            '"no-argument micro generation is retired; use --qualification "',
            '"no-argument micro generation enabled; "',
        ),
    )
    add(
        "legacy toy dry-run retirement removed",
        micro=_replace_once(
            MAKE_MICRO_SOURCE,
            '"the legacy toy dry-run is retired; use four-stage release "',
            '"the legacy toy dry-run is enabled; use four-stage release "',
        ),
    )
    add(
        "MATLAB CRN smoke lost namespace-isolation assertion",
        smoke=_replace_once(
            CRN_SMOKE_SOURCE,
            "assert(isequal(base.track,mutated.track));",
            "assert(true);",
        ),
    )
    add(
        "MATLAB CRN smoke lost exact F40 matched count",
        smoke=_replace_once(
            CRN_SMOKE_SOURCE,
            "assert(numel(shared_uid) == 30);",
            "assert(numel(shared_uid) == 29);",
        ),
    )
    add(
        "MATLAB CRN smoke numeric oracle drifted",
        smoke=_replace_once(
            CRN_SMOKE_SOURCE,
            "uint32(1955233256)",
            "uint32(1955233257)",
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
        source,
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
            source,
            env,
            save,
            micro,
            smoke,
            provenance_smoke,
        )
    print(f"GENERATION CONTRACT: ALL PASS ({len(mutations)} mutations caught)")

    _validate_damage_physics_contract(
        DAMAGE_PHYSICS_SOURCE,
        DAMAGE_MODAL_SOURCE,
        DAMAGE_DAMPING_SOURCE,
        DAMAGE_SMOKE_SOURCE,
    )
    damage_mutations = (
        (
            "nominal support stiffness drift",
            _replace_once(
                DAMAGE_PHYSICS_SOURCE,
                "DOF_Original_value = 344e6;",
                "DOF_Original_value = 345e6;",
            ),
            DAMAGE_SMOKE_SOURCE,
        ),
        (
            "scour loss sign inverted",
            _replace_once(
                DAMAGE_PHYSICS_SOURCE,
                "retained_stiffness = 1.0 - ...\n"
                "    Damage.scour_rates(positive_vertical_supports);",
                "retained_stiffness = 1.0 + ...\n"
                "    Damage.scour_rates(positive_vertical_supports);",
            ),
            DAMAGE_SMOKE_SOURCE,
        ),
        (
            "left bearing stiffness disconnected",
            _replace_once(
                DAMAGE_PHYSICS_SOURCE,
                "rot_stiff_values(positive_rotational_supports == 1) = ...\n"
                "    Damage.bearing_left;",
                "rot_stiff_values(positive_rotational_supports == 1) = 0;",
            ),
            DAMAGE_SMOKE_SOURCE,
        ),
        (
            "right bearing branch disabled",
            _replace_once(
                DAMAGE_PHYSICS_SOURCE,
                "rot_stiff_values(positive_rotational_supports == "
                "Beam.BC.supp_num) = ...\n"
                "    Damage.bearing_right;",
                "rot_stiff_values(positive_rotational_supports == "
                "Beam.BC.supp_num) = 0;",
            ),
            DAMAGE_SMOKE_SOURCE,
        ),
        (
            "scour range guard weakened",
            _replace_once(
                DAMAGE_PHYSICS_SOURCE,
                "any(Damage.scour_rates(:) < 0 | Damage.scour_rates(:) > 1)",
                "any(Damage.scour_rates(:) < 0)",
            ),
            DAMAGE_SMOKE_SOURCE,
        ),
        (
            "support-count guard removed",
            _replace_once(
                DAMAGE_PHYSICS_SOURCE,
                "numel(Damage.scour_rates) ~= Beam.BC.supp_num",
                "false",
            ),
            DAMAGE_SMOKE_SOURCE,
        ),
        (
            "analytic retained-stiffness oracle drift",
            DAMAGE_PHYSICS_SOURCE,
            _replace_once(
                DAMAGE_SMOKE_SOURCE,
                "expected_k4 = [344e6 1.25e9 0.70*344e6 "
                "0.40*344e6 344e6 3.75e9];",
                "expected_k4 = [344e6 1.25e9 0.80*344e6 "
                "0.40*344e6 344e6 3.75e9];",
            ),
        ),
        (
            "redux-zero rail no-op oracle removed",
            DAMAGE_PHYSICS_SOURCE,
            _replace_once(
                DAMAGE_SMOKE_SOURCE,
                "assert(isempty(Rail4.BC.DOF_with_values), ...",
                "assert(true, ...",
            ),
        ),
        (
            "elastic supports ignored in rigid-mode count",
            _replace_once(
                DAMAGE_PHYSICS_SOURCE,
                "Beam.Modal.num_rigid_modes = 2 - rigid_constraint_rank;",
                "Beam.Modal.num_rigid_modes = max([0, 2 - "
                "Beam.BC.num_DOF_fixed]);",
            ),
            DAMAGE_SMOKE_SOURCE,
        ),
    )
    for name, b02, smoke in damage_mutations:
        _must_reject_damage_mapping(name, b02, smoke)
    _must_reject_damage_mapping(
        "unsorted generalized eigenvalues feed modal damping",
        DAMAGE_PHYSICS_SOURCE,
        DAMAGE_SMOKE_SOURCE,
        b09=_replace_once(
            DAMAGE_MODAL_SOURCE,
            "[lambda,~] = local_sorted_eigenvalues( ...\n"
            "        lambda, Beam.Modal.num_rigid_modes);",
            "lambda = real(lambda);",
        ),
    )
    _must_reject_damage_mapping(
        "modal round-off tolerance scales loosely with global spectrum",
        DAMAGE_PHYSICS_SOURCE,
        DAMAGE_SMOKE_SOURCE,
        b09=_replace_once(
            DAMAGE_MODAL_SOURCE,
            "tol = 128 * eps(max(scale));",
            "tol = 1e-10 * max(scale);",
        ),
    )
    _must_reject_damage_mapping(
        "eigenproblem rigid-mode count is not cross-checked",
        DAMAGE_PHYSICS_SOURCE,
        DAMAGE_SMOKE_SOURCE,
        b09=_replace_once(
            DAMAGE_MODAL_SOURCE,
            "if sum(near_zero) ~= expected_rigid_modes",
            "if false",
        ),
    )
    _must_reject_damage_mapping(
        "Rayleigh calibration hard-coded to modes three and four",
        DAMAGE_PHYSICS_SOURCE,
        DAMAGE_SMOKE_SOURCE,
        b24=_replace_once(
            DAMAGE_DAMPING_SOURCE,
            "ref_modes = (1:2) + Beam.Modal.num_rigid_modes;",
            "ref_modes = 3:4;",
        ),
    )
    print(
        "DAMAGE-PHYSICS CONTRACT: ALL PASS "
        f"({len(damage_mutations) + 4} mutations caught)"
    )

    _validate_bridge_mesh_alignment_contract(
        MESH_OPTIONS_SOURCE,
        MESH_SELECTOR_SOURCE,
        DAMAGE_PHYSICS_SOURCE,
        MESH_ALIGNMENT_SMOKE_SOURCE,
        BALLAST_MODEL_SOURCE,
        BALLAST_MIRROR_SOURCE,
        NUMERICAL_VV_PREFLIGHT_SOURCE,
        NUMERICAL_VV_SMOKE_SOURCE,
    )
    mesh_mutations = (
        (
            "A04 disconnected from aligned bridge selector",
            {
                "options": _replace_once(
                    MESH_OPTIONS_SOURCE,
                    "Beam.Mesh.Ele.num_per_spacing = "
                    "bridge_mesh_elements_per_sleeper( ...\n"
                    "    Beam.Prop.L, Beam.Prop.num_spans, "
                    "Track.Sleeper.spacing, 2);",
                    "Beam.Mesh.Ele.num_per_spacing = 2;",
                )
            },
        ),
        (
            "selector ignores equal-span support alignment",
            {
                "selector": _replace_once(
                    MESH_SELECTOR_SOURCE,
                    "abs(span_element_count-round(span_element_count)) "
                    "<= tolerance",
                    "true",
                )
            },
        ),
        (
            "B02 support-on-node gate disabled",
            {
                "b02": _replace_once(
                    DAMAGE_PHYSICS_SOURCE,
                    "if ~isempty(misaligned_spring_supports)",
                    "if false && ~isempty(misaligned_spring_supports)",
                )
            },
        ),
        (
            "B02 fine-mesh cumulative-roundoff allowance removed",
            {
                "b02": _replace_once(
                    DAMAGE_PHYSICS_SOURCE,
                    "summation_roundoff_factor = max(256, "
                    "2*mesh_element_count);",
                    "summation_roundoff_factor = 256;",
                )
            },
        ),
        (
            "L60 aligned-density oracle drift",
            {
                "smoke": _replace_once(
                    MESH_ALIGNMENT_SMOKE_SOURCE,
                    "'expected_elements_per_bay', {3, 2, 3}",
                    "'expected_elements_per_bay', {2, 2, 2}",
                )
            },
        ),
        (
            "B54 ballast mass spread over every bridge vertical node",
            {
                "b54": _replace_once(
                    BALLAST_MODEL_SOURCE,
                    "Model.Mesh.DOF.beam_vert_under_sleeper,...\n"
                    "    funDiag(Track.Sleeper.num_onbeam,"
                    "Track.BallastOnBeam.Prop.m));",
                    "Model.Mesh.DOF.beam_vert,...\n"
                    "    funDiag(Beam.Mesh.Nodes.Tnum,"
                    "Track.BallastOnBeam.Prop.m/"
                    "Beam.Mesh.Ele.num_per_spacing));",
                )
            },
        ),
        (
            "B54 canonical ballast add canceled before legacy mass readdition",
            {
                "b54": _replace_once(
                    BALLAST_MODEL_SOURCE,
                    "Model.Mesh.Mg = funAdd1(Model.Mesh.Mg,...\n"
                    "    Model.Mesh.DOF.beam_vert_under_sleeper,...\n"
                    "    funDiag(Track.Sleeper.num_onbeam,"
                    "Track.BallastOnBeam.Prop.m));",
                    "Model.Mesh.Mg = funAdd1(Model.Mesh.Mg,...\n"
                    "    Model.Mesh.DOF.beam_vert_under_sleeper,...\n"
                    "    funDiag(Track.Sleeper.num_onbeam,"
                    "Track.BallastOnBeam.Prop.m));\n"
                    "Model.Mesh.Mg = funAdd1(Model.Mesh.Mg,...\n"
                    "    Model.Mesh.DOF.beam_vert_under_sleeper,...\n"
                    "    -funDiag(Track.Sleeper.num_onbeam,"
                    "Track.BallastOnBeam.Prop.m));\n"
                    "Model.Mesh.Mg = funAdd1(Model.Mesh.Mg,...\n"
                    "    Model.Mesh.DOF.beam_vert,...\n"
                    "    funDiag(Beam.Mesh.Nodes.Tnum,"
                    "Track.BallastOnBeam.Prop.m/"
                    "Beam.Mesh.Ele.num_per_spacing));",
                )
            },
        ),
        (
            "Python B54 ballast mass spread over every bridge vertical node",
            {
                "py_b54": _replace_once(
                    BALLAST_MIRROR_SOURCE,
                    "Model.Mesh.DOF.beam_vert_under_sleeper,\n"
                    "        funDiag(Track.Sleeper.num_onbeam, "
                    "Track.BallastOnBeam.Prop.m),",
                    "Model.Mesh.DOF.beam_vert,\n"
                    "        funDiag(Beam.Mesh.Nodes.Tnum, "
                    "Track.BallastOnBeam.Prop.m / "
                    "Beam.Mesh.Ele.num_per_spacing),",
                )
            },
        ),
        (
            "ballast assembly oracle defaults to nonproduction redux path",
            {
                "vv_preflight": _replace_once(
                    NUMERICAL_VV_PREFLIGHT_SOURCE,
                    "addParameter(parser, 'Redux', 0, "
                    "@local_logical_scalar);",
                    "addParameter(parser, 'Redux', 1, "
                    "@local_logical_scalar);",
                )
            },
        ),
        (
            "refined L99.6 ballast assembly oracle removed",
            {
                "vv_smoke": _replace_once(
                    NUMERICAL_VV_SMOKE_SOURCE,
                    "fine_L99 = numerical_vv_coupled_mesh_preflight("
                    "99.6, 4, 4, 4, ...\n"
                    "    'Assemble', true, 'Redux', 1);",
                    "fine_L99 = aligned_L99;",
                )
            },
        ),
        (
            "registered M3 ballast assembly oracles removed",
            {
                "vv_smoke": _replace_once(
                    NUMERICAL_VV_SMOKE_SOURCE,
                    "for density = [12, 8; 24, 16]'",
                    "for density = [12, 8; 12, 8]'",
                )
            },
        ),
        (
            "M3 preflight cumulative-roundoff allowance removed",
            {
                "vv_preflight": _replace_once(
                    NUMERICAL_VV_PREFLIGHT_SOURCE,
                    "summation_roundoff_factor = max(256, "
                    "2*bridge_element_count);",
                    "summation_roundoff_factor = 256;",
                )
            },
        ),
    )
    for name, changed in mesh_mutations:
        _must_reject_bridge_mesh_alignment(name, **changed)
    print(
        "BRIDGE-MESH ALIGNMENT CONTRACT: ALL PASS "
        f"({len(mesh_mutations)} mutations caught)"
    )


if __name__ == "__main__":
    main()

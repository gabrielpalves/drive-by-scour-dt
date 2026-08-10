"""Fail-closed inventory of pairwise MATLAB host-qualification receipts.

The pairwise comparator deliberately validates one stage and two hosts at a
time.  This module closes the campaign-level graph: for an explicitly declared
set of intended generation hosts it requires exactly one accepted, current
``matlab-environment-qualification-receipt-v4`` for every unordered host pair
and each mandatory qualification stage - and then reruns the complete
pairwise comparator behind every accepted edge.  That recomputation reopens
both retained endpoint directories, authenticates every payload and sidecar,
and requires the retained verdict, evidence blocks, comparison statistics and
raw-identity count to match exactly.

What that closes, stated exactly. A receipt graph whose recorded endpoints are
missing, moved, mutated after qualification, substituted for a different real
directory, reached through a symlink or junction, or whose retained evidence
disagrees with the bytes now on disk (including after a coherently regenerated
digest table) cannot pass. What it does NOT close: a graph accompanied by
fabricated but internally consistent datasets, because the host receipts it
rests on are SELF-ATTESTED - self-reported diagnostics covered by a digest over
their own descriptor, with no signing key, hardware attestation, or independent
witness. The repository's own fixture builder writes acceptable host receipts
and datasets from Python with SciPy and validates. Read every result of this
module as retained-artifact integrity and graph completeness under a
trusted-operator threat model, never as proof of physical execution origin.

The validation stack is split into four modules whose joint identity is
bound into every inventory root and safe receipt:

* ``qualification_receipt_schema`` - strict receipt JSON grammar (no disk);
* ``qualification_endpoint_revalidation`` - retained-endpoint reopening and
  full comparator recomputation (this is the layer that touches datasets);
* ``qualification_path_safety`` - canonical receipt paths and exact
  byte/identity snapshots;
* this module - graph closure, unconditional edge-revalidation dispatch, safe
  aggregate receipt, and the CLI.

The inventory never prints numerical results or substitutes its own numerical
test.  It delegates every accepted edge to the authoritative comparator, then
emits only identities and counts in the safe aggregate receipt.
"""
from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{_unsafe_python_path_variable} must be absent before evidence "
            "imports"
        )
_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.dirname(__file__)
)
_bootstrap_first_path = _bootstrap_sys.path[0] or _bootstrap_os.getcwd()
if (
    _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    or _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_source_root
    ))
):
    raise RuntimeError(
        "reviewed repository root must be the canonical first import path"
    )
_bootstrap_guard_dir = _bootstrap_os.path.join(
    _bootstrap_source_root, "campaign_import_guard"
)
_bootstrap_guard_init = _bootstrap_os.path.join(
    _bootstrap_guard_dir, "__init__.py"
)
if (
    not _bootstrap_os.path.isfile(_bootstrap_guard_init)
    or _bootstrap_os.path.islink(_bootstrap_guard_init)
    or _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_guard_dir
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_dir
    ))
    or any(
        entry.casefold().startswith("__init__.")
        and entry != "__init__.py"
        for entry in _bootstrap_os.listdir(_bootstrap_guard_dir)
    )
):
    raise RuntimeError(
        "reviewed campaign import guard package is absent or ambiguous"
    )
_bootstrap_loaded_guard = _bootstrap_sys.modules.get("campaign_import_guard")
if _bootstrap_loaded_guard is not None and (
    _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        getattr(_bootstrap_loaded_guard, "__file__", "")
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_init
    ))
    or getattr(_bootstrap_loaded_guard, "_BOUNDARY_ENFORCED", False) is not True
):
    raise RuntimeError(
        "preloaded campaign import guard is not the reviewed enforced module"
    )
from campaign_import_guard import (  # noqa: E402
    enforce_import_boundary as _enforce_import_boundary,
)
_enforce_import_boundary()

import argparse
import itertools
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import compare_generation_releases as comparator
import qualification_endpoint_revalidation as endpoint_revalidation
import qualification_path_safety as path_safety
import qualification_receipt_schema as schema
from qualification_receipt_schema import (  # noqa: F401  (re-exported API)
    CurrentInventoryPolicy,
    EndpointBinding,
    HostHardwareBinding,
    QualificationInventoryError,
    RECEIPT_SCHEMA,
    REQUIRED_STAGES,
    ReceiptEdge,
    _HOST_ID_RE,
    _SEMANTIC_EXACTNESS_DEFINITION,
    _canonical_json_bytes,
    _sha256_bytes,
    _sha256_text,
    _text,
)


INVENTORY_SCHEMA = "ttbi-matlab-qualification-inventory-v1"

# The modules that jointly implement qualification-evidence validation.
# Their combined root digest is the "checker" identity bound into every
# inventory root and safe aggregate receipt, so validation logic cannot
# migrate into satellite files outside the bound identity.
_QUALIFICATION_MODULE_NAMES = (
    "qualification_endpoint_revalidation.py",
    "qualification_path_safety.py",
    "qualification_receipt_inventory.py",
    "qualification_receipt_schema.py",
)


@dataclass(frozen=True)
class InventoryResult:
    intended_host_ids: tuple[str, ...]
    edges: tuple[ReceiptEdge, ...]
    endpoint_bindings: dict[tuple[str, str], EndpointBinding]
    inventory_root_sha256: str
    policy: CurrentInventoryPolicy


def _qualification_module_paths() -> tuple[tuple[str, Path], ...]:
    modules = {
        "qualification_receipt_inventory.py": Path(__file__).resolve(),
        "qualification_receipt_schema.py": Path(schema.__file__).resolve(),
        "qualification_endpoint_revalidation.py": Path(
            endpoint_revalidation.__file__
        ).resolve(),
        "qualification_path_safety.py": Path(
            path_safety.__file__
        ).resolve(),
    }
    return tuple((name, modules[name]) for name in _QUALIFICATION_MODULE_NAMES)


def current_policy() -> CurrentInventoryPolicy:
    comparator_path = Path(comparator.__file__).resolve()
    module_paths = _qualification_module_paths()
    source_paths = (
        (comparator_path, "qualification comparator"),
        *(
            (path, f"qualification module {name}")
            for name, path in module_paths
        ),
    )
    for path, label in source_paths:
        if not path.is_file() or path.is_symlink():
            raise QualificationInventoryError(
                f"current {label} must be one regular non-symlink file: {path}"
            )
    for name, path in module_paths:
        if path.name != name:
            raise QualificationInventoryError(
                f"qualification module {name} was loaded from an unexpected "
                f"file: {path}"
            )
    try:
        source_bytes = {
            path: comparator._read_single_link_regular_bytes(path, label)
            for path, label in source_paths
        }
        policy = comparator._current_policy()
        source_shas = {
            stage: comparator._expected_qualification_source_sha256(
                stage, policy.source_snapshot
            )
            for stage in REQUIRED_STAGES
        }
        executed_shas = {
            stage: _sha256_bytes(
                comparator._expected_qualification_source_bytes(
                    stage, policy.source_snapshot
                )
            )
            for stage in REQUIRED_STAGES
        }
    except (comparator.QualificationInputError, OSError, RuntimeError) as exc:
        raise QualificationInventoryError(
            f"cannot authenticate current comparator policy: {exc}"
        ) from exc
    # One root digest over all qualification modules; sorted
    # "name:sha" lines, matching the repository's source-root convention.
    checker_root_material = "".join(
        f"{name}:{_sha256_bytes(source_bytes[path])}\n"
        for name, path in sorted(module_paths)
    )
    result = CurrentInventoryPolicy(
        comparator_sha256=_sha256_bytes(source_bytes[comparator_path]),
        checker_sha256=_sha256_text(checker_root_material),
        environment_lock_sha256=policy.environment_lock_sha256,
        parser_environment=dict(policy.parser_environment),
        python_runtime_source_root_sha256=(
            policy.python_runtime_source_root_sha256
        ),
        python_runtime_source_file_count=(
            policy.python_runtime_source_file_count
        ),
        generator_source_root_sha256=policy.generator_source_root_sha256,
        generator_source_file_count=policy.generator_source_file_count,
        campaign_matlab_release=policy.campaign_matlab_release,
        campaign_matlab_environment_descriptor=(
            policy.campaign_matlab_environment_descriptor
        ),
        campaign_matlab_environment_sha256=(
            policy.campaign_matlab_environment_sha256
        ),
        gen_schema=policy.gen_schema,
        channel_schema_id=policy.channel_schema_id,
        generation_behavior_version=policy.generation_behavior_version,
        max_parfor_workers=policy.max_parfor_workers,
        qualification_source_sha256=source_shas,
        qualification_executed_file_sha256=executed_shas,
    )
    try:
        comparator._assert_policy_sources_unchanged(
            policy, "qualification inventory current policy"
        )
        for path, label in source_paths:
            if (
                comparator._read_single_link_regular_bytes(path, label)
                != source_bytes[path]
            ):
                raise QualificationInventoryError(
                    f"current {label} changed during policy authentication"
                )
    except (comparator.QualificationInputError, OSError, RuntimeError) as exc:
        raise QualificationInventoryError(
            f"current qualification source changed during policy "
            f"authentication: {exc}"
        ) from exc
    return result


def validate_host_ids(host_ids: Sequence[str]) -> tuple[str, ...]:
    if len(host_ids) < 2:
        raise QualificationInventoryError(
            "at least two intended qualification host IDs are required"
        )
    validated: list[str] = []
    for index, raw in enumerate(host_ids):
        host = _text(raw, f"host_ids[{index}]")
        if not _HOST_ID_RE.fullmatch(host):
            raise QualificationInventoryError(
                f"host_ids[{index}]={host!r} does not match "
                "[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
            )
        validated.append(host)
    if len(set(validated)) != len(validated):
        raise QualificationInventoryError(
            "intended qualification host IDs must be distinct"
        )
    return tuple(sorted(validated))


def collect_receipt_paths(
    *,
    receipt_files: Iterable[Path | str] = (),
    receipt_dirs: Iterable[Path | str] = (),
) -> tuple[Path, ...]:
    try:
        return path_safety.collect_receipt_paths(
            receipt_files=receipt_files,
            receipt_dirs=receipt_dirs,
        )
    except path_safety.QualificationPathError as exc:
        raise QualificationInventoryError(str(exc)) from exc


def validate_inventory(
    host_ids: Sequence[str],
    receipt_paths: Sequence[Path | str],
) -> InventoryResult:
    hosts = validate_host_ids(host_ids)
    policy = current_policy()
    expected_keys = {
        (stage, host_a, host_b)
        for stage in REQUIRED_STAGES
        for host_a, host_b in itertools.combinations(hosts, 2)
    }
    paths = collect_receipt_paths(receipt_files=receipt_paths)
    if len(paths) != len(expected_keys):
        raise QualificationInventoryError(
            f"receipt count is not exactly {len(REQUIRED_STAGES)}*C(H,2): "
            f"observed={len(paths)}, expected={len(expected_keys)}"
        )
    try:
        receipt_snapshots = tuple(
            path_safety.snapshot_receipt(
                path, f"qualification pair receipt {index}")
            for index, path in enumerate(paths, 1)
        )
    except path_safety.QualificationPathError as exc:
        raise QualificationInventoryError(str(exc)) from exc

    edges_by_key: dict[tuple[str, str, str], ReceiptEdge] = {}
    endpoint_bindings: dict[tuple[str, str], EndpointBinding] = {}
    endpoint_datasets: dict[tuple[str, str], dict[str, Any]] = {}
    retained_receipts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path_index, path in enumerate(paths):
        (
            edge,
            bindings,
            retained_datasets,
            retained_receipt,
        ) = schema._load_receipt(
            path,
            intended_hosts=frozenset(hosts),
            policy=policy,
            receipt_bytes=receipt_snapshots[path_index].raw,
        )
        if edge.key in edges_by_key:
            raise QualificationInventoryError(
                f"duplicate stage/host-pair receipt for {edge.key}: "
                f"{edges_by_key[edge.key].path} and {path}"
            )
        edges_by_key[edge.key] = edge
        retained_receipts[edge.key] = retained_receipt
        for endpoint, binding in bindings.items():
            previous = endpoint_bindings.get(endpoint)
            if previous is not None and previous != binding:
                raise QualificationInventoryError(
                    "inconsistent per-(host,stage) dataset/environment/source/"
                    f"host-diagnostic binding for {endpoint}"
                )
            endpoint_bindings[endpoint] = binding
        for endpoint, dataset in retained_datasets.items():
            previous_dataset = endpoint_datasets.get(endpoint)
            if previous_dataset is not None and previous_dataset != dataset:
                raise QualificationInventoryError(
                    "inconsistent per-(host,stage) retained dataset evidence "
                    f"(including its canonical path binding) for {endpoint}"
                )
            endpoint_datasets[endpoint] = dataset

    observed_keys = set(edges_by_key)
    if observed_keys != expected_keys:
        raise QualificationInventoryError(
            "qualification graph is incomplete or contains extras; "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}"
        )
    expected_endpoints = {
        (host, stage) for host in hosts for stage in REQUIRED_STAGES
    }
    if set(endpoint_bindings) != expected_endpoints:
        raise QualificationInventoryError(
            "qualification endpoint inventory is incomplete or contains extras"
        )

    structural_counts_by_stage: dict[str, tuple[int, int, int, int, int]] = {}
    for edge in edges_by_key.values():
        previous = structural_counts_by_stage.get(edge.stage)
        if (
            previous is not None
            and previous != edge.structural_comparison_counts
        ):
            raise QualificationInventoryError(
                f"{edge.stage}: structural comparison cardinalities differ "
                "between host-pair receipts"
            )
        structural_counts_by_stage[edge.stage] = (
            edge.structural_comparison_counts
        )

    fingerprints_by_stage: dict[str, set[str]] = {
        stage: set() for stage in REQUIRED_STAGES
    }
    for (_host, stage), binding in endpoint_bindings.items():
        fingerprints_by_stage[stage].add(binding.gen_fingerprint)
    inconsistent_fingerprints = {
        stage: len(values)
        for stage, values in fingerprints_by_stage.items()
        if len(values) != 1
    }
    if inconsistent_fingerprints:
        raise QualificationInventoryError(
            "each mandatory stage must bind exactly one generation "
            f"fingerprint across hosts: {inconsistent_fingerprints}"
        )
    stage_fingerprints = {
        next(iter(values)) for values in fingerprints_by_stage.values()
    }
    if len(stage_fingerprints) != len(REQUIRED_STAGES):
        raise QualificationInventoryError(
            "F40-S, F40-M, L99-S and L99-M must bind four distinct "
            "stage-specific generation fingerprints"
        )

    hardware_by_host: dict[str, HostHardwareBinding] = {}
    for (host, _stage), binding in sorted(endpoint_bindings.items()):
        hardware = HostHardwareBinding(
            actual_matlab_environment_descriptor=(
                binding.actual_matlab_environment_descriptor
            ),
            actual_matlab_environment_sha256=(
                binding.actual_matlab_environment_sha256
            ),
            matlab_release=binding.matlab_release,
            qualification_hostname=binding.qualification_hostname,
            qualification_cpu_identifier=(
                binding.qualification_cpu_identifier
            ),
            qualification_logical_processors=(
                binding.qualification_logical_processors
            ),
            qualification_matlab_max_threads=(
                binding.qualification_matlab_max_threads
            ),
            qualification_computer_arch=(
                binding.qualification_computer_arch
            ),
        )
        previous = hardware_by_host.get(host)
        if previous is not None and previous != hardware:
            raise QualificationInventoryError(
                f"intended host ID {host!r} does not bind one stable "
                "set of self-attested host diagnostics across all mandatory "
                "stages"
            )
        hardware_by_host[host] = hardware

    # Graph/schema checks alone cannot authenticate a receipt's numerical
    # claims.  Rerun the authoritative pairwise comparator for EVERY accepted
    # edge, using the payload parsed from the exact canonical receipt bytes
    # already hashed above.  This subsumes endpoint-only reopening and also
    # proves the retained verdict, all comparison statistics and raw-identity
    # count.  No public inventory path can skip this loop.
    for key in sorted(edges_by_key):
        edge = edges_by_key[key]
        endpoint_revalidation.revalidate_edge_comparison(
            retained_receipts[key],
            owner=str(edge.path),
        )
    try:
        for index, snapshot in enumerate(receipt_snapshots, 1):
            path_safety.assert_snapshot_unchanged(
                snapshot, f"qualification pair receipt {index}")
    except path_safety.QualificationPathError as exc:
        raise QualificationInventoryError(str(exc)) from exc
    final_policy = current_policy()
    if final_policy != policy:
        raise QualificationInventoryError(
            "current qualification policy changed during inventory validation"
        )

    edge_lines = [
        "|".join((*edge.key, edge.receipt_sha256))
        for edge in sorted(edges_by_key.values(), key=lambda item: item.key)
    ]
    endpoint_lines = []
    for endpoint in sorted(endpoint_bindings):
        binding = endpoint_bindings[endpoint]
        endpoint_lines.append(
            "|".join(
                (
                    endpoint[0],
                    endpoint[1],
                    binding.dataset_content_root_sha256,
                    binding.actual_matlab_environment_sha256,
                    binding.generator_source_root_sha256,
                    binding.qualification_source_sha256,
                    binding.qualification_executed_file_sha256,
                    binding.qualification_host_diagnostic_sha256,
                    binding.mechanism_coverage_sha256,
                )
            )
        )
    root_material = "\n".join(
        (
            f"schema={INVENTORY_SCHEMA}",
            f"comparator_sha256={policy.comparator_sha256}",
            f"checker_sha256={policy.checker_sha256}",
            f"environment_lock_sha256={policy.environment_lock_sha256}",
            *edge_lines,
            *endpoint_lines,
        )
    )
    return InventoryResult(
        intended_host_ids=hosts,
        edges=tuple(
            sorted(edges_by_key.values(), key=lambda item: item.key)
        ),
        endpoint_bindings=endpoint_bindings,
        inventory_root_sha256=_sha256_text(root_material),
        policy=policy,
    )


def inventory_receipt_payload(result: InventoryResult) -> dict[str, Any]:
    """Return a safe operational receipt containing identities, not results."""
    endpoint_rows = []
    for host, stage in sorted(result.endpoint_bindings):
        binding = result.endpoint_bindings[(host, stage)]
        endpoint_rows.append(
            {
                "host_id": host,
                "stage": stage,
                "dataset_content_root_sha256": (
                    binding.dataset_content_root_sha256
                ),
                "actual_matlab_environment_sha256": (
                    binding.actual_matlab_environment_sha256
                ),
                "qualification_executed_file_sha256": (
                    binding.qualification_executed_file_sha256
                ),
                "qualification_host_diagnostic_sha256": (
                    binding.qualification_host_diagnostic_sha256
                ),
                "mechanism_coverage_sha256": (
                    binding.mechanism_coverage_sha256
                ),
            }
        )
    return {
        "schema": INVENTORY_SCHEMA,
        "inventory_checker_sha256": result.policy.checker_sha256,
        "comparator_sha256": result.policy.comparator_sha256,
        "environment_lock_sha256": result.policy.environment_lock_sha256,
        "parser_environment": result.policy.parser_environment,
        "python_runtime_source_root_sha256": (
            result.policy.python_runtime_source_root_sha256
        ),
        "python_runtime_source_file_count": (
            result.policy.python_runtime_source_file_count
        ),
        "generator_source_root_sha256": (
            result.policy.generator_source_root_sha256
        ),
        "required_stages": list(REQUIRED_STAGES),
        "intended_host_ids": list(result.intended_host_ids),
        "expected_pairwise_receipt_count": len(result.edges),
        "accepted_pairwise_receipt_count": len(result.edges),
        "receipt_edges": [
            {
                "stage": edge.stage,
                "host_ids": [edge.host_a, edge.host_b],
                "receipt_sha256": edge.receipt_sha256,
            }
            for edge in result.edges
        ],
        "endpoint_bindings": endpoint_rows,
        "inventory_root_sha256": result.inventory_root_sha256,
    }


def _revalidate_inventory_result(
    result: InventoryResult,
    *,
    owner: str,
) -> InventoryResult:
    """Rebuild an InventoryResult from its retained edge paths.

    ``InventoryResult`` is a convenient in-process value, not an authority
    token.  A caller can retain it while policy, pair receipts, or endpoint
    trees change.  Publication therefore reruns the complete graph and accepts
    the object only if the canonical recomputation is field-for-field equal.
    """
    if not isinstance(result, InventoryResult):
        raise QualificationInventoryError(
            f"{owner}: expected one InventoryResult"
        )
    paths = tuple(edge.path for edge in result.edges)
    recomputed = validate_inventory(result.intended_host_ids, paths)
    if recomputed != result:
        raise QualificationInventoryError(
            f"{owner}: retained InventoryResult is stale or differs from the "
            "complete policy/edge/endpoint recomputation"
        )
    return recomputed


def write_inventory_receipt(path: Path | str, result: InventoryResult) -> None:
    try:
        target = path_safety.canonical_new_file_path(
            path, "qualification inventory receipt")
    except path_safety.QualificationPathError as exc:
        raise QualificationInventoryError(str(exc)) from exc
    current = _revalidate_inventory_result(
        result, owner="pre-publication qualification inventory"
    )
    encoded = _canonical_json_bytes(inventory_receipt_payload(current))
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise QualificationInventoryError(
            f"refusing to overwrite existing inventory receipt: {target}"
        ) from exc
    except OSError as exc:
        raise QualificationInventoryError(
            "could not persist the create-once inventory receipt; any "
            "created path was retained for forensic review"
        ) from exc
    try:
        persisted = path_safety.snapshot_receipt(
            target, "published qualification inventory receipt"
        )
        if persisted.raw != encoded:
            raise QualificationInventoryError(
                "inventory receipt did not persist as the exact regular "
                f"file: {target}"
            )
        after = _revalidate_inventory_result(
            current, owner="post-publication qualification inventory"
        )
        if after != current:
            raise QualificationInventoryError(
                "qualification inventory changed across publication"
            )
        path_safety.assert_snapshot_unchanged(
            persisted, "published qualification inventory receipt"
        )
    except (
        QualificationInventoryError,
        path_safety.QualificationPathError,
        OSError,
    ) as exc:
        # Never unlink through a mutable pathname after publication.  The
        # create-once bytes remain as forensic evidence; this call raises and
        # therefore returns no successful authorization.
        raise QualificationInventoryError(
            "post-publication inventory validation failed; the create-once "
            f"receipt was retained for forensic review: {exc}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        action="append",
        required=True,
        help="stable intended TTBI_QUALIFICATION_HOST_ID; repeat for every host",
    )
    parser.add_argument(
        "--receipt",
        action="append",
        default=[],
        type=Path,
        help="accepted v4 receipt file; repeat as needed",
    )
    parser.add_argument(
        "--receipt-dir",
        action="append",
        default=[],
        type=Path,
        help="directory whose immediate *.json children are all v4 receipts",
    )
    parser.add_argument(
        "--inventory-receipt",
        type=Path,
        help="optional new exclusive-write path for the safe inventory receipt",
    )
    args = parser.parse_args(argv)
    try:
        paths = collect_receipt_paths(
            receipt_files=args.receipt,
            receipt_dirs=args.receipt_dir,
        )
        result = validate_inventory(args.host, paths)
        if args.inventory_receipt is not None:
            write_inventory_receipt(args.inventory_receipt, result)
        print("MATLAB QUALIFICATION RECEIPT INVENTORY: PASS")
        print(f"  intended hosts: {len(result.intended_host_ids)}")
        print(f"  mandatory stages: {', '.join(REQUIRED_STAGES)}")
        print(f"  accepted pairwise receipts: {len(result.edges)}")
        print(f"  inventory root SHA-256: {result.inventory_root_sha256}")
        if args.inventory_receipt is not None:
            print(f"  inventory receipt: {args.inventory_receipt}")
        return 0
    except QualificationInventoryError as exc:
        print(f"INVALID QUALIFICATION RECEIPT INVENTORY: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

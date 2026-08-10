"""Synthetic/adversarial checks for qualification_receipt_inventory.py.

Fixtures are REAL comparator-genuine datasets: every (host, stage) endpoint is
one on-disk MICRO dataset written by check_generation_release_comparison's
proven builders, and every pair receipt is emitted by actually running
``compare_generation_releases.compare_directories`` on those directories, so
retained evidence blocks are exactly what the production comparator produces.

The mutation suite therefore proves two distinct properties fail closed:

* receipt-level forgeries (the retained mutation suite, applied to copies of
  the genuine receipts) are rejected by the schema/graph layers; and
* endpoint-level attacks - missing, stale, moved, coherently forged,
  digest-table-laundered, symlinked, and TOCTOU-mutated retained dataset
  directories - are rejected by the production from-disk revalidators.

Every retained baseline receipt is emitted by one real full comparison.  The
checker then reuses that authenticated ComparisonResult only for the exact
same ordered paths and tolerances; unknown/rebound endpoint pairs still run
the full comparator.  Schema/graph mutations additionally forbid any
comparator call, proving that they fail before expensive disk comparison.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import inspect
import itertools
import json
import os
from dataclasses import replace
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable

from scipy.io import savemat

import compare_generation_releases as comparator
import dispatch_authorization as dispatch
import qualification_endpoint_revalidation as revalidation
import qualification_path_safety as path_safety
import qualification_receipt_inventory as inventory
import qualification_receipt_schema as schema
from check_generation_release_comparison import (
    _campaign_environment,
    _write_dataset,
)


HOSTS3 = ("host-a", "host-b", "host-c")
HOSTS2 = ("host-a", "host-b")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(inventory._canonical_json_bytes(payload))


def _rewrite(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write(path, payload)


def _numerical_perturbation(data: dict[str, Any]) -> None:
    # In-tolerance solver-output drift: 5e-11 on a ~1.0 sample stays inside
    # the reviewed rtol=1e-10 policy, producing NUMERICALLY-EQUIVALENT.
    data["AcelPrimVag"][0][0, 1] += 5e-11


def _build_datasets(
    root: Path,
    hosts: tuple[str, ...],
    *,
    perturb_hosts: tuple[str, ...] = (),
) -> dict[tuple[str, str], Path]:
    # The proven fixture writer intentionally asks the comparator for current
    # policy on every invocation.  That fail-closed behavior is exercised in
    # the comparator suite; here all host/stage datasets belong to one
    # immutable fixture build.  Reuse one authenticated policy snapshot so the
    # 16 full-size fixture writes do not each rehash the entire source tree and
    # push this checker beyond the R4 subprocess ceiling.
    current_policy = comparator._current_policy()
    original_current_policy = comparator._current_policy
    comparator._current_policy = lambda: current_policy
    try:
        campaign = _campaign_environment()
        datasets: dict[tuple[str, str], Path] = {}
        for host in hosts:
            for stage in inventory.REQUIRED_STAGES:
                directory = root / f"{host}__{stage}"
                _write_dataset(
                    directory,
                    release=campaign.release,
                    stage=stage,
                    host_id=host,
                    mutate_data=(
                        _numerical_perturbation
                        if host in perturb_hosts
                        else None
                    ),
                )
                datasets[(host, stage)] = directory
        return datasets
    finally:
        comparator._current_policy = original_current_policy


def _emit_edge_receipt(
    path: Path,
    dir_a: Path,
    dir_b: Path,
    *,
    accept: bool,
    expected_verdict: str,
) -> None:
    result = comparator.compare_directories(dir_a, dir_b)
    if result.verdict != expected_verdict:
        raise AssertionError(
            f"fixture pair {dir_a.name}/{dir_b.name} produced "
            f"{result.verdict!r}, expected {expected_verdict!r}: "
            f"{result.stats.mismatches}"
        )
    # The equal-roots forgery case below relies on genuine receipts binding
    # two distinct content roots; fail loudly if the fixture ever changes.
    assert (
        result.evidence_a.dataset_content_root_sha256
        != result.evidence_b.dataset_content_root_sha256
    )
    _write(path, comparator._receipt_payload(result, accepted=accept))


def _emit_graph_receipts(
    receipt_dir: Path,
    datasets: dict[tuple[str, str], Path],
    hosts: tuple[str, ...],
    *,
    accept: bool = False,
    expected_verdict: str = "SEMANTICALLY-BIT-IDENTICAL",
) -> list[Path]:
    receipt_dir.mkdir()
    paths: list[Path] = []
    for stage in inventory.REQUIRED_STAGES:
        for host_a, host_b in itertools.combinations(hosts, 2):
            path = receipt_dir / f"{stage}__{host_a}__{host_b}.json"
            _emit_edge_receipt(
                path,
                datasets[(host_a, stage)],
                datasets[(host_b, stage)],
                accept=accept,
                expected_verdict=expected_verdict,
            )
            paths.append(path)
    return paths


def _copy_receipts(source_paths: list[Path], case_dir: Path) -> list[Path]:
    case_dir.mkdir(parents=True)
    copies: list[Path] = []
    for source in source_paths:
        target = case_dir / source.name
        target.write_bytes(source.read_bytes())
        copies.append(target)
    return copies


def _expect_invalid(
    name: str,
    hosts: tuple[str, ...],
    paths: list[Path],
) -> None:
    try:
        inventory.validate_inventory(hosts, paths)
    except inventory.QualificationInventoryError:
        print(f"  [PASS] rejected {name}")
        return
    raise AssertionError(f"invalid qualification inventory escaped: {name}")


def _expect_invalid_before_disk_comparison(
    name: str,
    hosts: tuple[str, ...],
    paths: list[Path],
) -> None:
    """Prove a receipt grammar/graph attack fails before recomputation."""

    def forbidden_disk_comparison(
        *_args: object,
        **_kwargs: object,
    ) -> comparator.ComparisonResult:
        raise AssertionError(
            f"schema/graph mutation {name!r} reached the disk comparator"
        )

    active_comparator = comparator.compare_directories
    comparator.compare_directories = forbidden_disk_comparison
    try:
        _expect_invalid(name, hosts, paths)
    finally:
        comparator.compare_directories = active_comparator


def _expect_invalid_message(
    name: str,
    hosts: tuple[str, ...],
    paths: list[Path],
    *needles: str,
) -> None:
    try:
        inventory.validate_inventory(hosts, paths)
    except inventory.QualificationInventoryError as exc:
        message = str(exc)
        for needle in needles:
            if needle not in message:
                raise AssertionError(
                    f"{name}: rejection message lacks {needle!r}: {message}"
                ) from exc
        print(f"  [PASS] rejected {name}")
        return
    raise AssertionError(f"invalid qualification inventory escaped: {name}")


def _expect_dispatch_invalid_message(
    name: str,
    action: Callable[[], object],
    *needles: str,
) -> None:
    try:
        action()
    except dispatch.DispatchAuthorizationError as exc:
        message = str(exc)
        for needle in needles:
            if needle not in message:
                raise AssertionError(
                    f"{name}: dispatch rejection lacks {needle!r}: {message}"
                ) from exc
        print(f"  [PASS] dispatch rejected {name}")
        return
    raise AssertionError(f"invalid qualification evidence escaped dispatch: {name}")


def _rebuild_descriptor(dataset: dict[str, Any]) -> tuple[str, str]:
    """Rebuild the canonical host descriptor from a dataset block's own
    fields, so coherent forgeries stay internally consistent one level deep
    and only the intended cross-receipt/recomputation guard can reject them.
    """
    lines: list[str] = []
    for label, source in schema._HOST_DESCRIPTOR_FIELDS:
        value = source if label == "schema" else dataset[source]
        lines.append(f"{label}={value}")
    descriptor = "\n".join(lines)
    return descriptor, _sha(descriptor)


def _restamp_host_diagnostic(
    payload: dict[str, Any],
    index: int,
    field: str,
) -> None:
    dataset = payload[field]
    descriptor, diagnostic = _rebuild_descriptor(dataset)
    dataset["qualification_host_canonical_descriptor"] = descriptor
    dataset["qualification_host_diagnostic_sha256"] = diagnostic
    payload["qualification_host_diagnostic_sha256"][index] = diagnostic


def _rebind_endpoint_path(
    paths: list[Path],
    *,
    host: str,
    stage: str,
    new_path: str,
) -> None:
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["stage"] != stage:
            continue
        changed = False
        for field in ("dataset_a", "dataset_b"):
            if payload[field]["qualification_host_id"] == host:
                payload[field]["path"] = new_path
                changed = True
        if changed:
            _write(path, payload)


def _regenerate_digest_table(dataset_dir: Path) -> None:
    """Recompute a fully coherent file_digests.mat + completion marker for the
    directory's CURRENT bytes - the digest-laundering attack a local adversary
    would use after mutating retained files."""
    names = sorted(
        item.name
        for item in dataset_dir.iterdir()
        if re.fullmatch(r"\d{4}\.mat", item.name)
    ) + ["case_info.mat", "damage_states.mat"]
    per_file = {name: _file_sha(dataset_dir / name) for name in names}
    digest_lines = "\n".join(
        f"{name}:{per_file[name]}" for name in sorted(per_file)
    )
    dataset_root = hashlib.sha256(digest_lines.encode("utf-8")).hexdigest()
    savemat(
        dataset_dir / "file_digests.mat",
        {
            "file_digests": {
                "schema": "source-digests-v2",
                "scope": "NNNN.mat+case_info.mat+damage_states.mat",
                "digest_lines": digest_lines,
                "root": dataset_root,
            }
        },
        long_field_names=True,
    )
    marker = dataset_dir / "_GENERATION_COMPLETE"
    lines = marker.read_bytes().decode("utf-8").splitlines()
    lines[2] = dataset_root
    marker.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def _flip_byte(path: Path, offset: int) -> None:
    raw = bytearray(path.read_bytes())
    raw[offset] ^= 0x01
    path.write_bytes(bytes(raw))


def _assert_safe_inventory_payload(
    payload: dict[str, Any],
    result: inventory.InventoryResult,
) -> None:
    expected_top_fields = {
        "schema",
        "inventory_checker_sha256",
        "comparator_sha256",
        "environment_lock_sha256",
        "parser_environment",
        "python_runtime_source_root_sha256",
        "python_runtime_source_file_count",
        "generator_source_root_sha256",
        "required_stages",
        "intended_host_ids",
        "expected_pairwise_receipt_count",
        "accepted_pairwise_receipt_count",
        "receipt_edges",
        "endpoint_bindings",
        "inventory_root_sha256",
    }
    assert set(payload) == expected_top_fields
    assert payload["schema"] == inventory.INVENTORY_SCHEMA
    assert payload["inventory_checker_sha256"] == result.policy.checker_sha256
    assert payload["comparator_sha256"] == result.policy.comparator_sha256
    assert (
        payload["environment_lock_sha256"]
        == result.policy.environment_lock_sha256
    )
    assert payload["parser_environment"] == result.policy.parser_environment
    assert (
        payload["python_runtime_source_root_sha256"]
        == result.policy.python_runtime_source_root_sha256
    )
    assert (
        payload["python_runtime_source_file_count"]
        == result.policy.python_runtime_source_file_count
    )
    assert (
        payload["generator_source_root_sha256"]
        == result.policy.generator_source_root_sha256
    )
    assert payload["required_stages"] == list(inventory.REQUIRED_STAGES)
    assert payload["intended_host_ids"] == list(result.intended_host_ids)
    expected_edges = (
        len(inventory.REQUIRED_STAGES)
        * len(result.intended_host_ids)
        * (len(result.intended_host_ids) - 1)
        // 2
    )
    assert len(result.edges) == expected_edges
    assert payload["expected_pairwise_receipt_count"] == expected_edges
    assert payload["accepted_pairwise_receipt_count"] == expected_edges
    assert payload["inventory_root_sha256"] == result.inventory_root_sha256

    edge_rows = payload["receipt_edges"]
    assert len(edge_rows) == expected_edges
    assert all(
        set(row) == {"stage", "host_ids", "receipt_sha256"}
        for row in edge_rows
    )
    assert edge_rows == [
        {
            "stage": edge.stage,
            "host_ids": [edge.host_a, edge.host_b],
            "receipt_sha256": edge.receipt_sha256,
        }
        for edge in result.edges
    ]

    endpoint_rows = payload["endpoint_bindings"]
    assert len(endpoint_rows) == (
        len(inventory.REQUIRED_STAGES) * len(result.intended_host_ids)
    )
    expected_endpoint_fields = {
        "host_id",
        "stage",
        "dataset_content_root_sha256",
        "actual_matlab_environment_sha256",
        "qualification_executed_file_sha256",
        "qualification_host_diagnostic_sha256",
        "mechanism_coverage_sha256",
    }
    assert all(set(row) == expected_endpoint_fields for row in endpoint_rows)
    expected_endpoint_rows = []
    for host, stage in sorted(result.endpoint_bindings):
        binding = result.endpoint_bindings[(host, stage)]
        expected_endpoint_rows.append(
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
    assert endpoint_rows == expected_endpoint_rows


def _expect_payload_assertion(
    name: str,
    payload: dict[str, Any],
    result: inventory.InventoryResult,
) -> None:
    try:
        _assert_safe_inventory_payload(payload, result)
    except AssertionError:
        print(f"  [PASS] payload assertion rejected {name}")
        return
    raise AssertionError(f"invalid safe inventory payload escaped: {name}")


def _rewrite_foreign_endpoint(path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        dataset = payload["dataset_b"]
        descriptor = "foreign-production-environment"
        actual_sha = _sha(descriptor)
        dataset["actual_matlab_environment_descriptor"] = descriptor
        dataset["actual_matlab_environment_sha256"] = actual_sha
        payload["actual_matlab_environment_sha256"][1] = actual_sha
        _restamp_host_diagnostic(payload, 1, "dataset_b")

    _rewrite(path, mutate)


def _rewrite_cross_stage_host(paths: list[Path]) -> None:
    # Mutate host-a consistently on every incident F40-M edge so the per-stage
    # graph remains coherent while the cross-stage physical-host binding does
    # not.
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["stage"] != "F40-M":
            continue
        changed = False
        for index, field in enumerate(("dataset_a", "dataset_b")):
            dataset = payload[field]
            if dataset["qualification_host_id"] != "host-a":
                continue
            dataset["qualification_hostname"] = "replacement-host.example"
            _restamp_host_diagnostic(payload, index, field)
            changed = True
        if changed:
            _write(path, payload)


def _rewrite_stage_fingerprint(paths: list[Path]) -> None:
    # Change every endpoint copy for one stage so the graph remains internally
    # consistent; only the missing SHA-256 validation can catch this mutation.
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["stage"] != "F40-S":
            continue
        payload["dataset_a"]["gen_fingerprint"] = "not-a-sha"
        payload["dataset_b"]["gen_fingerprint"] = "not-a-sha"
        _write(path, payload)


def _rewrite_leaf_class_drift(path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["comparison"]["exact_leaves"] -= 1
        payload["comparison"]["tolerant_leaves"] += 1

    _rewrite(path, mutate)


def _rewrite_cross_stage_fingerprint(paths: list[Path]) -> None:
    f40s_payload = json.loads(paths[0].read_text(encoding="utf-8"))
    reused = f40s_payload["dataset_a"]["gen_fingerprint"]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["stage"] != "F40-M":
            continue
        payload["dataset_a"]["gen_fingerprint"] = reused
        payload["dataset_b"]["gen_fingerprint"] = reused
        _write(path, payload)


def _rewrite_host_identity(paths: list[Path], *, hostname: str) -> None:
    # Apply the coherent forged identity to host-a in every stage and edge,
    # preserving both incident-edge and cross-stage consistency.
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for index, field in enumerate(("dataset_a", "dataset_b")):
            dataset = payload[field]
            if dataset["qualification_host_id"] != "host-a":
                continue
            dataset["qualification_hostname"] = hostname
            _restamp_host_diagnostic(payload, index, field)
            changed = True
        if changed:
            _write(path, payload)


def _rewrite_pair_coverage(path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        for field in ("dataset_a", "dataset_b"):
            payload[field]["mechanism_coverage"][
                "crack_active_passages"
            ] += 1

    _rewrite(path, mutate)


def _rewrite_pair_channel_schema(path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        for field in ("dataset_a", "dataset_b"):
            payload[field]["channel_schema_id"] = "legacy_virtual8"

    _rewrite(path, mutate)


def _rewrite_stage_executed_sha(paths: list[Path]) -> None:
    # Rebuild every host descriptor and every duplicated endpoint binding for
    # one stage. Graph consistency alone cannot identify this coherent forgery;
    # only recomputing the canonical qualification executable can reject it.
    forged_sha = "0" * 64
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["stage"] != "F40-S":
            continue
        for index, field in enumerate(("dataset_a", "dataset_b")):
            payload[field]["qualification_executed_file_sha256"] = forged_sha
            _restamp_host_diagnostic(payload, index, field)
        _write(path, payload)


def _rewrite_incident_endpoint(paths: list[Path]) -> None:
    # The first two F40-S edges both contain host-a.
    def mutate(payload: dict[str, Any]) -> None:
        dataset = payload["dataset_a"]
        dataset["dataset_content_root_sha256"] = _sha("inconsistent-copy")
        payload["dataset_content_roots_sha256"][0] = (
            dataset["dataset_content_root_sha256"]
        )

    _rewrite(paths[1], mutate)


def main() -> int:
    print("QUALIFICATION RECEIPT INVENTORY ADVERSARIAL CHECKS")

    # The revalidators expose no skip/trust parameter surface.
    assert tuple(
        inspect.signature(revalidation.revalidate_endpoint).parameters
    ) == ("path_text", "retained_dataset_block", "policy", "owner")
    assert tuple(
        inspect.signature(revalidation.revalidate_edge_comparison).parameters
    ) == ("retained_receipt_payload", "owner")
    print("  [PASS] endpoint revalidators expose no skip/trust parameters")

    with tempfile.TemporaryDirectory(
        prefix="ttbi-qualification-inventory-"
    ) as tmp, ExitStack() as cleanup:
        root = Path(tmp).resolve(strict=True)
        policy = inventory.current_policy()
        assert policy.channel_schema_id == "physical8_v1"
        print("  [PASS] inventory policy pins physical8_v1 channel semantics")

        # Authenticate the complete comparator policy/source snapshot once for
        # this source-quiet subprocess. Production drift behavior is tested
        # explicitly below; all heavy fixture comparisons can therefore bind
        # to this exact immutable object without rehashing the repository on
        # every endpoint and edge.
        full_current_comparator_policy = comparator._current_policy
        full_policy_source_assertion = (
            comparator._assert_policy_sources_unchanged
        )
        authenticated_comparator_policy = full_current_comparator_policy()
        full_policy_source_assertion(
            authenticated_comparator_policy,
            "qualification inventory fixture bootstrap",
        )

        def authenticated_current_comparator_policy():
            return authenticated_comparator_policy

        def assert_authenticated_policy_snapshot(
            candidate: comparator.CurrentPolicy,
            owner: str,
        ) -> None:
            if candidate != authenticated_comparator_policy:
                raise comparator.QualificationInputError(
                    f"{owner}: comparator policy differs from the "
                    "authenticated fixture snapshot"
                )

        comparator._current_policy = authenticated_current_comparator_policy
        comparator._assert_policy_sources_unchanged = (
            assert_authenticated_policy_snapshot
        )
        cleanup.callback(
            setattr,
            comparator,
            "_assert_policy_sources_unchanged",
            full_policy_source_assertion,
        )
        cleanup.callback(
            setattr,
            comparator,
            "_current_policy",
            full_current_comparator_policy,
        )

        # Receipt construction below still executes one genuine full
        # comparison for every unique ordered endpoint pair/tolerance.  Later
        # validations of those unchanged endpoints reuse the exact immutable
        # result; a new or rebound path necessarily falls through to the real
        # comparator.  This preserves the endpoint-attack controls while
        # avoiding dozens of identical full-tree comparisons in one checker.
        authenticated_comparisons: dict[
            tuple[str, str, float, float], comparator.ComparisonResult
        ] = {}
        full_compare_directories = comparator.compare_directories

        def compare_once_per_endpoint_pair(
            dir_a: Path | str,
            dir_b: Path | str,
            *,
            rtol: float = comparator._DEFAULT_RTOL,
            atol: float = comparator._DEFAULT_ATOL,
        ) -> comparator.ComparisonResult:
            key = (
                str(Path(dir_a).resolve(strict=True)),
                str(Path(dir_b).resolve(strict=True)),
                float(rtol),
                float(atol),
            )
            cached = authenticated_comparisons.get(key)
            if cached is not None:
                return cached
            observed = full_compare_directories(
                dir_a,
                dir_b,
                rtol=rtol,
                atol=atol,
            )
            authenticated_comparisons[key] = observed
            return observed

        comparator.compare_directories = compare_once_per_endpoint_pair
        cleanup.callback(
            setattr,
            comparator,
            "compare_directories",
            full_compare_directories,
        )

        path_fixture = root / "receipt-path-safety"
        path_fixture.mkdir()
        canonical_receipt = path_fixture / "receipt.json"
        canonical_receipt.write_bytes(b"canonical\n")
        assert inventory.collect_receipt_paths(
            receipt_files=(canonical_receipt,)
        ) == (canonical_receipt,)
        receipt_snapshot = path_safety.snapshot_receipt(
            canonical_receipt, "path-safety fixture")
        canonical_receipt.write_bytes(b"changed!!\n")
        try:
            path_safety.assert_snapshot_unchanged(
                receipt_snapshot, "path-safety fixture")
        except path_safety.QualificationPathError:
            pass
        else:
            raise AssertionError("receipt TOCTOU mutation was accepted")
        canonical_receipt.write_bytes(b"canonical\n")
        try:
            inventory.collect_receipt_paths(
                receipt_files=(Path("receipt.json"),))
        except inventory.QualificationInventoryError:
            pass
        else:
            raise AssertionError("relative receipt path was accepted")
        hardlink = path_fixture / "receipt-hardlink.json"
        try:
            os.link(canonical_receipt, hardlink)
        except OSError:
            print(
                "  [N/A] standalone hard-link receipt path "
                "(filesystem unavailable)"
            )
        else:
            try:
                inventory.collect_receipt_paths(
                    receipt_files=(canonical_receipt,))
            except inventory.QualificationInventoryError:
                pass
            else:
                raise AssertionError("hard-linked receipt path was accepted")
            hardlink.unlink()
        print(
            "  [PASS] standalone receipt paths are canonical, single-link "
            "and TOCTOU-bound"
        )

        # One shared set of REAL datasets per host set; receipts are emitted
        # by genuinely running the pairwise comparator per edge.
        datasets3 = _build_datasets(root / "datasets", HOSTS3)
        baseline_dir = root / "baseline"
        baseline_paths = _emit_graph_receipts(baseline_dir, datasets3, HOSTS3)

        receipt_snapshot = path_safety.snapshot_receipt(
            baseline_paths[0], "receipt byte-binding fixture")
        baseline_paths[0].write_bytes(b"foreign pathname bytes\n")
        try:
            bound_edge, _bindings, _datasets, _payload = (
                schema._load_receipt(
                    baseline_paths[0],
                    intended_hosts=frozenset(HOSTS3),
                    policy=policy,
                    receipt_bytes=receipt_snapshot.raw,
                )
            )
            assert bound_edge.receipt_sha256 == receipt_snapshot.sha256
            try:
                schema._load_receipt(
                    baseline_paths[0],
                    intended_hosts=frozenset(HOSTS3),
                    policy=policy,
                )
            except inventory.QualificationInventoryError:
                pass
            else:
                raise AssertionError(
                    "pathname re-read accepted foreign receipt bytes")
        finally:
            baseline_paths[0].write_bytes(receipt_snapshot.raw)
        print(
            "  [PASS] receipt grammar/comparison consumes the exact "
            "authenticated byte snapshot"
        )

        result = inventory.validate_inventory(
            ("host-c", "host-a", "host-b"), baseline_paths
        )
        assert result.intended_host_ids == ("host-a", "host-b", "host-c")
        assert len(result.edges) == 12
        assert len(result.endpoint_bindings) == 12
        assert len(result.inventory_root_sha256) == 64
        print(
            "  [PASS] complete 4-stage C(3,2) graph of real datasets accepted"
        )

        two_hosts = root / "two-hosts"
        two_paths = _copy_receipts(
            [
                path
                for path in baseline_paths
                if path.name.endswith("__host-a__host-b.json")
            ],
            two_hosts,
        )
        two_result = inventory.validate_inventory(HOSTS2, two_paths)
        assert len(two_result.edges) == 4
        print("  [PASS] minimum H=2 graph accepted")

        retained_policy_payload = json.loads(
            two_paths[0].read_text(encoding="utf-8")
        )
        cached_comparison = comparator.compare_directories(
            Path(retained_policy_payload["dataset_a"]["path"]),
            Path(retained_policy_payload["dataset_b"]["path"]),
            rtol=retained_policy_payload["tolerances"]["rtol"],
            atol=retained_policy_payload["tolerances"]["atol"],
        )
        original_revalidation_compare = (
            revalidation.comparator.compare_directories
        )
        revalidation.comparator.compare_directories = (
            lambda *_args, **_kwargs: cached_comparison
        )
        try:
            policy_receipt_mutations = (
                ("environment_lock_sha256", "0" * 64),
                ("parser_environment", {"python": "forged"}),
                ("python_runtime_source_root_sha256", "0" * 64),
                (
                    "python_runtime_source_file_count",
                    cached_comparison.python_runtime_source_file_count + 1,
                ),
            )
            for field, value in policy_receipt_mutations:
                forged = json.loads(json.dumps(retained_policy_payload))
                forged[field] = value
                try:
                    revalidation.revalidate_edge_comparison(
                        forged, owner=f"direct-policy-{field}"
                    )
                except inventory.QualificationInventoryError:
                    pass
                else:
                    raise AssertionError(
                        f"direct edge revalidation accepted stale {field}"
                    )
            forged = json.loads(json.dumps(retained_policy_payload))
            forged["tolerances"]["rtol"] = 0.0
            try:
                revalidation.revalidate_edge_comparison(
                    forged, owner="direct-policy-tolerance"
                )
            except inventory.QualificationInventoryError:
                pass
            else:
                raise AssertionError(
                    "direct edge revalidation accepted a tolerance/result "
                    "binding mismatch"
                )
        finally:
            revalidation.comparator.compare_directories = (
                original_revalidation_compare
            )
        print(
            "  [PASS] direct edge revalidation binds environment/parser/"
            "Python provenance and exact ComparisonResult tolerances"
        )

        original_endpoint_policy = revalidation.current_comparator_policy
        original_revalidation_compare = (
            revalidation.comparator.compare_directories
        )
        endpoint_policy_calls = {"count": 0}

        def drifting_endpoint_policy():
            observed = original_endpoint_policy()
            endpoint_policy_calls["count"] += 1
            if endpoint_policy_calls["count"] >= 2:
                return replace(
                    observed, generator_source_root_sha256="0" * 64
                )
            return observed

        revalidation.current_comparator_policy = drifting_endpoint_policy
        revalidation.comparator.compare_directories = (
            lambda *_args, **_kwargs: cached_comparison
        )
        try:
            try:
                revalidation.revalidate_edge_comparison(
                    retained_policy_payload,
                    owner="direct-endpoint-policy-drift",
                )
            except inventory.QualificationInventoryError as exc:
                if "policy changed during edge revalidation" not in str(exc):
                    raise
            else:
                raise AssertionError(
                    "edge revalidation accepted current-policy drift"
                )
        finally:
            revalidation.current_comparator_policy = original_endpoint_policy
            revalidation.comparator.compare_directories = (
                original_revalidation_compare
            )
        assert endpoint_policy_calls["count"] >= 2
        print("  [PASS] edge revalidation reasserts current policy at the end")

        original_policy_source_reader = (
            comparator._read_single_link_regular_bytes
        )
        policy_source_target = Path(revalidation.__file__).resolve()
        policy_source_reads = {"count": 0}

        def changing_policy_source_reader(path: Path, owner: str) -> bytes:
            payload = original_policy_source_reader(path, owner)
            if Path(path) == policy_source_target:
                policy_source_reads["count"] += 1
                if policy_source_reads["count"] >= 2:
                    return payload + b"\n# policy-source drift probe\n"
            return payload

        comparator._read_single_link_regular_bytes = (
            changing_policy_source_reader
        )
        try:
            try:
                inventory.current_policy()
            except inventory.QualificationInventoryError:
                pass
            else:
                raise AssertionError(
                    "inventory policy accepted checker-source drift between "
                    "hash and return"
                )
        finally:
            comparator._read_single_link_regular_bytes = (
                original_policy_source_reader
            )
        assert policy_source_reads["count"] >= 2
        print("  [PASS] inventory policy reasserts exact checker source bytes")

        original_inventory_policy = inventory.current_policy
        original_inventory_edge_revalidator = (
            inventory.endpoint_revalidation.revalidate_edge_comparison
        )
        inventory_policy_calls = {"count": 0}

        def drifting_inventory_policy():
            observed = original_inventory_policy()
            inventory_policy_calls["count"] += 1
            if inventory_policy_calls["count"] >= 2:
                return replace(observed, checker_sha256="0" * 64)
            return observed

        inventory.current_policy = drifting_inventory_policy
        inventory.endpoint_revalidation.revalidate_edge_comparison = (
            lambda _payload, *, owner: None
        )
        try:
            _expect_invalid_message(
                "inventory current-policy drift",
                HOSTS2,
                two_paths,
                "policy changed during inventory validation",
            )
        finally:
            inventory.current_policy = original_inventory_policy
            inventory.endpoint_revalidation.revalidate_edge_comparison = (
                original_inventory_edge_revalidator
            )
        assert inventory_policy_calls["count"] >= 2

        # The preceding controls exercised both source-byte drift during
        # current_policy() and policy drift across validate_inventory().  The
        # remaining cases all belong to this same source-quiet fixture run, so
        # reuse the already authenticated immutable policy instead of hashing
        # the full repository twice for every schema mutation.
        inventory.current_policy = lambda: policy
        cleanup.callback(
            setattr,
            inventory,
            "current_policy",
            original_inventory_policy,
        )

        # Permanent production-path regression: dispatch consumes genuine
        # comparator-emitted receipts plus the exact safe inventory receipt.
        # The wrapper records calls but still executes the real revalidator.
        # Two calls per edge prove both the autonomous inventory pass and
        # dispatch's explicit defence-in-depth pass remain behavioural, not
        # merely source-token assertions.
        dispatch_inventory_dir = root / "dispatch-inventory"
        dispatch_inventory_dir.mkdir()
        dispatch_inventory_path = dispatch_inventory_dir / "inventory.json"
        inventory.write_inventory_receipt(
            dispatch_inventory_path,
            two_result,
        )
        pair_path_text = tuple(
            sorted(str(path.resolve()) for path in two_paths)
        )
        revalidated_owners: list[str] = []
        original_edge_revalidator = (
            revalidation.revalidate_edge_comparison
        )

        def recording_edge_revalidator(
            retained_receipt_payload: Any,
            *,
            owner: str,
        ) -> None:
            revalidated_owners.append(owner)
            original_edge_revalidator(
                retained_receipt_payload,
                owner=owner,
            )

        revalidation.revalidate_edge_comparison = (
            recording_edge_revalidator
        )
        try:
            dispatch_evidence, dispatch_snapshots = (
                dispatch._qualification_evidence(
                    HOSTS2,
                    pair_path_text,
                    str(dispatch_inventory_path),
                    repo=dispatch.ROOT,
                )
            )
        finally:
            revalidation.revalidate_edge_comparison = (
                original_edge_revalidator
            )
        owner_counts = {
            owner: revalidated_owners.count(owner)
            for owner in set(revalidated_owners)
        }
        assert owner_counts == {
            path: 2 for path in pair_path_text
        }
        assert (
            dispatch_evidence["inventory_root_sha256"]
            == two_result.inventory_root_sha256
        )
        assert dispatch_evidence["accepted_pairwise_receipt_count"] == 4
        assert len(dispatch_snapshots) == 5
        print(
            "  [PASS] dispatch genuinely revalidates every accepted edge "
            "after autonomous inventory revalidation"
        )

        # A coherent-looking statistic mutation is schema-valid for H=2: there
        # is only one edge per stage, and increasing the numeric-value count
        # preserves all local accounting inequalities.  Only rerunning the
        # comparator can establish the true count.
        coherent_statistics_paths = _copy_receipts(
            two_paths,
            root / "coherent-comparison-statistics",
        )
        _rewrite(
            coherent_statistics_paths[0],
            lambda payload: payload["comparison"].__setitem__(
                "compared_numeric_values",
                payload["comparison"]["compared_numeric_values"] + 3,
            ),
        )
        _expect_invalid_message(
            "coherent comparison-statistic forgery",
            HOSTS2,
            coherent_statistics_paths,
            "full from-disk recomputation",
            "compared_numeric_values",
        )
        coherent_path_text = tuple(
            sorted(
                str(path.resolve())
                for path in coherent_statistics_paths
            )
        )
        _expect_dispatch_invalid_message(
            "coherent comparison-statistic forgery",
            lambda: dispatch._qualification_evidence(
                HOSTS2,
                coherent_path_text,
                str(dispatch_inventory_path),
                repo=dispatch.ROOT,
            ),
            "qualification receipt graph failed revalidation",
            "compared_numeric_values",
        )

        # Genuinely numerical receipts: host-b regenerated with an
        # in-tolerance solver perturbation, then explicitly accepted.
        datasets_numerical = dict(datasets3)
        perturbed = _build_datasets(
            root / "datasets-numerical",
            ("host-b",),
            perturb_hosts=("host-b",),
        )
        datasets_numerical.update(perturbed)
        numerical_dir = root / "accepted-numerical"
        numerical_paths = _emit_graph_receipts(
            numerical_dir,
            datasets_numerical,
            HOSTS2,
            accept=True,
            expected_verdict="NUMERICALLY-EQUIVALENT",
        )
        numeric_result = inventory.validate_inventory(HOSTS2, numerical_paths)
        assert len(numeric_result.edges) == 4
        print(
            "  [PASS] genuinely numerical, explicitly accepted receipts "
            "accepted"
        )

        pending_paths = _copy_receipts(numerical_paths, root / "pending")
        _rewrite(
            pending_paths[0],
            lambda payload: payload.__setitem__(
                "numerical_equivalence_explicitly_accepted", False
            ),
        )
        _expect_invalid(
            "pending numerical equivalence", HOSTS2, pending_paths
        )

        forged_worst_paths = _copy_receipts(
            numerical_paths, root / "forged-worst-path"
        )
        _rewrite(
            forged_worst_paths[0],
            lambda payload: payload["comparison"].__setitem__(
                "worst_path", "bogus.AcelPrimVag"
            ),
        )
        _expect_invalid_before_disk_comparison(
            "forged numerical worst-path prefix", HOSTS2, forged_worst_paths
        )

        out_of_range_paths = _copy_receipts(
            numerical_paths, root / "out-of-range-worst-path"
        )
        _rewrite(
            out_of_range_paths[0],
            lambda payload: payload["comparison"].__setitem__(
                "worst_path", "0001.mat.data.AcelPrimVag[3]"
            ),
        )
        _expect_invalid(
            "out-of-range numerical passage index", HOSTS2, out_of_range_paths
        )

        equal_root_paths = _copy_receipts(two_paths, root / "equal-roots")

        def forge_equal_roots(payload: dict[str, Any]) -> None:
            payload["dataset_b"]["dataset_content_root_sha256"] = (
                payload["dataset_a"]["dataset_content_root_sha256"]
            )
            payload["dataset_content_roots_sha256"][1] = (
                payload["dataset_content_roots_sha256"][0]
            )
            payload["raw_byte_identical_states"] = 0

        _rewrite(equal_root_paths[0], forge_equal_roots)
        _expect_invalid(
            "equal roots with incomplete raw identity",
            HOSTS2,
            equal_root_paths,
        )

        _expect_invalid("H<2", ("host-a",), baseline_paths)
        _expect_invalid(
            "duplicate intended host ID",
            ("host-a", "host-a", "host-c"),
            baseline_paths,
        )
        _expect_invalid("missing edge", HOSTS3, baseline_paths[:-1])
        _expect_invalid(
            "duplicate receipt path",
            HOSTS3,
            baseline_paths[:-1] + [baseline_paths[0]],
        )

        def _mutated_case(
            name: str,
            mutate: Callable[
                [list[Path], inventory.CurrentInventoryPolicy], None
            ],
        ) -> None:
            paths = _copy_receipts(
                baseline_paths, root / "mutations" / name
            )
            mutate(paths, policy)

            _expect_invalid_before_disk_comparison(name, HOSTS3, paths)

        mutations: list[
            tuple[
                str,
                Callable[
                    [list[Path], inventory.CurrentInventoryPolicy],
                    None,
                ],
            ]
        ] = [
            (
                "stale schema",
                lambda ps, _: _rewrite(
                    ps[0], lambda p: p.__setitem__("schema", "v3")
                ),
            ),
            (
                "extra field",
                lambda ps, _: _rewrite(
                    ps[0], lambda p: p.__setitem__("extra", True)
                ),
            ),
            (
                "stale comparator",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p.__setitem__("comparator_sha256", "0" * 64),
                ),
            ),
            (
                "stale environment lock",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p.__setitem__(
                        "environment_lock_sha256", "0" * 64
                    ),
                ),
            ),
            (
                "stale parser environment",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p["parser_environment"].__setitem__(
                        "numpy", "0"
                    ),
                ),
            ),
            (
                "stale Python runtime root",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p.__setitem__(
                        "python_runtime_source_root_sha256", "0" * 64
                    ),
                ),
            ),
            (
                "stale generator root",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p.__setitem__(
                        "generator_source_root_sha256", "0" * 64
                    ),
                ),
            ),
            (
                "stale qualification source",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p.__setitem__(
                        "qualification_source_sha256", "0" * 64
                    ),
                ),
            ),
            (
                "spurious numerical acceptance on exact verdict",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p.__setitem__(
                        "numerical_equivalence_explicitly_accepted", True
                    ),
                ),
            ),
            (
                "material verdict",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p.__setitem__(
                        "verdict", "MATERIALLY-DIFFERENT"
                    ),
                ),
            ),
            (
                "undeclared host",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p["qualification_host_id"].__setitem__(
                        1, "host-z"
                    ),
                ),
            ),
            (
                "top-level dataset-root mismatch",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p["dataset_content_roots_sha256"].__setitem__(
                        0, "0" * 64
                    ),
                ),
            ),
            (
                "host diagnostic forgery",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p["dataset_a"].__setitem__(
                        "qualification_hostname", "forged.example"
                    ),
                ),
            ),
            (
                "loose tolerance",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p["tolerances"].__setitem__("rtol", 1e-4),
                ),
            ),
            (
                "exact verdict with numerical evidence",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p["comparison"].__setitem__(
                        "numerical_difference", True
                    ),
                ),
            ),
            (
                "comparison mismatches retained",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p["comparison"].__setitem__(
                        "mismatches", ["synthetic mismatch"]
                    ),
                ),
            ),
            (
                "wrong state inventory",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p["dataset_a"]["state_files"].pop(),
                ),
            ),
            (
                "nonzero disabled mechanism count",
                lambda ps, _: _rewrite(
                    ps[3],
                    lambda p: p["dataset_a"]["mechanism_coverage"].__setitem__(
                        "ballast_patch_rows", 1
                    ),
                ),
            ),
            (
                "forged disabled mechanism witness",
                lambda ps, _: _rewrite(
                    ps[3],
                    lambda p: p["dataset_a"]["mechanism_coverage"][
                        "witnesses"
                    ].__setitem__("ballast", "0001.mat:passage-1"),
                ),
            ),
            (
                "invalid disabled mechanism witness location",
                lambda ps, _: _rewrite(
                    ps[3],
                    lambda p: p["dataset_a"]["mechanism_coverage"][
                        "witnesses"
                    ].__setitem__("crack", "not-a-state-passage"),
                ),
            ),
            (
                "pairwise mechanism coverage disagreement",
                lambda ps, _: _rewrite(
                    ps[3],
                    lambda p: p["dataset_a"]["mechanism_coverage"].__setitem__(
                        "crack_active_passages",
                        p["dataset_a"]["mechanism_coverage"][
                            "crack_active_passages"
                        ]
                        + 1,
                    ),
                ),
            ),
            (
                "incident-edge mechanism coverage inconsistency",
                lambda ps, _: _rewrite_pair_coverage(ps[3]),
            ),
            (
                "non-SHA generation fingerprint",
                lambda ps, _: _rewrite_stage_fingerprint(ps),
            ),
            (
                "coherent legacy channel schema",
                lambda ps, _: _rewrite_pair_channel_schema(ps[0]),
            ),
            (
                "missing channel schema field",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda payload: payload["dataset_a"].pop(
                        "channel_schema_id"
                    ),
                ),
            ),
            (
                "coherently forged qualification executable SHA",
                lambda ps, _: _rewrite_stage_executed_sha(ps),
            ),
            (
                "structural comparison count drift across pairs",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p["comparison"].__setitem__(
                        "compared_numeric_values",
                        p["comparison"]["compared_numeric_values"] + 3,
                    ),
                ),
            ),
            (
                "structural leaf-class drift across pairs",
                lambda ps, _: _rewrite_leaf_class_drift(ps[0]),
            ),
            (
                "coherent cross-stage fingerprint reuse",
                lambda ps, _: _rewrite_cross_stage_fingerprint(ps),
            ),
            (
                "canonical host text with surrounding whitespace",
                lambda ps, _: _rewrite_host_identity(
                    ps, hostname=" host-a.fixture.invalid"
                ),
            ),
            (
                "oversized canonical host text",
                lambda ps, _: _rewrite_host_identity(
                    ps, hostname="h" * 1025
                ),
            ),
            (
                "inconsistent comparison leaf accounting",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p["comparison"].__setitem__(
                        "compared_leaves",
                        p["comparison"]["compared_leaves"] + 1,
                    ),
                ),
            ),
            (
                "signal count exceeds numeric count",
                lambda ps, _: _rewrite(
                    ps[0],
                    lambda p: p["comparison"].__setitem__(
                        "compared_signal_values",
                        p["comparison"]["compared_numeric_values"] + 1,
                    ),
                ),
            ),
            (
                "single foreign production endpoint",
                lambda ps, _: _rewrite_foreign_endpoint(ps[0]),
            ),
            (
                "cross-stage host-ID reuse",
                lambda ps, _: _rewrite_cross_stage_host(ps),
            ),
            (
                "inconsistent incident endpoint",
                lambda ps, _: _rewrite_incident_endpoint(ps),
            ),
            (
                "duplicate graph edge replacing another",
                lambda ps, _: ps[-1].write_bytes(ps[0].read_bytes()),
            ),
        ]
        (root / "mutations").mkdir()
        for name, mutation in mutations:
            _mutated_case(name.replace(" ", "-"), mutation)

        # Mixed exact/numerical acceptance across stages, all genuine: exact
        # Three exact stages plus the genuinely numerical accepted F40-M receipt.
        mixed_dir = root / "mixed-exact-numerical"
        mixed_dir.mkdir()
        mixed_paths: list[Path] = []
        for source in two_paths:
            if "F40-M" in source.name:
                continue
            target = mixed_dir / source.name
            target.write_bytes(source.read_bytes())
            mixed_paths.append(target)
        numerical_f40m = next(
            path for path in numerical_paths if "F40-M" in path.name
        )
        target = mixed_dir / numerical_f40m.name
        target.write_bytes(numerical_f40m.read_bytes())
        mixed_paths.append(target)
        mixed_result = inventory.validate_inventory(HOSTS2, mixed_paths)
        assert len(mixed_result.edges) == 4
        print(
            "  [PASS] mixed exact/numerical graph with stable structural "
            "counts"
        )

        malformed_paths = _copy_receipts(two_paths, root / "malformed")
        raw = malformed_paths[0].read_text(encoding="utf-8")
        malformed_paths[0].write_text(
            raw.replace(
                '"schema": "matlab-environment-qualification-receipt-v4",',
                (
                    '"schema": "matlab-environment-qualification-receipt-v4",\n'
                    '  "schema": "matlab-environment-qualification-receipt-v4",'
                ),
                1,
            ),
            encoding="utf-8",
            newline="\n",
        )
        _expect_invalid("duplicate JSON key", HOSTS2, malformed_paths)

        noncanonical_paths = _copy_receipts(two_paths, root / "noncanonical")
        payload = json.loads(
            noncanonical_paths[0].read_text(encoding="utf-8")
        )
        noncanonical_paths[0].write_text(
            json.dumps(payload, sort_keys=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _expect_invalid("noncanonical JSON bytes", HOSTS2, noncanonical_paths)

        # --- Retained-endpoint disk revalidation (the P1-1 attack surface) ---
        endpoint_root = root / "endpoint-cases"
        endpoint_root.mkdir()

        # Positive control: a byte-identical copied dataset, coherently
        # rebound in the receipt, revalidates from disk.
        rebound_paths = _copy_receipts(two_paths, endpoint_root / "rebound")
        relocated = endpoint_root / "relocated-host-a-f40s"
        shutil.copytree(datasets3[("host-a", "F40-S")], relocated)
        _rebind_endpoint_path(
            rebound_paths,
            host="host-a",
            stage="F40-S",
            new_path=str(relocated),
        )
        rebound_result = inventory.validate_inventory(HOSTS2, rebound_paths)
        assert len(rebound_result.edges) == 4
        print(
            "  [PASS] rebound byte-identical dataset copy revalidates from "
            "disk"
        )

        # TOCTOU: the SAME receipts that just passed must fail on the next
        # run after one retained state file is mutated - proving the
        # inventory reopens endpoint directories on every invocation.
        state_file = relocated / "0001.mat"
        _flip_byte(state_file, state_file.stat().st_size - 1)
        _expect_invalid_message(
            "STALE/TOCTOU retained state mutated between runs",
            HOSTS2,
            rebound_paths,
            "host-a",
            "F40-S",
            "changed during validation",
        )

        # MOVED: rename the retained directory out from under the receipt.
        relocated.rename(endpoint_root / "relocated-host-a-f40s-moved")
        _expect_invalid_message(
            "MOVED retained dataset directory",
            HOSTS2,
            rebound_paths,
            "host-a",
            "F40-S",
            "cannot be resolved",
        )

        # MISSING: the auditor's exact bypass repro - a coherent receipt
        # graph whose recorded dataset path never exists on disk.
        missing_paths = _copy_receipts(two_paths, endpoint_root / "missing")
        _rebind_endpoint_path(
            missing_paths,
            host="host-a",
            stage="F40-S",
            new_path=str(endpoint_root / "missing" / "vanished-dataset"),
        )
        _expect_invalid_message(
            "MISSING retained dataset directory",
            HOSTS2,
            missing_paths,
            "host-a",
            "F40-S",
            "cannot be resolved",
        )

        # COHERENTLY FORGED root: receipt-internally consistent fabricated
        # content root over the genuine directory; only recomputing the
        # digest table from disk can reject it.
        forged_paths = _copy_receipts(
            two_paths, endpoint_root / "forged-root"
        )

        def forge_content_root(payload: dict[str, Any]) -> None:
            if payload["stage"] != "F40-S":
                return
            payload["dataset_a"]["dataset_content_root_sha256"] = _sha(
                "coherently-forged-endpoint-root"
            )
            payload["dataset_content_roots_sha256"][0] = (
                payload["dataset_a"]["dataset_content_root_sha256"]
            )

        for path in forged_paths:
            _rewrite(path, forge_content_root)
        _expect_invalid_message(
            "coherently forged dataset content root",
            HOSTS2,
            forged_paths,
            "host-a",
            "F40-S",
            "dataset_content_root_sha256",
        )

        # SUBSTITUTED: retained path rebound to a different REAL directory
        # (host-a's F40-M dataset) whose recomputed evidence differs.
        substituted_paths = _copy_receipts(
            two_paths, endpoint_root / "substituted"
        )
        _rebind_endpoint_path(
            substituted_paths,
            host="host-a",
            stage="F40-S",
            new_path=str(datasets3[("host-a", "F40-M")]),
        )
        _expect_invalid_message(
            "substituted real dataset directory",
            HOSTS2,
            substituted_paths,
            "host-a",
            "F40-S",
        )

        # DIGEST-TABLE LAUNDERING: mutate one retained byte (MAT header
        # text - semantically inert), then regenerate file_digests.mat and
        # the completion marker COHERENTLY.  The directory revalidates as a
        # dataset, but its recomputed content root no longer matches the
        # retained receipt.
        laundered_paths = _copy_receipts(
            two_paths, endpoint_root / "laundered"
        )
        laundered_dataset = endpoint_root / "laundered-host-a-f40s"
        shutil.copytree(datasets3[("host-a", "F40-S")], laundered_dataset)
        _rebind_endpoint_path(
            laundered_paths,
            host="host-a",
            stage="F40-S",
            new_path=str(laundered_dataset),
        )
        _flip_byte(laundered_dataset / "0001.mat", 100)
        _regenerate_digest_table(laundered_dataset)
        _expect_invalid_message(
            "coherently regenerated digest table after byte mutation",
            HOSTS2,
            laundered_paths,
            "host-a",
            "F40-S",
            "dataset_content_root_sha256",
        )

        # SYMLINK/JUNCTION indirection to a genuine dataset directory.
        link_paths = _copy_receipts(two_paths, endpoint_root / "linked")
        link = endpoint_root / "linked-dataset"
        link_created = False
        if hasattr(os, "symlink"):
            try:
                os.symlink(
                    datasets3[("host-a", "F40-S")],
                    link,
                    target_is_directory=True,
                )
                link_created = True
            except OSError:
                link_created = False
        if not link_created:
            junction = subprocess.run(
                [
                    "cmd",
                    "/c",
                    "mklink",
                    "/J",
                    str(link),
                    str(datasets3[("host-a", "F40-S")]),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            link_created = junction.returncode == 0 and link.exists()
        if link_created:
            _rebind_endpoint_path(
                link_paths,
                host="host-a",
                stage="F40-S",
                new_path=str(link),
            )
            _expect_invalid_message(
                "symlinked/junction retained dataset directory",
                HOSTS2,
                link_paths,
                "host-a",
                "F40-S",
            )
        else:
            print(
                "  [N/A] symlink/junction dataset test "
                "(host privilege unavailable)"
            )

        output = root / "inventory-receipt.json"
        inventory.write_inventory_receipt(output, result)
        persisted = json.loads(output.read_text(encoding="utf-8"))
        expected_payload = inventory.inventory_receipt_payload(result)
        assert persisted == expected_payload
        _assert_safe_inventory_payload(persisted, result)
        print(
            "  [PASS] safe payload has exact policy/root/count/edge/endpoint "
            "fidelity"
        )

        stale_result = replace(
            result,
            policy=replace(result.policy, checker_sha256="0" * 64),
        )
        stale_output = root / "stale-inventory-result.json"
        try:
            inventory.write_inventory_receipt(stale_output, stale_result)
        except inventory.QualificationInventoryError as exc:
            if "stale" not in str(exc):
                raise
        else:
            raise AssertionError(
                "writer accepted an arbitrary stale InventoryResult"
            )
        assert not stale_output.exists()
        print("  [PASS] writer recomputes and rejects stale InventoryResult")

        post_failure_output = root / "post-failure-inventory-result.json"
        expected_post_failure_bytes = inventory._canonical_json_bytes(
            inventory.inventory_receipt_payload(result)
        )
        original_result_revalidator = inventory._revalidate_inventory_result
        writer_revalidation_calls = {"count": 0}

        def fail_after_inventory_publication(
            candidate: inventory.InventoryResult,
            *,
            owner: str,
        ) -> inventory.InventoryResult:
            writer_revalidation_calls["count"] += 1
            if writer_revalidation_calls["count"] >= 2:
                raise inventory.QualificationInventoryError(
                    "synthetic post-publication endpoint drift"
                )
            return candidate

        inventory._revalidate_inventory_result = (
            fail_after_inventory_publication
        )
        try:
            try:
                inventory.write_inventory_receipt(
                    post_failure_output, result
                )
            except inventory.QualificationInventoryError as exc:
                if "retained for forensic review" not in str(exc):
                    raise
            else:
                raise AssertionError(
                    "post-publication inventory drift was accepted"
                )
        finally:
            inventory._revalidate_inventory_result = (
                original_result_revalidator
            )
        assert writer_revalidation_calls["count"] == 2
        assert (
            post_failure_output.read_bytes() == expected_post_failure_bytes
        )
        print(
            "  [PASS] post-publication failure retains exact create-once "
            "inventory receipt for forensics"
        )

        dropped = json.loads(json.dumps(persisted))
        dropped.pop("environment_lock_sha256")
        _expect_payload_assertion("dropped policy field", dropped, result)

        duplicated_edge = json.loads(json.dumps(persisted))
        duplicated_edge["receipt_edges"].append(
            dict(duplicated_edge["receipt_edges"][0])
        )
        _expect_payload_assertion(
            "extra receipt-edge row", duplicated_edge, result
        )

        forged_endpoint = json.loads(json.dumps(persisted))
        forged_endpoint["endpoint_bindings"][0][
            "dataset_content_root_sha256"
        ] = "0" * 64
        _expect_payload_assertion(
            "forged endpoint root", forged_endpoint, result
        )

        original = output.read_bytes()
        try:
            inventory.write_inventory_receipt(output, result)
        except inventory.QualificationInventoryError:
            pass
        else:
            raise AssertionError("inventory receipt overwrite was accepted")
        assert output.read_bytes() == original
        print("  [PASS] safe inventory receipt is exclusive-write")

        assert (
            inventory.main(
                [
                    "--host",
                    "host-a",
                    "--host",
                    "host-b",
                    "--receipt-dir",
                    str(two_hosts),
                ]
            )
            == 0
        )
        print("  [PASS] CLI directory inventory")

        assert policy == inventory.current_policy()
    print("QUALIFICATION RECEIPT INVENTORY: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

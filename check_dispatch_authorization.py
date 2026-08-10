"""Adversarial checks for the mechanical Paper-1 dispatch-evidence gate.

These checks do not fabricate qualifying benchmark, MATLAB-host, or contact
evidence.  They exercise the strict manifest grammar, cross-evidence bindings,
create-once publication, TOCTOU snapshots, Git A->B boundary, and production
call graph.  The three underlying verifiers retain their own independent
synthetic/mutation suites.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import dispatch_authorization as gate
import dispatch_manifest as manifest
import qualification_endpoint_revalidation as endpoint_revalidation
import qualification_receipt_inventory as qualification_inventory


FAILURES = 0


def check(label: str, condition: bool) -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        FAILURES += 1


def expect_failure(label: str, action) -> None:
    try:
        action()
    except gate.DispatchAuthorizationError:
        check(label, True)
    else:
        check(label, False)


def expect_failure_cause(label: str, action, needle: str) -> None:
    """Require rejection by the intended defence layer, not a later guard."""
    try:
        action()
    except gate.DispatchAuthorizationError as exc:
        causes: list[BaseException] = []
        current: BaseException | None = exc
        while current is not None and current not in causes:
            causes.append(current)
            current = current.__cause__ or current.__context__
        check(label, any(needle in str(cause) for cause in causes))
    else:
        check(label, False)


def sha(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def valid_payload() -> dict:
    source_hashes = {
        name: sha(f"source:{name}") for name in gate.POLICY_SOURCE_FILES
    }
    return {
        "schema": gate.MANIFEST_SCHEMA,
        "status": gate.STATUS,
        "tested_source_commit": "a" * 40,
        "policy": {
            "schema": gate.POLICY_SCHEMA,
            "tested_source_tree_oid": "b" * 40,
            "source_sha256": source_hashes,
            "source_root_sha256": sha("policy-root"),
        },
        "benchmark": {
            "schema": gate.REQUIRED_BENCHMARK_SCHEMA,
            "status": "PASS",
            "tested_source_commit": "a" * 40,
            "descriptor_sha256": sha("descriptor"),
            "evidence_root_sha256": sha("benchmark-root"),
            "summary_sha256": sha("summary"),
            "run_state_sha256": sha("run-state"),
            "run_directory": str(Path("C:/evidence/benchmark")),
        },
        "qualification": {
            "required_stages": list(gate.REQUIRED_QUALIFICATION_STAGES),
            "intended_host_ids": ["host-a", "host-b"],
            "pair_receipt_paths": [
                str(Path(f"C:/evidence/pair-{index}.json"))
                for index in range(4)
            ],
            "inventory_receipt_path": str(
                Path("C:/evidence/inventory.json")),
            "inventory_receipt_sha256": sha("inventory-receipt"),
            "inventory_root_sha256": sha("inventory-root"),
            "accepted_pairwise_receipt_count": 4,
            "generator_source_root_sha256": sha("generator"),
            "matlab_environment_sha256": sha("matlab"),
        },
        "contact_closure": {
            "gate_directory": str(Path("C:/evidence/contact")),
            "authorization_receipt_path": str(
                Path("C:/evidence/contact-receipt.json")),
            "authorization_receipt_sha256": sha("contact-receipt"),
            "receipt_schema":
                "contact-closure-authorization-receipt-v2",
            "status": "ACCEPTED",
            "declared_host_id": "host-a",
            "matlab_environment_sha256": sha("matlab"),
            "generator_source_root_sha256": sha("generator"),
            "policy_sha256": sha("contact-policy"),
            "selection_sha256": sha("selection"),
            "case_artifact_root_sha256": sha("case-root"),
            "gate_summary_sha256": sha("gate-summary"),
            "gate_artifact_root_sha256": sha("gate-artifacts"),
            "dataset_descriptors_root_sha256": sha("datasets"),
            "expected_cases": gate.EXPECTED_CONTACT_CASES,
            "accepted_cases": gate.EXPECTED_CONTACT_CASES,
            "channel_schema_id": gate.REQUIRED_CHANNEL_SCHEMA_ID,
        },
    }


def mutate(
    label: str,
    mutator,
) -> None:
    payload = valid_payload()
    mutator(payload)
    expect_failure(label, lambda: gate._validate_manifest_shape(payload))


print("DISPATCH AUTHORIZATION CHECKS")

baseline = valid_payload()
try:
    gate._validate_manifest_shape(baseline)
except gate.DispatchAuthorizationError as exc:
    print(f"    baseline unexpectedly failed: {exc}")
    check("strict manifest baseline", False)
else:
    check("strict manifest baseline", True)

mutations = (
    ("top-level extra field",
     lambda p: p.__setitem__("comment", "AUTHORIZED")),
    ("top-level missing evidence",
     lambda p: p.pop("benchmark")),
    ("status prose cannot authorize",
     lambda p: p.__setitem__("status", "PENDING AUTHORIZED")),
    ("tested commit must be lowercase SHA",
     lambda p: p.__setitem__("tested_source_commit", "A" * 40)),
    ("policy has exact fields",
     lambda p: p["policy"].__setitem__("trusted", True)),
    ("policy source inventory is exact",
     lambda p: p["policy"]["source_sha256"].pop(
         "dispatch_authorization.py")),
    ("policy source hash is lowercase SHA-256",
     lambda p: p["policy"]["source_sha256"].__setitem__(
         "dispatch_authorization.py", "0" * 63)),
    ("benchmark has exact fields",
     lambda p: p["benchmark"].__setitem__("objective_value", 0.0)),
    ("legacy R11 benchmark evidence cannot authorize Paper-1",
     lambda p: p["benchmark"].__setitem__(
         "schema", "ttbi-r11-benchmark-authorization-evidence-v1")),
    ("legacy Paper-1 v1 benchmark evidence cannot authorize v2",
     lambda p: p["benchmark"].__setitem__(
         "schema", "ttbi-paper1-benchmark-authorization-evidence-v1")),
    ("benchmark must pass",
     lambda p: p["benchmark"].__setitem__("status", "FAIL")),
    ("benchmark binds tested A",
     lambda p: p["benchmark"].__setitem__(
         "tested_source_commit", "c" * 40)),
    ("qualification has exact fields",
     lambda p: p["qualification"].__setitem__("accepted", True)),
    ("qualification requires multiple hosts",
     lambda p: p["qualification"].__setitem__(
         "intended_host_ids", ["host-a"])),
    ("qualification count is typed",
     lambda p: p["qualification"].__setitem__(
         "accepted_pairwise_receipt_count", True)),
    ("contact has exact fields",
     lambda p: p["contact_closure"].__setitem__("note", "pass")),
    ("contact receipt schema is exact",
     lambda p: p["contact_closure"].__setitem__(
         "receipt_schema", "contact-closure-authorization-receipt-v0")),
    ("contact must accept all 420",
     lambda p: p["contact_closure"].__setitem__("accepted_cases", 419)),
    ("contact source root must equal qualification",
     lambda p: p["contact_closure"].__setitem__(
         "generator_source_root_sha256", sha("foreign-generator"))),
    ("contact environment must equal qualification",
     lambda p: p["contact_closure"].__setitem__(
         "matlab_environment_sha256", sha("foreign-matlab"))),
    ("contact host must belong to intended inventory",
     lambda p: p["contact_closure"].__setitem__(
         "declared_host_id", "host-c")),
)
for label, mutator in mutations:
    mutate(label, mutator)

# Strict JSON rejects formatting ambiguity in addition to semantic mutations.
canonical = gate._canonical_json_bytes(valid_payload())
parsed = gate._strict_json_bytes(canonical, "synthetic manifest")
check("canonical JSON round trip is exact", parsed == valid_payload())
expect_failure(
    "pretty-print/key-order drift rejected",
    lambda: gate._strict_json_bytes(
        b'{"status":"AUTHORIZED","schema":"x"}\n', "noncanonical"),
)
expect_failure(
    "CRLF JSON rejected",
    lambda: gate._strict_json_bytes(
        canonical.replace(b"\n", b"\r\n"), "CRLF manifest"),
)
expect_failure(
    "missing final LF rejected",
    lambda: gate._strict_json_bytes(canonical[:-1], "unterminated manifest"),
)
expect_failure_cause(
    "duplicate JSON key rejected",
    lambda: gate._strict_json_bytes(
        b'{\n  "schema": "a",\n  "schema": "b"\n}\n',
        "duplicate-key manifest",
    ),
    "duplicate JSON key",
)
expect_failure(
    "NaN JSON rejected",
    lambda: gate._strict_json_bytes(
        b'{\n  "value": NaN\n}\n', "nonfinite manifest"),
)
manifest_raw = (gate.ROOT / "bundle_source_files.txt").read_bytes()
policy_names = gate._policy_source_names(manifest_raw)
manifest_python = {
    line for line in
    (gate.ROOT / "bundle_source_files.txt").read_text(
        encoding="utf-8").splitlines()
    if line.endswith(".py")
}
check(
    "policy binds every shipped Python source plus environment locks",
    manifest_python.issubset(set(policy_names))
    and {
        "bundle_source_files.txt",
        "environment/campaign-py313-cu128.json",
        "requirements-campaign-py313-cu128.txt",
    }.issubset(policy_names),
)
expect_failure(
    "unsorted policy-source manifest rejected",
    lambda: gate._policy_source_names(
        b"dispatch_authorization.py\nbenchmark_paper1_compute.py\n"),
)
expect_failure(
    "bundle-builder omission from source manifest rejected",
    lambda: gate._policy_source_names(
        manifest_raw.replace(b"build_stage_bundles.py\n", b"", 1)
    ),
)

# A regular file snapshot is content- and identity-stable, not just a pathname.
with tempfile.TemporaryDirectory(prefix="dispatch-snapshot-") as td:
    path = Path(td, "receipt.json").resolve()
    path.write_bytes(b"one\n")
    snapshot = gate._snapshot_regular(path, "synthetic receipt")
    gate._assert_snapshot_unchanged(snapshot, "synthetic receipt")
    check("unchanged snapshot accepted", True)
    path.write_bytes(b"two\n")
    expect_failure(
        "same-size content mutation trips TOCTOU guard",
        lambda: gate._assert_snapshot_unchanged(
            snapshot, "synthetic receipt"),
    )
    path.write_bytes(b"one\n")
    expect_failure(
        "restored bytes do not erase identity/mtime mutation",
        lambda: gate._assert_snapshot_unchanged(
            snapshot, "synthetic receipt"),
    )
    hardlink = Path(td, "hardlink.json")
    try:
        os.link(path, hardlink)
    except OSError:
        print("  [N/A] hard-link evidence test (filesystem unavailable)")
    else:
        expect_failure(
            "hard-link evidence alias rejected",
            lambda: gate._canonical_existing_path(
                str(path), "hard-linked receipt", kind="file"),
        )
        hardlink.unlink()
    expect_failure(
        "relative evidence path rejected",
        lambda: gate._canonical_existing_path(
            "receipt.json", "relative receipt", kind="file"),
    )
    repository_fixture = Path(td, "repo").resolve()
    repository_fixture.mkdir()
    internal_receipt = repository_fixture / "internal.json"
    internal_receipt.write_bytes(b"internal\n")
    expect_failure(
        "authorization receipt inside repository rejected",
        lambda: gate._require_external_to_repo(
            internal_receipt, repository_fixture, "internal receipt"),
    )
    if hasattr(os, "symlink"):
        link = Path(td, "linked.json")
        try:
            link.symlink_to(path)
        except OSError:
            print("  [N/A] symlink evidence test (host privilege unavailable)")
        else:
            expect_failure(
                "symlink evidence path rejected",
                lambda: gate._canonical_existing_path(
                    str(link.resolve().parent / link.name),
                    "linked receipt",
                    kind="file",
                ),
            )

# Publication uses direct exclusive creation. Existing finals are immutable;
# every legacy/stale temporary remains untouched for forensic review.
with tempfile.TemporaryDirectory(prefix="dispatch-publish-") as td:
    root = Path(td).resolve()
    output = root / "authorization.json"
    digest = gate._publish_create_once(output, valid_payload())
    check(
        "create-once publication writes exact canonical bytes",
        digest == hashlib.sha256(output.read_bytes()).hexdigest()
        and output.read_bytes() == canonical,
    )
    expect_failure(
        "create-once publication refuses overwrite",
        lambda: gate._publish_create_once(output, valid_payload()),
    )
    recover_output = root / "recover.json"
    recover_tmp = recover_output.with_name(recover_output.name + ".tmp")
    recover_tmp.write_bytes(canonical)
    expect_failure(
        "legacy exact temporary is retained for forensic review",
        lambda: gate._publish_create_once(recover_output, valid_payload()),
    )
    check(
        "legacy exact temporary was not unlinked",
        recover_tmp.read_bytes() == canonical and not recover_output.exists(),
    )
    bad_output = root / "bad.json"
    bad_output.with_name(bad_output.name + ".tmp").write_bytes(b"foreign\n")
    expect_failure(
        "foreign interrupted temporary is fail-closed",
        lambda: gate._publish_create_once(bad_output, valid_payload()),
    )

# Creation must cross a stronger boundary than receipt stability alone.  These
# production-flow tests mutate one underlying evidence class immediately after
# the create-once link becomes visible.  The second full evidence collection
# must see the mutation and reject authorization.  The create-once file stays
# in place as forensic evidence; no check-then-unlink pathname race is allowed.
with tempfile.TemporaryDirectory(prefix="dispatch-post-publish-") as td:
    fixture_root = Path(td).resolve()
    fixture_repo = fixture_root / "repo"
    external = fixture_root / "external"
    fixture_repo.mkdir()
    external.mkdir()
    benchmark_dir = external / "benchmark"
    contact_dir = external / "contact"
    pair_dirs = tuple(external / f"pair-{index}" for index in range(4))
    benchmark_dir.mkdir()
    contact_dir.mkdir()
    for directory in pair_dirs:
        directory.mkdir()
    benchmark_file = benchmark_dir / "summary.json"
    qualification_file = external / "inventory.json"
    contact_file = external / "contact-receipt.json"
    pair_files = tuple(
        directory / "receipt.json" for directory in pair_dirs
    )
    evidence_files = (
        benchmark_file,
        qualification_file,
        contact_file,
        *pair_files,
    )
    for index, path in enumerate(evidence_files):
        path.write_bytes(f"baseline-{index}\n".encode("ascii"))

    def synthetic_collection() -> tuple[dict, tuple]:
        payload = valid_payload()
        payload["benchmark"]["run_directory"] = str(benchmark_dir)
        payload["benchmark"]["summary_sha256"] = hashlib.sha256(
            benchmark_file.read_bytes()
        ).hexdigest()
        payload["benchmark"]["evidence_root_sha256"] = sha(
            "benchmark:" + payload["benchmark"]["summary_sha256"])
        payload["qualification"]["pair_receipt_paths"] = [
            str(path) for path in pair_files
        ]
        payload["qualification"]["inventory_receipt_path"] = str(
            qualification_file)
        payload["qualification"]["inventory_receipt_sha256"] = (
            hashlib.sha256(qualification_file.read_bytes()).hexdigest()
        )
        payload["qualification"]["inventory_root_sha256"] = sha(
            "qualification:"
            + payload["qualification"]["inventory_receipt_sha256"]
        )
        payload["contact_closure"]["gate_directory"] = str(contact_dir)
        payload["contact_closure"]["authorization_receipt_path"] = str(
            contact_file)
        payload["contact_closure"]["authorization_receipt_sha256"] = (
            hashlib.sha256(contact_file.read_bytes()).hexdigest()
        )
        payload["contact_closure"]["case_artifact_root_sha256"] = sha(
            "contact:"
            + payload["contact_closure"]["authorization_receipt_sha256"]
        )
        snapshots = tuple(
            gate._snapshot_regular(path, f"synthetic evidence {index}")
            for index, path in enumerate(evidence_files)
        )
        return payload, snapshots

    original_collect_evidence = gate._collect_evidence
    original_git_head = gate._git_head
    original_publish_create_once = gate._publish_create_once
    gate._git_head = lambda _repo: "a" * 40
    gate._collect_evidence = lambda **_kwargs: synthetic_collection()
    try:
        mutation_targets = (
            ("benchmark payload", benchmark_file),
            ("qualification endpoint/inventory payload", qualification_file),
            ("contact gate/dataset payload", contact_file),
        )
        for index, (owner, target) in enumerate(mutation_targets):
            for file_index, path in enumerate(evidence_files):
                path.write_bytes(
                    f"baseline-{file_index}\n".encode("ascii"))
            output = external / f"authorization-{index}.json"
            expected_published = gate._canonical_json_bytes(
                synthetic_collection()[0]
            )

            def publish_then_mutate(path, payload, *, target=target):
                digest = original_publish_create_once(path, payload)
                target.write_bytes(b"mutated-after-publication\n")
                return digest

            gate._publish_create_once = publish_then_mutate
            expect_failure(
                f"post-publication full revalidation catches {owner} change",
                lambda output=output: gate.create_dispatch_authorization_manifest(
                    output,
                    tested_source_commit="a" * 40,
                    benchmark_run_directory=str(benchmark_dir),
                    qualification_host_ids=("host-a", "host-b"),
                    qualification_pair_receipts=tuple(
                        str(path) for path in pair_files),
                    qualification_inventory_receipt=str(qualification_file),
                    contact_gate_directory=str(contact_dir),
                    contact_authorization_receipt=str(contact_file),
                    repo=fixture_repo,
                ),
            )
            check(
                f"failed {owner} post-check retains create-once manifest",
                os.path.lexists(output)
                and output.read_bytes() == expected_published,
            )

        # If the published path itself no longer has the intended bytes, there
        # is no authenticated snapshot from which cleanup would be safe.
        output = external / "authorization-mutated.json"

        def publish_then_replace(path, payload):
            digest = original_publish_create_once(path, payload)
            path.write_bytes(b"foreign-after-publication\n")
            return digest

        gate._publish_create_once = publish_then_replace
        expect_failure(
            "changed published manifest is left for forensic review",
            lambda: gate.create_dispatch_authorization_manifest(
                output,
                tested_source_commit="a" * 40,
                benchmark_run_directory=str(benchmark_dir),
                qualification_host_ids=("host-a", "host-b"),
                qualification_pair_receipts=tuple(
                    str(path) for path in pair_files),
                qualification_inventory_receipt=str(qualification_file),
                contact_gate_directory=str(contact_dir),
                contact_authorization_receipt=str(contact_file),
                repo=fixture_repo,
            ),
        )
        check(
            "changed published manifest remains untouched for forensic review",
            output.read_bytes() == b"foreign-after-publication\n",
        )
    finally:
        gate._collect_evidence = original_collect_evidence
        gate._git_head = original_git_head
        gate._publish_create_once = original_publish_create_once

# The inventory performs its own read, but dispatch has already captured an
# earlier exact byte snapshot.  A returned graph for pathname A with the SHA of
# replacement bytes B must be rejected before aggregate evidence is consumed.
with tempfile.TemporaryDirectory(prefix="dispatch-pair-ab-binding-") as td:
    external = Path(td).resolve()
    pair_dir = external / "pair"
    aggregate_dir = external / "aggregate"
    pair_dir.mkdir()
    aggregate_dir.mkdir()
    pair_path = pair_dir / "receipt.json"
    inventory_path = aggregate_dir / "inventory.json"
    pair_path.write_bytes(b"pair-version-A\n")
    inventory_path.write_bytes(b"{}\n")
    fake_result = SimpleNamespace(
        edges=(
            SimpleNamespace(
                path=pair_path,
                receipt_sha256=hashlib.sha256(
                    b"pair-version-B\n"
                ).hexdigest(),
            ),
        ),
    )
    original_validate_inventory = qualification_inventory.validate_inventory
    original_inventory_payload = (
        qualification_inventory.inventory_receipt_payload
    )
    original_revalidate_edge = (
        endpoint_revalidation.revalidate_edge_comparison
    )
    qualification_inventory.validate_inventory = (
        lambda _hosts, _paths: fake_result
    )
    qualification_inventory.inventory_receipt_payload = lambda _result: {}
    endpoint_revalidation.revalidate_edge_comparison = (
        lambda _payload, *, owner: None
    )
    try:
        expect_failure(
            "dispatch rejects inventory SHA B for initial pair snapshot A",
            lambda: gate._qualification_evidence(
                ("host-a", "host-b"),
                (str(pair_path),),
                str(inventory_path),
                repo=gate.ROOT,
            ),
        )
    finally:
        qualification_inventory.validate_inventory = (
            original_validate_inventory
        )
        qualification_inventory.inventory_receipt_payload = (
            original_inventory_payload
        )
        endpoint_revalidation.revalidate_edge_comparison = (
            original_revalidate_edge
        )


def run_git(repo: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            process.stderr.decode("utf-8", errors="replace"))
    return process.stdout.decode("utf-8").strip()


with tempfile.TemporaryDirectory(prefix="dispatch-git-") as td:
    repo = Path(td, "repo")
    repo.mkdir()
    run_git(repo, "init")
    run_git(repo, "config", "user.name", "Dispatch Guard")
    run_git(repo, "config", "user.email", "dispatch@example.invalid")
    run_git(repo, "config", "core.autocrlf", "false")
    (repo / "runtime.py").write_text(
        "VALUE = 1\n", encoding="utf-8", newline="\n")
    (repo / "docs").mkdir()
    report = repo / gate.REPORT_PATH
    report.write_text("blocked\n", encoding="utf-8", newline="\n")
    run_git(repo, "add", "--all")
    run_git(repo, "commit", "-m", "tested A")
    tested_a = run_git(repo, "rev-parse", "HEAD")
    gate._assert_git_boundary(repo, tested_a, tested_a)
    check("clean tested A boundary accepted for manifest creation", True)
    report.write_text("authorized\n", encoding="utf-8", newline="\n")
    run_git(repo, "add", "--all")
    run_git(repo, "commit", "-m", "report-only B")
    dispatch_b = run_git(repo, "rev-parse", "HEAD")
    gate._assert_git_boundary(repo, tested_a, dispatch_b)
    check("clean report-only A-to-B boundary accepted", True)
    (repo / "runtime.py").write_text(
        "VALUE = 2\n", encoding="utf-8", newline="\n")
    run_git(repo, "add", "--all")
    run_git(repo, "commit", "-m", "runtime delta")
    runtime_head = run_git(repo, "rev-parse", "HEAD")
    expect_failure(
        "committed runtime delta after A rejected",
        lambda: gate._assert_git_boundary(repo, tested_a, runtime_head),
    )
    (repo / "untracked.txt").write_text(
        "foreign\n", encoding="utf-8", newline="\n")
    expect_failure(
        "untracked evidence-time file rejected",
        lambda: gate._assert_git_boundary(repo, tested_a, runtime_head),
    )

# The production API has no injectable validators and calls each authoritative
# implementation directly.  This guards against a green test backend becoming
# a CLI-accessible authorization bypass.
verify_parameters = tuple(
    inspect.signature(
        gate.verify_dispatch_authorization_manifest
    ).parameters
)
check(
    "production verifier exposes no validator/bypass injection",
    verify_parameters == (
        "manifest_path",
        "tested_source_commit",
        "dispatch_source_commit",
        "expected_manifest_sha256",
        "repo",
    ),
)
source = Path(gate.__file__).read_text(encoding="utf-8")
# The evidence-agnostic mechanics (manifest grammar, canonical JSON, TOCTOU
# snapshots, canonical paths, create-once publication) live in
# dispatch_manifest; the orchestration calls stay in the entry module.  Pins
# are therefore evaluated against the module that owns each concern, and the
# create-once/TOCTOU mechanics against the union of the two.
manifest_source = Path(manifest.__file__).read_text(encoding="utf-8")
combined_source = source + "\n" + manifest_source
check(
    "production gate invokes all three authoritative revalidators",
    "benchmark.verify_completed_receipt(" in source
    and "inventory.validate_inventory(" in source
    and "contact.verify_existing_authorization_receipt(" in source,
)
check(
    "production gate reruns the full pairwise comparator per retained edge",
    "endpoint_revalidation.revalidate_edge_comparison(" in source,
)
check(
    "retained inventory must equal canonical recomputation",
    "inventory._canonical_json_bytes(expected_payload)" in source
    and "inventory_snapshot.raw != expected_bytes" in source,
)
check(
    "manifest is external, canonical, create-once and TOCTOU guarded",
    "must be external to the repository" in combined_source
    and 'with path.open("xb") as stream:' in combined_source
    and ".unlink(" not in combined_source
    and "_assert_snapshot_unchanged(" in combined_source
    and "_strict_json_bytes(" in combined_source,
)
check(
    "dispatch policy identity covers split qualification/contact modules",
    (
        {
        "benchmark_paper1_compute.py",
        "qualification_receipt_schema.py",
        "qualification_endpoint_revalidation.py",
        "qualification_path_safety.py",
        "dispatch_manifest.py",
        }
        <= set(gate.POLICY_SOURCE_FILES)
        and set(manifest.CONTACT_GATE_VERIFIER_SOURCE_FILES)
        <= set(gate.POLICY_SOURCE_FILES)
    ),
)
check(
    "contact revalidation pins A across report-only B",
    "report_only_git_check" in source
    and "_assert_git_boundary(" in source
    and "tested_source_commit" in source
    and "dispatch_source_commit" in source,
)

# Exercise the dispatch-to-benchmark boundary itself without copying or
# weakening the benchmark checker's independent production-shaped fixture.
# Monkeypatching is confined to this checker process; the production function
# exposes no validator argument or environment-controlled backend.
import benchmark_paper1_compute as benchmark  # noqa: E402

with tempfile.TemporaryDirectory(prefix="dispatch-benchmark-boundary-") as td:
    run_dir = Path(td, "run").resolve()
    run_dir.mkdir()
    summary = run_dir / "summary.json"
    run_state = run_dir / "run_state.json"
    summary.write_bytes(b"summary fixture\n")
    run_state.write_bytes(b"state fixture\n")
    summary_sha = hashlib.sha256(summary.read_bytes()).hexdigest()
    state_sha = hashlib.sha256(run_state.read_bytes()).hexdigest()
    tested_a = "a" * 40
    expected_evidence = {
        "schema": gate.REQUIRED_BENCHMARK_SCHEMA,
        "status": "PASS",
        "tested_source_commit": tested_a,
        "descriptor_sha256": sha("descriptor-boundary"),
        "evidence_root_sha256": sha("evidence-boundary"),
        "summary_sha256": summary_sha,
        "run_state_sha256": state_sha,
        "run_directory": str(run_dir),
    }
    calls = []
    original_benchmark_verifier = benchmark.verify_completed_receipt

    def passing_benchmark_verifier(path, commit, *, repo):
        calls.append((Path(path), commit, Path(repo)))
        return dict(expected_evidence)

    benchmark.verify_completed_receipt = passing_benchmark_verifier
    try:
        observed, _ = gate._benchmark_evidence(
            str(run_dir), tested_a, repo=gate.ROOT)
    finally:
        benchmark.verify_completed_receipt = original_benchmark_verifier
    check(
        "dispatch invokes benchmark verifier once and binds its exact return",
        observed == expected_evidence
        and calls == [(run_dir, tested_a, gate.ROOT)],
    )

    def failing_benchmark_verifier(path, commit, *, repo):
        raise benchmark.ContractError("forged zero-compute receipt")

    benchmark.verify_completed_receipt = failing_benchmark_verifier
    try:
        expect_failure(
            "benchmark verifier rejection propagates to dispatch",
            lambda: gate._benchmark_evidence(
                str(run_dir), tested_a, repo=gate.ROOT),
        )
    finally:
        benchmark.verify_completed_receipt = original_benchmark_verifier

    def extra_field_verifier(path, commit, *, repo):
        forged = dict(expected_evidence)
        forged["claimed_trials"] = 100
        return forged

    benchmark.verify_completed_receipt = extra_field_verifier
    try:
        expect_failure(
            "dispatch rejects extra fields in benchmark evidence",
            lambda: gate._benchmark_evidence(
                str(run_dir), tested_a, repo=gate.ROOT),
        )
    finally:
        benchmark.verify_completed_receipt = original_benchmark_verifier

    def mismatched_digest_verifier(path, commit, *, repo):
        forged = dict(expected_evidence)
        forged["summary_sha256"] = "0" * 64
        return forged

    benchmark.verify_completed_receipt = mismatched_digest_verifier
    try:
        expect_failure(
            "dispatch rejects benchmark summary digest substitution",
            lambda: gate._benchmark_evidence(
                str(run_dir), tested_a, repo=gate.ROOT),
        )
    finally:
        benchmark.verify_completed_receipt = original_benchmark_verifier

if FAILURES:
    raise SystemExit(f"{FAILURES} dispatch-authorization check(s) failed")
print("DISPATCH AUTHORIZATION CHECKS: ALL PASS")

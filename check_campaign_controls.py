"""Adversarial checks for sensor-budget controls and RUN_TAG artifacts.

The campaign driver cannot be imported without a complete generated dataset,
by design.  This checker compiles the relevant function definitions directly
from its AST, so it exercises the shipped implementation without touching
``data/``, ``results/``, studies, or bundles.

Run:  python check_campaign_controls.py
"""
from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace
import zipfile

import numpy as np


DRIVER = Path(__file__).with_name("comprehensive_ablation_multidamage.py")
SOURCE = DRIVER.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(DRIVER))
FUNCTIONS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
fails = 0


def check(name: str, cond: bool) -> None:
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails += 1


def extracted(*names: str, namespace: dict | None = None) -> dict:
    ns = {} if namespace is None else dict(namespace)
    missing = [n for n in names if n not in FUNCTIONS]
    if missing:
        raise RuntimeError(f"driver functions missing: {missing}")
    module = ast.Module(body=[FUNCTIONS[n] for n in names], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(DRIVER), "exec"), ns)
    return ns


print("CAMPAIGN CONTROL CHECKS")

# 1. The function that formerly crashed on ALL_DOFS must accept both pairs and
# controls, preserve the legacy pair label, and bound the all-array path.
calls = []
names = {i: f"DOF{i}" for i in range(8)}
ns = extracted(
    "phase_2_pair",
    namespace={
        "ALL_DOFS": list(range(8)),
        "IDX_TO_DOF_NAME": names,
        "run_phase": lambda label, dofs: calls.append((label, list(dofs))),
    },
)
phase = ns["phase_2_pair"]
phase([1, 3])
phase(list(range(8)))
check("two-sensor phase keeps existing pair label",
      calls[0] == ("MD0_Pair_DOF1__DOF3", [1, 3]))
check("ALL_DOFS control executes instead of unpacking/crashing",
      calls[1] == ("MD0_Control_8DOF_0_1_2_3_4_5_6_7", list(range(8))))
check("full-array phase label stays compact", len(calls[1][0]) < 64)

for bad in ([], [1, 1], [0, 8]):
    try:
        phase(bad)
    except ValueError:
        check(f"invalid sensor set rejected: {bad}", True)
    else:
        check(f"invalid sensor set rejected: {bad}", False)

# 2. At a deployment rung the selected architecture and carried architecture
# both need same-architecture 2-vs-8 controls. Otherwise sensor count is
# confounded with architecture.
control_ns = extracted(
    "_control_comparator_keys",
    namespace={
        "STAGE": "s16_all",
        "MULTI_ARCH_PAIR_SELECTION_STAGES": {"s0_scour", "s16_all", "s23_all4"},
        "ALL_ARCHITECTURES": [
            {"name_short": name}
            for name in ("A", "B", "C", "D")
        ],
        "CHAMPION_ARCH": "CARRIED",
        "CONTROL_SETS": [list(range(8))],
    },
)
control_keys = control_ns["_control_comparator_keys"]("WINNER")
expected = {
    (name, tuple(range(8))) for name in ("A", "B", "C", "D")
}
check("full-factorial controls cover every trained architecture",
      control_keys == expected)
same_ns = extracted(
    "_control_comparator_keys",
    namespace={
        "STAGE": "s11_bear",
        "MULTI_ARCH_PAIR_SELECTION_STAGES": {"s0_scour", "s16_all", "s23_all4"},
        "ALL_ARCHITECTURES": [{"name_short": "WINNER"}],
        "CHAMPION_ARCH": "WINNER",
        "CONTROL_SETS": [list(range(8))],
    },
)
check("identical winner/carried control is deduplicated",
      same_ns["_control_comparator_keys"]("WINNER")
      == {("WINNER", tuple(range(8)))})

same_pair_ns = extracted(
    "_same_pair_architecture_keys",
    namespace={
        "STAGE": "s0_scour",
        "MULTI_ARCH_PAIR_SELECTION_STAGES": {"s0_scour"},
        "ALL_ARCHITECTURES": [
            {"name_short": name} for name in ("A", "B", "C", "D")
        ],
    },
)
check("selected pair is compared under every architecture",
      same_pair_ns["_same_pair_architecture_keys"]([3, 1])
      == {(name, (1, 3)) for name in ("A", "B", "C", "D")})
check("s0 pair phase is full architecture x placement factorial",
      "if STAGE in MULTI_ARCH_PAIR_SELECTION_STAGES:" in SOURCE
      and "ARCHITECTURES = ALL_ARCHITECTURES" in SOURCE)

# 3. Median leaderboards preserve the all-array row and its sensor count.
with tempfile.TemporaryDirectory(prefix="campaign-control-") as td:
    lb_ns = extracted(
        "_write_rows_csv", "_median_leaderboard",
        namespace={
            "np": np,
            "os": os,
            "csv": csv,
            "SEEDS": [42, 1337, 2026],
            "SUMMARY_DIR": td,
        },
    )
    rows = []
    for seed, value in zip([42, 1337, 2026], [3.0, 2.0, 4.0]):
        rows.append({
            "phase": "control", "architecture": "WINNER",
            "dofs": "+".join(f"DOF{i}" for i in range(8)),
            "seed": seed, "n_sensors": 8, "inner_val_mse": value,
        })
    agg = lb_ns["_median_leaderboard"](rows, "control.csv")
    check("8-DOF control survives median leaderboard",
          len(agg) == 1 and agg[0]["n_sensors"] == 8
          and agg[0]["n_seeds"] == 3 and agg[0]["inner_val_mse"] == 3.0)

# 4. The same pre-test completeness guard works for an arbitrary-length control
# key; it must hard-fail if even one required seed/weight is missing.
with tempfile.TemporaryDirectory(prefix="campaign-preflight-") as td:
    configs = []
    for seed in [42, 1337, 2026]:
        name = f"ctrl-{seed}"
        weight_dir = Path(td, name)
        weight_dir.mkdir()
        Path(weight_dir, "DT_champion_weights.pth").write_bytes(b"fixture")
        configs.append({
            "name_short": "WINNER", "dofs": list(range(8)),
            "seed": seed, "name": name,
        })
    fake_optuna = SimpleNamespace(load_study=lambda **_: object())
    pre_ns = extracted(
        "_preflight_comparators",
        namespace={
            "_RAN_PHASES": [{
                "configs": configs, "output_dir": td, "db": "fixture",
            }],
            "IDX_TO_DOF_NAME": names,
            "SEEDS": [42, 1337, 2026],
            "N_TRIALS": 100,
            "optuna": fake_optuna,
            "_study_is_finished": lambda *args, **kwargs: True,
            "verify_digital_twin_package": lambda study, cfg, out: {},
            "os": os,
        },
    )
    comparator = {("WINNER", tuple(range(8)))}
    try:
        pre_ns["_preflight_comparators"](comparator)
    except Exception as err:
        print(f"    unexpected {type(err).__name__}: {err}")
        check("8-DOF preflight accepts complete seed/weight matrix", False)
    else:
        check("8-DOF preflight accepts complete seed/weight matrix", True)
    Path(td, "ctrl-2026", "DT_champion_weights.pth").unlink()
    try:
        pre_ns["_preflight_comparators"](comparator)
    except RuntimeError:
        check("8-DOF preflight rejects missing seed weight", True)
    else:
        check("8-DOF preflight rejects missing seed weight", False)

# 5. RUN_TAG must isolate every top-level mutable selection artifact, not only
# studies and summaries.
tag_ns = extracted("_with_run_tag", namespace={"RUN_TAG": ""})
check("empty RUN_TAG preserves first-run artifact name",
      tag_ns["_with_run_tag"]("artifact") == "artifact")
tag_ns = extracted("_with_run_tag", namespace={"RUN_TAG": "replicate2"})
check("nonempty RUN_TAG suffixes artifact name",
      tag_ns["_with_run_tag"]("artifact") == "artifact_replicate2")
check("summary path uses RUN_TAG helper",
      '_with_run_tag(f"{STAGE}_summary_ph-{PROTOCOL_HASH_SHORT}")' in SOURCE)
check("champion manifest requires a durable absolute environment path",
      'CHAMPION_MANIFEST = os.environ.get("CHAMPION_MANIFEST", "")' in SOURCE
      and '"CHAMPION_MANIFEST", CHAMPION_MANIFEST' in SOURCE
      and "path.is_absolute()" in SOURCE)
check("deployment manifest uses RUN_TAG helper",
      "_with_run_tag(\n                f\"_deployment_selection_" in SOURCE)
check("control architecture policy is protocol-hashed",
      '"control_arch_policy"' in
      Path(__file__).with_name("core").joinpath("protocol.py")
      .read_text(encoding="utf-8"))

# 6. Bundle contents come from regular blobs in a reviewed Git tree. Exercise
# the actual importable builder against disposable repositories; never invoke
# it against this repository's stale bundles.
builder = Path(__file__).with_name("build_stage_bundles.py")
source_manifest = Path(__file__).with_name("bundle_source_files.txt")
required_new = {
    "analyze_cross_rung_contrasts.py",
    "bundle_source_files.txt",
    "benchmark_r5_compute.py",
    "check_capacity_preflight.py",
    "check_cross_rung_inference.py",
    "check_hyperparameter_policy.py",
    "scour_MATLAB/B54_TrackVectors.m",
    "core/campaign_contract.py",
    "core/capacity_preflight.py",
    "core/cross_rung_inference.py",
    "core/hyperparameter_policy.py",
    "core/statistical_inference.py",
    "check_paa.py",
    "check_weighted_head_mse.py",
    "check_sensor_noise_pairing.py",
    "check_campaign_controls.py",
    "scour_MATLAB/smoke_b54_overlap_parity.m",
    "check_b54_overlap_parity.py",
    "check_statistical_inference.py",
    "check_artifact_provenance.py",
    "core/environment.py",
    "environment/campaign-py313-cu128.json",
    "requirements-campaign-py313-cu128.txt",
    "check_environment_lock.py",
    "check_execution_blocking.py",
    "check_familytable_roundtrip.py",
    "check_generation_contract.py",
    "check_generation_release_comparison.py",
    "check_source_provenance.py",
    "check_benchmark_contract.py",
    "check_r4_mutation_guards.py",
    "check_training_policy_mutation_guards.py",
    "compare_generation_releases.py",
    "make_micro_smoke.py",
    "TTBI_2D/b54_model_matrices.py",
    "scour_MATLAB/contact_closure_study.m",
    "scour_MATLAB/current_matlab_environment.m",
    "scour_MATLAB/generator_source_root.m",
    "scour_MATLAB/matlab_environment_identity.m",
    "scour_MATLAB/save_progress.m",
    "scour_MATLAB/smoke_contact_closure.m",
    "scour_MATLAB/smoke_crn_state_design.m",
    "scour_MATLAB/smoke_familytable.m",
    "scour_MATLAB/smoke_r11_provenance_serialization.m",
    "core/artifact_provenance.py",
    "core/execution_environment.py",
    "core/source_provenance.py",
    "docs/audit_r4_results.md",
    "docs/audit_r5_results.md",
}
if builder.is_file():
    builder_text = builder.read_text(encoding="utf-8")
    manifest_files = {
        line.strip() for line in source_manifest.read_text(
            encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    check("bundle manifest includes every required runtime/test file",
          required_new <= manifest_files)
    check("every required bundle-manifest file exists",
          all((Path(__file__).resolve().parent / Path(p)).is_file()
              for p in required_new))
    tracked_result = subprocess.run(
        ["git", "ls-files", "--stage", "--", *sorted(required_new)],
        cwd=Path(__file__).resolve().parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    tracked_rows = tracked_result.stdout.decode(
        "utf-8", errors="replace"
    ).splitlines()
    tracked_regular = {
        row.split("\t", 1)[1]
        for row in tracked_rows
        if "\t" in row and row.split(" ", 1)[0] in {"100644", "100755"}
    }
    check("every required runtime/test file is a regular tracked Git blob",
          tracked_result.returncode == 0 and required_new <= tracked_regular)
    check("bundle source manifest is tracked and explicit",
          "bundle_source_files.txt" in builder_text
          and "multidamage_stage2_bundle.zip" not in builder_text)
    check("bundle builder is import-safe",
          'if __name__ == "__main__":' in builder_text)
    check("bundle README separates dispatch checks from audit-only compute",
          "FAST DISPATCH PREFLIGHTS" in builder_text
          and "smoke_r11_provenance_serialization" in builder_text
          and "check_source_provenance.py" in builder_text
          and "check_benchmark_contract.py" in builder_text
          and "AUDIT-ONLY" in builder_text
          and "check_r4_mutation_guards.py" in builder_text
          and "check_training_policy_mutation_guards.py" in builder_text
          and "benchmark_r5_compute.py" in builder_text)
    check("bundle README classifies one-time cross-language gates",
          "ONE-TIME CROSS-LANGUAGE CONTRACT AUDIT" in builder_text
          and "smoke_familytable" in builder_text
          and "check_familytable_roundtrip.py" in builder_text
          and "missing genuine-MATLAB artifact is a hard failure" in
          builder_text
          and "MANDATORY after the first R11 state" in builder_text
          and "smoke_raw_parity" in builder_text
          and "check_raw_parity.py" in builder_text)
    check("bundle build requires an explicit legacy-named R11 authorization",
          "docs/audit_r5_results.md" in builder_text
          and "# Audit R11 results (legacy filename)" in builder_text
          and "**Status: DISPATCH AUTHORIZED.**" in builder_text
          and "_parse_authorized_report" in builder_text)
    check("bundle build binds evidence commit B to tested source commit A",
          "**Tested source commit:**" in builder_text
          and "merge-base" in builder_text
          and "--is-ancestor" in builder_text
          and "--name-only" in builder_text
          and "evidence_diff != [AUDIT_REPORT_NAME]" in builder_text)
    check("bundle payloads are read by immutable HEAD object ID",
          '"ls-tree"' in builder_text and '"cat-file", "blob"' in builder_text
          and "MappingProxyType(blobs)" in builder_text)
    check("builder and manifest bypass Git index stat flags",
          '"hash-object"' in builder_text
          and "_assert_worktree_head_equivalent" in builder_text
          and '"--literal-pathspecs"' in builder_text)
    check("bundle-set publication is staged and fail-closed",
          "TemporaryDirectory" in builder_text
          and "INVALID_BUILD_IN_PROGRESS" in builder_text
          and "complete_bundle_count" in builder_text)

    def fixture_git(
        repo: Path,
        *args: str,
        input_bytes: bytes | None = None,
        check_return: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            input=input_bytes,
            capture_output=True,
            check=False,
        )
        if check_return and proc.returncode != 0:
            raise RuntimeError(
                f"fixture git {' '.join(args)} failed: "
                + proc.stderr.decode("utf-8", errors="replace")
            )
        return proc

    def fixture_report(
        status: str,
        tested_source: str,
        suffix: str = "",
    ) -> str:
        return (
            "# Audit R11 results (legacy filename)\n\n"
            f"**Status: {status}.**\n\n"
            f"**Tested source commit:** {tested_source}\n"
            f"{suffix}"
        )

    def fixture_commit(repo: Path, message: str) -> str:
        fixture_git(repo, "add", "--all")
        fixture_git(repo, "commit", "-m", message)
        return fixture_git(
            repo, "rev-parse", "HEAD"
        ).stdout.decode("ascii").strip()

    def make_fixture(
        repo: Path,
        *,
        manifest_extras: tuple[str, ...] = (),
        ignored: tuple[str, ...] = (),
        omitted: tuple[str, ...] = (),
        symlink_entry: str | None = None,
        fenced_authorization: bool = False,
    ) -> tuple[str, str]:
        repo.mkdir()
        fixture_git(repo, "init")
        fixture_git(repo, "config", "user.name", "Bundle Guard")
        fixture_git(repo, "config", "user.email", "guard@example.invalid")
        fixture_git(repo, "config", "core.autocrlf", "false")

        (repo / "build_stage_bundles.py").write_bytes(builder.read_bytes())
        (repo / "comprehensive_ablation_multidamage.py").write_text(
            'STAGE = "fixture"\nSENSOR_NOISE = None\n',
            encoding="utf-8",
            newline="\n",
        )
        (repo / "scour_MATLAB").mkdir()
        (repo / "scour_MATLAB" / "A00_Run.m").write_text(
            "STAGE = 'fixture';\n", encoding="utf-8", newline="\n"
        )
        (repo / "docs").mkdir()
        fence = (
            "\n```text\n**Status: DISPATCH AUTHORIZED.**\n```\n"
            if fenced_authorization else ""
        )
        (repo / "docs" / "audit_r5_results.md").write_text(
            fixture_report("DISPATCH BLOCKED", "PENDING", fence),
            encoding="utf-8",
            newline="\n",
        )
        (repo / "payload_assume.txt").write_bytes(b"HEAD assume payload\n")
        (repo / "payload_skip.txt").write_bytes(b"HEAD skip payload\n")
        for name in manifest_extras:
            path = repo.joinpath(*name.split("/"))
            if name not in omitted and name != symlink_entry:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"fixture {name}\n".encode("utf-8"))
        if ignored:
            (repo / ".gitignore").write_text(
                "".join(f"/{name}\n" for name in ignored),
                encoding="utf-8",
                newline="\n",
            )
        names_in_fixture = sorted({
            "comprehensive_ablation_multidamage.py",
            "docs/audit_r5_results.md",
            "payload_assume.txt",
            "payload_skip.txt",
            "scour_MATLAB/A00_Run.m",
            *manifest_extras,
        })
        (repo / "bundle_source_files.txt").write_text(
            "# fixture manifest\n" + "\n".join(names_in_fixture) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        fixture_git(repo, "add", "--all")
        for name in omitted:
            fixture_git(
                repo, "rm", "--cached", "--ignore-unmatch", "--", name
            )
        if symlink_entry is not None:
            target_oid = fixture_git(
                repo, "hash-object", "-w", "--stdin",
                input_bytes=b"payload_skip.txt",
            ).stdout.decode("ascii").strip()
            fixture_git(
                repo, "update-index", "--add", "--cacheinfo",
                f"120000,{target_oid},{symlink_entry}",
            )
        fixture_git(repo, "commit", "-m", "tested source A")
        tested_a = fixture_git(
            repo, "rev-parse", "HEAD"
        ).stdout.decode("ascii").strip()
        branch = fixture_git(
            repo, "branch", "--show-current"
        ).stdout.decode("utf-8").strip()
        return tested_a, branch

    def authorize_fixture(
        repo: Path, tested_source: str
    ) -> str:
        (repo / "docs" / "audit_r5_results.md").write_text(
            fixture_report(
                "DISPATCH AUTHORIZED", f"`{tested_source}`"
            ),
            encoding="utf-8",
            newline="\n",
        )
        return fixture_commit(repo, "evidence-only dispatch B")

    def rejects_plan(bundle_module, repo: Path) -> bool:
        try:
            bundle_module.prepare_bundle_plan(repo)
        except bundle_module.BundleBuildError:
            return True
        return False

    # Import itself must not inspect Git or mutate any real ZIP/manifest.
    def publication_snapshot(root: Path) -> dict[str, tuple[str, int]]:
        paths = {
            *root.glob("bundle_*.zip"),
            root / "bundle_sha256.txt",
            root / "bundle_sha256.txt.INVALID_BUILD_IN_PROGRESS",
        }
        return {
            path.name: (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
            for path in paths if path.is_file()
        }

    before_import = publication_snapshot(builder.parent)
    spec = importlib.util.spec_from_file_location(
        "_bundle_builder_contract", builder
    )
    bundle_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bundle_module
    spec.loader.exec_module(bundle_module)
    after_import = publication_snapshot(builder.parent)
    check("importing bundle builder has zero publication side effects",
          before_import == after_import)
    check(
        "registered bundle stages are immutable",
        isinstance(bundle_module.STAGES, type(MappingProxyType({}))),
    )
    try:
        bundle_module.STAGES["fixture"] = ("all_mult", "fixture")
    except TypeError:
        stages_immutable = True
    else:
        stages_immutable = False
    check("registered bundle stage mapping rejects mutation", stages_immutable)
    check(
        "bundle publisher exposes no partial-stage parameter",
        tuple(inspect.signature(bundle_module.build_bundles).parameters)
        == ("repo",),
    )
    generated_readme = bundle_module.readme(
        "s0_scour", "all_mult", "fixture", "b" * 40, "a" * 40
    )
    readme_sections = (
        "FAST DISPATCH PREFLIGHTS",
        "AUDIT-ONLY",
        "ONE-TIME CROSS-LANGUAGE CONTRACT AUDIT",
    )
    check(
        "generated bundle README preserves the three audit classes in order",
        all(section in generated_readme for section in readme_sections)
        and [
            generated_readme.index(section) for section in readme_sections
        ] == sorted(
            generated_readme.index(section) for section in readme_sections
        ),
    )
    check(
        "generated bundle README records authorized A/B benchmark evidence",
        "Dispatch bundle commit B: `" + "b" * 40 + "`" in generated_readme
        and "Tested source commit A: `" + "a" * 40 + "`" in generated_readme
        and "has already approved the genuine" in generated_readme
        and "`attempt_count=1`" in generated_readme
        and "`prior_unaccepted_attempt_count=0`" in generated_readme
        and "`timing_complete=true`" in generated_readme
        and "`memory_complete=true`" in generated_readme
        and "execution is still pending" not in generated_readme
        and "do not authorize dispatch" not in generated_readme,
    )
    fast_section = generated_readme.split(
        "### AUDIT-ONLY", 1
    )[0]
    audit_section = generated_readme.split(
        "### AUDIT-ONLY", 1
    )[1].split("### ONE-TIME CROSS-LANGUAGE", 1)[0]
    check(
        "campaign-control Git audit is audit-only, not a dispatch-host preflight",
        "check_campaign_controls.py" not in fast_section
        and "check_campaign_controls.py" in audit_section,
    )
    parser_fixture_sha = "a" * 40
    check(
        "authorization parser accepts the exact R11 legacy heading",
        bundle_module._parse_authorized_report(
            fixture_report(
                "DISPATCH AUTHORIZED", f"`{parser_fixture_sha}`"
            )
        ) == parser_fixture_sha,
    )
    try:
        bundle_module._parse_authorized_report(
            fixture_report(
                "DISPATCH AUTHORIZED", f"`{parser_fixture_sha}`"
            ).replace(
                "# Audit R11 results (legacy filename)",
                "# Audit R5 results — stale heading",
                1,
            )
        )
    except bundle_module.BundleBuildError:
        stale_heading_rejected = True
    else:
        stale_heading_rejected = False
    check("stale R5 authorization heading is rejected", stale_heading_rejected)
    check(
        "generated bundle README authenticates every follower block reference",
        "`CHAMPION_MANIFEST`" in generated_readme
        and "`TTBI_BLOCK_REFERENCE_SHA256=<64-lowercase-hex>`" in
        generated_readme
        and "export `TTBI_BLOCK_REFERENCE_SHA256` with that exact value" in
        generated_readme
        and "mandatory for followers" in generated_readme,
    )
    check(
        "generated bundle README makes genuine MATLAB and first-state parity hard gates",
        "missing genuine-MATLAB artifact is a hard failure" in generated_readme
        and "MANDATORY after the first R11 state" in generated_readme
        and "`smoke_raw_parity('Results/<case>')`" in generated_readme
        and "`python check_raw_parity.py" in generated_readme,
    )
    for label, payload in (
        ("whitespace-only manifest line", b"alpha.py\n   \nbeta.py\n"),
        ("indented manifest comment", b"alpha.py\n  # hidden\nbeta.py\n"),
    ):
        try:
            bundle_module._parse_source_manifest(payload)
        except bundle_module.BundleBuildError:
            check(f"builder rejects {label}", True)
        else:
            check(f"builder rejects {label}", False)

    # Blocked status — even accompanied by a tempting marker in a code fence —
    # must leave pre-existing ZIP and completion marker byte-for-byte untouched.
    with tempfile.TemporaryDirectory(prefix="bundle-blocked-") as td:
        repo = Path(td, "repo")
        make_fixture(repo, fenced_authorization=True)
        sentinel_zip = repo / "bundle_s0_scour.zip"
        sentinel_manifest = repo / "bundle_sha256.txt"
        sentinel_zip.write_bytes(b"old zip sentinel")
        sentinel_manifest.write_bytes(b"old manifest sentinel")
        before = publication_snapshot(repo)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                bundle_module.build_bundles(
                    repo,
                )
        except bundle_module.BundleBuildError:
            blocked = True
        else:
            blocked = False
        after = publication_snapshot(repo)
        check("blocked/fenced authorization causes zero ZIP mutation",
              blocked and before == after)
        check("status marker inside code fence is never authorization",
              rejects_plan(bundle_module, repo))

    # The intended A -> evidence-only B flow must pass and package immutable
    # HEAD bytes even if index flags hide malicious working-copy substitutions.
    with tempfile.TemporaryDirectory(prefix="bundle-authorized-") as td:
        repo = Path(td, "repo")
        tested_a, _ = make_fixture(repo)
        dispatch_b = authorize_fixture(repo, tested_a)
        (repo / "payload_skip.txt").write_bytes(b"MUTATED skip working copy\n")
        (repo / "payload_assume.txt").write_bytes(
            b"MUTATED assume working copy\n"
        )
        fixture_git(
            repo, "update-index", "--skip-worktree", "payload_skip.txt"
        )
        fixture_git(
            repo, "update-index", "--assume-unchanged", "payload_assume.txt"
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                built = bundle_module.build_bundles(
                    repo,
                )
        except Exception as err:
            print(f"    unexpected {type(err).__name__}: {err}")
            authorized_passed = False
            head_bytes_preserved = False
        else:
            authorized_passed = (
                built.source_commit == dispatch_b
                and len(built.bundles) == 10
                and {path.name for path in built.bundles} == {
                    f"bundle_{stage}.zip"
                    for stage in bundle_module.EXPECTED_STAGE_ORDER
                }
                and built.sha_manifest.is_file()
            )
            with zipfile.ZipFile(built.bundles[0]) as archive:
                head_bytes_preserved = all(
                    archive.read(name) == fixture_git(
                        repo, "cat-file", "blob", f"HEAD:{name}"
                    ).stdout
                    for name in ("payload_skip.txt", "payload_assume.txt")
                )
        check("authorized report-only B over tested A builds", authorized_passed)
        check("bundle bytes equal HEAD despite hidden working-copy changes",
              head_bytes_preserved)

    # A committed runtime delta after A cannot be laundered through a report
    # that still names A.
    with tempfile.TemporaryDirectory(prefix="bundle-runtime-delta-") as td:
        repo = Path(td, "repo")
        tested_a, _ = make_fixture(repo)
        authorize_fixture(repo, tested_a)
        (repo / "payload_skip.txt").write_bytes(b"post-test code delta\n")
        fixture_commit(repo, "unaudited runtime delta")
        check("runtime/code delta after tested A is rejected",
              rejects_plan(bundle_module, repo))
    with tempfile.TemporaryDirectory(prefix="bundle-dirty-source-") as td:
        repo = Path(td, "repo")
        tested_a, _ = make_fixture(repo)
        authorize_fixture(repo, tested_a)
        (repo / "payload_skip.txt").write_bytes(b"ordinary dirty source\n")
        check("ordinary dirty runtime source is rejected",
              rejects_plan(bundle_module, repo))

    # Invalid and real-but-nonancestor tested SHAs are distinct failure modes.
    with tempfile.TemporaryDirectory(prefix="bundle-invalid-sha-") as td:
        repo = Path(td, "repo")
        make_fixture(repo)
        authorize_fixture(repo, "0" * 40)
        check("invalid tested SHA is rejected",
              rejects_plan(bundle_module, repo))
    with tempfile.TemporaryDirectory(prefix="bundle-nonancestor-") as td:
        repo = Path(td, "repo")
        tested_a, main_branch = make_fixture(repo)
        fixture_git(repo, "checkout", "-b", "side-evidence")
        (repo / "side-only.txt").write_bytes(b"side branch\n")
        side_commit = fixture_commit(repo, "side commit")
        fixture_git(repo, "checkout", main_branch)
        authorize_fixture(repo, side_commit)
        check("nonancestor tested SHA is rejected",
              rejects_plan(bundle_module, repo))

    # A path named by the manifest must exist as a regular blob in HEAD. This
    # catches ordinary untracked files, ignored files, and Git symlink entries.
    for label, ignored, omitted in (
        ("untracked", (), ("untracked.txt",)),
        ("ignored untracked", ("ignored.txt",), ("ignored.txt",)),
    ):
        with tempfile.TemporaryDirectory(
            prefix=f"bundle-{label.replace(' ', '-')}-"
        ) as td:
            repo = Path(td, "repo")
            tested_a, _ = make_fixture(
                repo,
                manifest_extras=omitted,
                ignored=ignored,
                omitted=omitted,
            )
            authorize_fixture(repo, tested_a)
            check(f"{label} manifest entry is rejected",
                  rejects_plan(bundle_module, repo))
    with tempfile.TemporaryDirectory(prefix="bundle-symlink-") as td:
        repo = Path(td, "repo")
        tested_a, _ = make_fixture(
            repo,
            manifest_extras=("linked.txt",),
            symlink_entry="linked.txt",
        )
        authorize_fixture(repo, tested_a)
        check("symlink manifest entry is rejected",
              rejects_plan(bundle_module, repo))

    # Canonical names are a cross-platform extraction invariant.
    with tempfile.TemporaryDirectory(prefix="bundle-noncanonical-") as td:
        repo = Path(td, "repo")
        tested_a, _ = make_fixture(repo)
        manifest = repo / "bundle_source_files.txt"
        lines = manifest.read_text(encoding="utf-8").splitlines()
        lines.append("z//noncanonical.txt")
        manifest.write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )
        fixture_commit(repo, "noncanonical manifest")
        check("noncanonical manifest entry is rejected",
              rejects_plan(bundle_module, repo))

    # The two control files themselves are executed/read from the worktree, so
    # they get an extra content-hash guard that index flags cannot suppress.
    for control_name, index_flag in (
        ("build_stage_bundles.py", "--assume-unchanged"),
        ("bundle_source_files.txt", "--skip-worktree"),
    ):
        with tempfile.TemporaryDirectory(prefix="bundle-control-copy-") as td:
            repo = Path(td, "repo")
            tested_a, _ = make_fixture(repo)
            authorize_fixture(repo, tested_a)
            with (repo / control_name).open(
                "ab"
            ) as stream:
                stream.write(b"\n# hidden working-copy mutation\n")
            fixture_git(repo, "update-index", index_flag, control_name)
            check(f"hidden {control_name} mutation is rejected",
                  rejects_plan(bundle_module, repo))
else:
    # Stage bundles intentionally do not ship their own bundle builder/source
    # ZIP. This Git/publication branch is audit-only and therefore not
    # applicable inside a dispatched stage bundle; runtime controls above
    # remain applicable.
    print("  [N/A] audit-only bundle-manifest checks (builder absent)")

# ── Pre-registered PRIMARY ESTIMAND (audit r5) ───────────────────────────────
# The objective policy is structured executable data in TRAIN_PROTOCOL, not a
# prose declaration. The trainer passes that hashed mapping into the generic,
# fail-closed selector. These checks cover policy behaviour, malformed/missing
# inputs, and the actual trainer call site.
from core import task as _task                                   # noqa: E402
from core.utils import DETERMINISM_POLICY as _DETERMINISM_POLICY # noqa: E402
from training.trainer import (                                  # noqa: E402
    TRAIN_PROTOCOL as _TRAIN_PROTOCOL,
    resolve_trial_seed as _resolve_trial_seed,
)

_objective_policy = _TRAIN_PROTOCOL["objective"]
_bearing_config = {
    "task": "regression",
    "target_supports": [2, 3],
    "bearing_targets": ["left", "right"],
}
_scour_config = {"task": "regression", "target_supports": [2, 3]}
_classification_config = {"task": "classification", "discretization": 1}

# Bearing rung: scour_mse and the aggregate disagree -> policy must select SCOUR.
_bearing_metrics = {"mse": 4.0, "scour_mse": 1.0, "bearing_mse": 7.0}
check("objective is SCOUR-primary when bearing heads exist",
      _task.objective_value(
          _bearing_metrics, _bearing_config, _objective_policy) == 1.0)
check("objective ignores the all-head aggregate on bearing rungs",
      _task.objective_value(
          _bearing_metrics, _bearing_config, _objective_policy)
      != _bearing_metrics["mse"])
# Scour-only regression and classification explicitly select aggregate MSE.
check("objective selects aggregate MSE on scour-only regression",
      _task.objective_value(
          {"mse": 2.5}, _scour_config, _objective_policy) == 2.5)
check("objective selects aggregate MSE on classification",
      _task.objective_value(
          {"mse": 3.5}, _classification_config, _objective_policy) == 3.5)
# A lower scour error must always rank better, whatever bearing does.
check("a scour improvement outranks a bearing improvement",
      _task.objective_value(
          {"mse": 9.0, "scour_mse": 1.0}, _bearing_config, _objective_policy)
      < _task.objective_value(
          {"mse": 2.0, "scour_mse": 3.0}, _bearing_config, _objective_policy))

# Prove the protocol is executable: mutating only its bearing metric changes
# selection behaviour. The separate protocol-hash checker proves it also
# changes the core hash.
_mutated_policy = {
    **_objective_policy,
    "regression_with_bearing_heads": "mse",
}
check("objective behaviour is derived from TRAIN_PROTOCOL policy",
      _task.objective_value(
          _bearing_metrics, _bearing_config, _mutated_policy)
      == _bearing_metrics["mse"])

try:
    _task.objective_value({"mse": 4.0}, _bearing_config, _objective_policy)
except KeyError:
    check("bearing objective fails closed when scour_mse is missing", True)
else:
    check("bearing objective fails closed when scour_mse is missing", False)

try:
    _task.objective_value(
        _bearing_metrics,
        _bearing_config,
        {"default": "mse", "unexpected": "scour_mse"},
    )
except ValueError:
    check("malformed objective policy is rejected", True)
else:
    check("malformed objective policy is rejected", False)

# Structural end-to-end guard: the trial loop must pass the hashed policy as
# the selector's third argument. This catches bypassing objective_value or
# replacing the call site with metrics["mse"].
_trainer_path = Path(__file__).with_name("training").joinpath("trainer.py")
_trainer_tree = ast.parse(
    _trainer_path.read_text(encoding="utf-8"), filename=str(_trainer_path))
_objective_calls = [
    node for node in ast.walk(_trainer_tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and isinstance(node.func.value, ast.Name)
    and node.func.value.id == "task"
    and node.func.attr == "objective_value"
]


def _is_train_protocol_arg(node, key: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "TRAIN_PROTOCOL"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == key
    )


check("trainer consumes the protocol-hashed objective mapping",
      len(_objective_calls) == 1
      and len(_objective_calls[0].args) == 3
      and _is_train_protocol_arg(_objective_calls[0].args[2], "objective"))

# Trial seeding and deterministic execution follow the same single-source
# contract. Missing config seeds fail rather than collapsing independent arms.
check("TRAIN_PROTOCOL embeds the canonical executable determinism policy",
      _TRAIN_PROTOCOL["determinism"] is _DETERMINISM_POLICY)
check("trial seed is resolved from the registered config field",
      _resolve_trial_seed(
          {"seed": 1337}, _TRAIN_PROTOCOL["trial_seed"]) == 1337)
try:
    _resolve_trial_seed({}, _TRAIN_PROTOCOL["trial_seed"])
except KeyError:
    check("missing trial seed fails closed", True)
else:
    check("missing trial seed fails closed", False)
_mutated_seed_policy = {
    **_TRAIN_PROTOCOL["trial_seed"],
    "key": "alternate_seed",
}
check("trial-seed behaviour is derived from TRAIN_PROTOCOL policy",
      _resolve_trial_seed(
          {"seed": 42, "alternate_seed": 2026}, _mutated_seed_policy) == 2026)


def _calls_named(tree, name: str) -> list[ast.Call]:
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


# Every production training path must consume the same structured loss,
# optimizer and scheduler policies: two trainer entry points plus finalist CV.
_driver_fold_tree = FUNCTIONS["_fit_predict_finalist_fold"]
_loss_calls = (
    _calls_named(_trainer_tree, "make_criterion")
    + _calls_named(_driver_fold_tree, "make_criterion")
)
_optimizer_calls = (
    _calls_named(_trainer_tree, "make_optimizer")
    + _calls_named(_driver_fold_tree, "make_optimizer")
)
_scheduler_calls = (
    _calls_named(_trainer_tree, "make_scheduler")
    + _calls_named(_driver_fold_tree, "make_scheduler")
)
check("all production losses consume TRAIN_PROTOCOL loss policy",
      len(_loss_calls) == 3
      and all(len(call.args) >= 2
              and _is_train_protocol_arg(call.args[1], "loss")
              for call in _loss_calls))
check("all production optimizers consume TRAIN_PROTOCOL optimizer policy",
      len(_optimizer_calls) == 3
      and all(len(call.args) >= 3
              and _is_train_protocol_arg(call.args[2], "optimizer")
              for call in _optimizer_calls))
check("all production schedulers consume TRAIN_PROTOCOL scheduler policy",
      len(_scheduler_calls) == 3
      and all(len(call.args) >= 3
              and _is_train_protocol_arg(call.args[2], "scheduler")
              for call in _scheduler_calls))

_pipeline_path = Path(__file__).with_name("training").joinpath("pipeline.py")
_pipeline_tree = ast.parse(
    _pipeline_path.read_text(encoding="utf-8"), filename=str(_pipeline_path))
_seed_calls = (
    _calls_named(_trainer_tree, "set_global_seed")
    + _calls_named(_pipeline_tree, "set_global_seed")
    + _calls_named(TREE, "set_global_seed")
)
check("all production seeding calls consume TRAIN_PROTOCOL determinism policy",
      len(_seed_calls) == 5
      and all(len(call.args) >= 2
              and _is_train_protocol_arg(call.args[1], "determinism")
              for call in _seed_calls))
check("Optuna trainer has no silent config-seed fallback",
      ".get('seed'," not in _trainer_path.read_text(encoding="utf-8")
      and '.get("seed",' not in _trainer_path.read_text(encoding="utf-8"))

print()
print("CAMPAIGN CONTROLS: ALL PASS" if fails == 0
      else f"CAMPAIGN CONTROLS: {fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)

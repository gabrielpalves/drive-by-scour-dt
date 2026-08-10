"""Adversarial checks for content-addressed generator/runtime source roots."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import core.source_provenance as provenance
from core.source_provenance import (
    DRIVER,
    REPO,
    SOURCE_MANIFEST,
    SourceProvenanceError,
    generator_source_root,
    python_runtime_source_root,
    repository_source_snapshot,
)


fails = 0


def check(name: str, condition: bool) -> None:
    global fails
    ok = bool(condition)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    fails += int(not ok)


def rejects(name: str, fn) -> None:
    try:
        fn()
    except SourceProvenanceError:
        check(name, True)
    else:
        check(name, False)


def rejects_message(name: str, fragment: str, fn) -> None:
    try:
        fn()
    except SourceProvenanceError as exc:
        check(name, fragment in str(exc))
    else:
        check(name, False)


def _copy_boundary(destination: Path) -> None:
    manifest = REPO / SOURCE_MANIFEST
    destination.mkdir(parents=True)
    shutil.copy2(manifest, destination / SOURCE_MANIFEST)
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        if (
            name.startswith("scour_MATLAB/")
            or name.endswith(".py")
            or name
            in {
                "environment/campaign-py313-cu128.json",
                "requirements-campaign-py313-cu128.txt",
            }
        ):
            source = REPO.joinpath(*name.split("/"))
            target = destination.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


print("SOURCE PROVENANCE CHECKS")
allowlisted_ast_harness = ast.parse(
    "exec(compile('AUDIT_SLICE = 1', 'audit-slice', 'exec'))\n"
)
check(
    "exact authenticated AST harness may call direct compile/exec",
    provenance._imported_modules(
        allowlisted_ast_harness, "check_campaign_controls.py"
    ) == set(),
)
live_python = python_runtime_source_root()
live_matlab = generator_source_root()
check(
    "live Python runtime root is non-vacuous SHA-256",
    len(live_python.sha256) == 64
    and live_python.file_count >= 20
    and DRIVER in live_python.files,
)
check(
    "live MATLAB generator root is non-vacuous SHA-256",
    len(live_matlab.sha256) == 64
    and live_matlab.file_count >= 40
    and "scour_MATLAB/A00_Run.m" in live_matlab.files,
)

with tempfile.TemporaryDirectory(prefix="source-provenance-") as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    py0 = python_runtime_source_root(fixture)
    mat0 = generator_source_root(fixture)

    driver = fixture / DRIVER
    original_driver = driver.read_bytes()
    manifest_env_mutant = original_driver.replace(
        b'TRAINING_JOB_MANIFEST_ENV = "TTBI_TRAINING_JOB_MANIFEST"',
        b'TRAINING_JOB_MANIFEST_ENV = "TTBI_FOREIGN_JOB_MANIFEST"',
        1,
    )
    check(
        "manifest environment binding is an exact authenticated driver byte",
        manifest_env_mutant != original_driver,
    )
    driver.write_bytes(manifest_env_mutant)
    check(
        "one driver byte moves the Python runtime root",
        python_runtime_source_root(fixture).sha256 != py0.sha256,
    )
    driver.write_bytes(original_driver + b"\nHIDDEN_EFFECT = True\n")
    check(
        "bundle-local executable suffix moves the Python runtime root",
        python_runtime_source_root(fixture).sha256 != py0.sha256,
    )
    driver.write_bytes(original_driver)

    matlab_source = fixture / "scour_MATLAB" / "A01_Train.m"
    matlab_source.write_bytes(matlab_source.read_bytes() + b"\n% mutation\n")
    check(
        "one generator-source byte moves the MATLAB root",
        generator_source_root(fixture).sha256 != mat0.sha256,
    )

with tempfile.TemporaryDirectory(prefix="source-provenance-missing-") as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    (fixture / "core" / "protocol.py").unlink()
    rejects(
        "missing reviewed runtime file fails closed",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(prefix="source-provenance-package-") as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    (fixture / "TTBI_2D" / "unreviewed_oracle.py").write_text(
        "UNREVIEWED = True\n",
        encoding="utf-8",
    )
    rejects(
        "unmanifested regular-package module fails closed",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(prefix="source-provenance-import-") as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    dependency = fixture / "unreviewed_runtime_dependency.py"
    dependency.write_text("UNREVIEWED = True\n", encoding="utf-8")
    importer = fixture / "benchmark_r5_compute.py"
    importer.write_bytes(
        importer.read_bytes() + b"\nimport unreviewed_runtime_dependency\n"
    )
    rejects(
        "ordinary top-level project import outside manifest fails closed",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(prefix="source-provenance-toctou-bytes-") as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    importer = fixture / "benchmark_r5_compute.py"
    original_snapshot = provenance._regular_snapshot
    fired = {"value": False}

    def mutate_after_snapshot(root: Path, name: str):
        snapshot = original_snapshot(root, name)
        if name == "benchmark_r5_compute.py" and not fired["value"]:
            fired["value"] = True
            importer.write_bytes(importer.read_bytes() + b"\nMUTATED = True\n")
        return snapshot

    provenance._regular_snapshot = mutate_after_snapshot
    try:
        rejects(
            "source mutation after AST/hash snapshot fails final reassertion",
            lambda: python_runtime_source_root(fixture),
        )
    finally:
        provenance._regular_snapshot = original_snapshot
    check("source-byte TOCTOU probe fired", fired["value"])

with tempfile.TemporaryDirectory(
    prefix="source-provenance-coherent-hybrid-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    early_source = fixture / "core" / "dataset.py"
    original_snapshot = provenance._regular_snapshot
    fired = {"value": False}

    def mutate_earlier_member_during_capture(root: Path, name: str):
        captured = original_snapshot(root, name)
        if name == "scour_MATLAB/A00_Run.m" and not fired["value"]:
            fired["value"] = True
            early_source.write_bytes(
                early_source.read_bytes() + b"\n_HYBRID_PROBE = True\n"
            )
        return captured

    provenance._regular_snapshot = mutate_earlier_member_during_capture
    try:
        rejects_message(
            "coherent policy capture rejects a cross-file hybrid tree",
            "changed during root calculation",
            lambda: repository_source_snapshot(fixture),
        )
    finally:
        provenance._regular_snapshot = original_snapshot
    check("coherent hybrid-source probe fired", fired["value"])

with tempfile.TemporaryDirectory(
    prefix="source-provenance-snapshot-aba-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    captured = repository_source_snapshot(fixture)
    source = fixture / "core" / "dataset.py"
    replacement = fixture / "core" / "dataset.py.aba-replacement"
    replacement.write_bytes(source.read_bytes())
    os.replace(replacement, source)
    rejects_message(
        "same-byte pathname replacement is rejected by retained identity",
        "changed during root calculation",
        captured.assert_unchanged,
    )

with tempfile.TemporaryDirectory(
    prefix="source-provenance-toctou-inventory-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    original_inventory = provenance._validate_matlab_source_inventory
    fired = {"value": False}

    def mutate_after_inventory(root: Path, manifest_names: set[str]) -> None:
        original_inventory(root, manifest_names)
        if not fired["value"]:
            fired["value"] = True
            (root / "scour_MATLAB" / "late_unreviewed.m").write_text(
                "function y = late_unreviewed(); y = 1; end\n",
                encoding="utf-8",
            )

    provenance._validate_matlab_source_inventory = mutate_after_inventory
    try:
        rejects(
            "MATLAB inventory mutation during root calculation fails closed",
            lambda: python_runtime_source_root(fixture),
        )
    finally:
        provenance._validate_matlab_source_inventory = original_inventory
    check("source-inventory TOCTOU probe fired", fired["value"])

with tempfile.TemporaryDirectory(
    prefix="source-provenance-toctou-import-target-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    importer = fixture / "benchmark_r5_compute.py"
    importer.write_bytes(
        importer.read_bytes() + b"\nimport late_unreviewed_dependency\n"
    )
    dependency = fixture / "late_unreviewed_dependency.py"
    original_closure = provenance._validate_python_import_closure
    fired = {"value": False}

    def add_import_target_after_closure(
        root: Path,
        manifest_names: set[str],
        snapshots: dict,
    ) -> None:
        original_closure(root, manifest_names, snapshots)
        if not fired["value"]:
            fired["value"] = True
            dependency.write_text("UNREVIEWED = True\n", encoding="utf-8")

    provenance._validate_python_import_closure = add_import_target_after_closure
    try:
        rejects(
            "late import target is caught by final AST-closure reassertion",
            lambda: python_runtime_source_root(fixture),
        )
    finally:
        provenance._validate_python_import_closure = original_closure
    check("import-target TOCTOU probe fired", fired["value"])

with tempfile.TemporaryDirectory(prefix="source-provenance-dynamic-") as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    dependency = fixture / "unreviewed_dynamic_dependency.py"
    dependency.write_text("UNREVIEWED = True\n", encoding="utf-8")
    importer = fixture / "benchmark_r5_compute.py"
    importer.write_bytes(
        importer.read_bytes()
        + b"\nimport importlib\n"
        + b"importlib.import_module('unreviewed_dynamic_dependency')\n"
    )
    rejects(
        "literal dynamic project import outside manifest fails closed",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(
    prefix="source-provenance-dynamic-alias-keyword-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    dependency = fixture / "unreviewed_alias_dependency.py"
    dependency.write_text("UNREVIEWED = True\n", encoding="utf-8")
    importer = fixture / "benchmark_r5_compute.py"
    injection = (
        b"\nimport importlib as project_loader\n"
        b"project_loader.import_module("
        b"name='unreviewed_alias_dependency')\n"
    )
    check("alias/keyword dynamic-import probe is nonempty", bool(injection))
    importer.write_bytes(importer.read_bytes() + injection)
    rejects(
        "importlib alias with keyword name cannot escape the manifest",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(
    prefix="source-provenance-dynamic-from-alias-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    dependency = fixture / "unreviewed_from_alias.py"
    dependency.write_text("UNREVIEWED = True\n", encoding="utf-8")
    importer = fixture / "benchmark_r5_compute.py"
    injection = (
        b"\nfrom importlib import import_module as load_project_module\n"
        b"load_project_module(name='unreviewed_from_alias')\n"
    )
    check("from-alias dynamic-import probe is nonempty", bool(injection))
    importer.write_bytes(importer.read_bytes() + injection)
    rejects(
        "from-import alias with keyword name cannot escape the manifest",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(
    prefix="source-provenance-dynamic-assignment-alias-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    dependency = fixture / "unreviewed_assignment_alias.py"
    dependency.write_text("UNREVIEWED = True\n", encoding="utf-8")
    importer = fixture / "benchmark_r5_compute.py"
    injection = (
        b"\nimport importlib as loader_library\n"
        b"load_project_module = loader_library.import_module\n"
        b"load_project_module(name='unreviewed_assignment_alias')\n"
    )
    check("assignment-alias dynamic-import probe is nonempty", bool(injection))
    importer.write_bytes(importer.read_bytes() + injection)
    rejects(
        "assigned import_module alias cannot escape the manifest",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(
    prefix="source-provenance-dynamic-relative-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    package = fixture / "unreviewed_dynamic_package"
    package.mkdir()
    (package / "dependency.py").write_text(
        "UNREVIEWED = True\n", encoding="utf-8"
    )
    importer = fixture / "benchmark_r5_compute.py"
    injection = (
        b"\nimport importlib as relative_loader\n"
        b"relative_loader.import_module("
        b"name='.dependency', package='unreviewed_dynamic_package')\n"
    )
    check("relative dynamic-import probe is nonempty", bool(injection))
    importer.write_bytes(importer.read_bytes() + injection)
    rejects(
        "literal relative dynamic import resolves before manifest closure",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(
    prefix="source-provenance-dynamic-relative-from-alias-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    package = fixture / "unreviewed_relative_alias_package"
    package.mkdir()
    (package / "dependency.py").write_text(
        "UNREVIEWED = True\n", encoding="utf-8"
    )
    importer = fixture / "benchmark_r5_compute.py"
    injection = (
        b"\nfrom importlib import import_module as load_relative_module\n"
        b"load_relative_module("
        b"name='.dependency', package='unreviewed_relative_alias_package')\n"
    )
    check("relative from-alias import probe is nonempty", bool(injection))
    importer.write_bytes(importer.read_bytes() + injection)
    rejects(
        "from-import alias preserves relative import_module semantics",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(
    prefix="source-provenance-dynamic-nonliteral-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    importer = fixture / "benchmark_r5_compute.py"
    injection = (
        b"\nimport importlib as guarded_loader\n"
        b"runtime_name = 'math'\n"
        b"guarded_loader.import_module(name=runtime_name)\n"
    )
    check("nonliteral dynamic-import probe is nonempty", bool(injection))
    importer.write_bytes(importer.read_bytes() + injection)
    rejects(
        "nonliteral alias/keyword dynamic import fails closed",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(
    prefix="source-provenance-dynamic-noncanonical-name-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    importer = fixture / "benchmark_r5_compute.py"
    injection = (
        b"\nimport importlib as guarded_loader\n"
        b"guarded_loader.import_module(name='unreviewed-module')\n"
    )
    check("noncanonical module-name probe is nonempty", bool(injection))
    importer.write_bytes(importer.read_bytes() + injection)
    rejects(
        "noncanonical dynamic module names fail closed",
        lambda: python_runtime_source_root(fixture),
    )

for label, injection, diagnostic in (
    (
        "concatenated getattr import_module",
        b"\nimport importlib\n"
        b"getattr(importlib, 'import_' + 'module')('math')\n",
        "reflective getattr",
    ),
    (
        "concatenated importlib.__dict__ subscript",
        b"\nimport importlib\n"
        b"importlib.__dict__['import_' + 'module']('math')\n",
        "__dict__",
    ),
    (
        "import_module factory through functools.partial",
        b"\nimport functools\nimport importlib\n"
        b"factory = functools.partial(importlib.import_module, 'math')\n"
        b"factory()\n",
        "factory/reference",
    ),
):
    with tempfile.TemporaryDirectory(
        prefix="source-provenance-loader-indirection-"
    ) as raw:
        fixture = Path(raw, "repo")
        _copy_boundary(fixture)
        importer = fixture / "benchmark_r5_compute.py"
        check(f"{label} probe is nonempty", len(injection) >= 40)
        importer.write_bytes(importer.read_bytes() + injection)
        rejects_message(
            f"{label} fails at the loader-indirection guard",
            diagnostic,
            lambda: python_runtime_source_root(fixture),
        )

for builtin_name, injection in (
    ("exec", b"\npayload = 'AUDIT_ESCAPE = 1'\nexec(payload)\n"),
    ("eval", b"\npayload = '1 + 1'\neval(payload)\n"),
    (
        "compile",
        b"\npayload = 'AUDIT_ESCAPE = 1'\n"
        b"compile(payload, 'escape', 'exec')\n",
    ),
):
    with tempfile.TemporaryDirectory(
        prefix=f"source-provenance-bare-{builtin_name}-"
    ) as raw:
        fixture = Path(raw, "repo")
        _copy_boundary(fixture)
        production = fixture / "benchmark_r5_compute.py"
        check(
            f"bare {builtin_name} production probe is nonempty",
            f"{builtin_name}(".encode("ascii") in injection
            and len(injection) >= 20,
        )
        production.write_bytes(production.read_bytes() + injection)
        rejects_message(
            f"bare {builtin_name} is unavailable to production sources",
            f"bare {builtin_name}()",
            lambda fixture=fixture: repository_source_snapshot(fixture),
        )

with tempfile.TemporaryDirectory(prefix="source-provenance-matlab-") as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    (fixture / "scour_MATLAB" / "unreviewed_solver.m").write_text(
        "function y = unreviewed_solver(); y = 1; end\n",
        encoding="utf-8",
    )
    rejects(
        "unmanifested MATLAB source outside generated allowlist fails closed",
        lambda: generator_source_root(fixture),
    )

with tempfile.TemporaryDirectory(
    prefix="source-provenance-matlab-generated-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    before = generator_source_root(fixture)
    results = fixture / "scour_MATLAB" / "Results" / "fixture"
    results.mkdir(parents=True)
    (results / "0001.mat").write_bytes(b"generated result")
    sensitivity = (
        fixture
        / "scour_MATLAB"
        / "Results_sensitivity"
        / "dry_ballast_stiffness_sign"
        / "retained-stiffening"
        / "fixture"
    )
    sensitivity.mkdir(parents=True)
    (sensitivity / "0001.mat").write_bytes(b"generated sensitivity result")
    (fixture / "scour_MATLAB" / "micro_A00_smoke.m").write_text(
        "% generated\n",
        encoding="utf-8",
    )
    (
        fixture
        / "scour_MATLAB"
        / "micro_A00_qualification_fixture.m"
    ).write_text("% generated\n", encoding="utf-8")
    check(
        "only generated result trees and micro drivers are excluded",
        generator_source_root(fixture).sha256 == before.sha256,
    )

with tempfile.TemporaryDirectory(prefix="source-provenance-hardlink-") as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    source = fixture / "core" / "protocol.py"
    os.link(source, fixture / "hardlink-probe.bin")
    rejects(
        "multi-link reviewed source file fails closed",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(
    prefix="source-provenance-manifest-hardlink-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    manifest = fixture / SOURCE_MANIFEST
    os.link(manifest, fixture / "manifest-hardlink-probe.txt")
    rejects(
        "multi-link source manifest fails closed",
        lambda: python_runtime_source_root(fixture),
    )

with tempfile.TemporaryDirectory(
    prefix="source-provenance-package-alias-"
) as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    package = fixture / "core"
    package_target = fixture / "core-real"
    package.rename(package_target)
    if os.name == "nt":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(package), str(package_target)],
            check=False,
            capture_output=True,
            text=True,
        ).returncode == 0
    else:
        package.symlink_to(package_target, target_is_directory=True)
        created = True
    check("package junction/symlink fixture was created", created)
    if created:
        rejects(
            "reviewed package junction/symlink fails closed",
            lambda: python_runtime_source_root(fixture),
        )

with tempfile.TemporaryDirectory(prefix="source-provenance-alias-") as raw:
    root = Path(raw)
    fixture = root / "repo"
    _copy_boundary(fixture)
    alias = root / "repo-alias"
    if os.name == "nt":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(fixture)],
            check=False,
            capture_output=True,
            text=True,
        ).returncode == 0
    else:
        alias.symlink_to(fixture, target_is_directory=True)
        created = True
    check("junction/symlink alias fixture was created", created)
    if created:
        rejects(
            "repository junction/symlink canonical alias fails closed",
            lambda: python_runtime_source_root(alias),
        )

with tempfile.TemporaryDirectory(prefix="source-provenance-manifest-") as raw:
    fixture = Path(raw, "repo")
    _copy_boundary(fixture)
    manifest = fixture / SOURCE_MANIFEST
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text + "\ncore/protocol.py\n",
        encoding="utf-8",
        newline="\n",
    )
    rejects(
        "duplicate/noncanonical source manifest fails closed",
        lambda: python_runtime_source_root(fixture),
    )

for label, injected in (
    ("whitespace-only source-manifest line", "   "),
    ("indented source-manifest comment", "  # hidden"),
):
    with tempfile.TemporaryDirectory(
        prefix="source-provenance-manifest-whitespace-"
    ) as raw:
        fixture = Path(raw, "repo")
        _copy_boundary(fixture)
        manifest = fixture / SOURCE_MANIFEST
        text = manifest.read_text(encoding="utf-8")
        manifest.write_text(
            injected + "\n" + text,
            encoding="utf-8",
            newline="\n",
        )
        rejects(
            f"{label} fails closed like the MATLAB parser",
            lambda fixture=fixture: python_runtime_source_root(fixture),
        )

if fails:
    raise SystemExit(f"SOURCE PROVENANCE: {fails} FAILURE(S)")
print("SOURCE PROVENANCE: ALL PASS")

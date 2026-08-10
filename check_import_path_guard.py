"""Behavioral checks for the pre-import campaign boundary.

Every negative subprocess first proves that the corresponding unguarded import
candidate is reachable.  A rejection counts only when the reviewed bootstrap
runs (or deliberately rejects before importing anything) and the probe entry
body remains unexecuted.
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
    ImportBoundaryError,
    enforce_import_boundary as _enforce_import_boundary,
    validate_source_tree,
)
_enforce_import_boundary()

import campaign_import_guard as guard_module  # noqa: E402
import ast
import importlib.machinery
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parent
GUARD_SOURCE = ROOT / "campaign_import_guard" / "__init__.py"
PROJECT_INITIALIZERS = (
    Path("core", "__init__.py"),
    Path("training", "__init__.py"),
    Path("TTBI_2D", "__init__.py"),
)
SOURCE_MANIFEST_NAME = "bundle_source_files.txt"
ENTRYPOINT_MARKERS = {
    "benchmark_r5_compute.py": "import argparse",
    "benchmark_paper1_compute.py": "import argparse",
    "capacity_preflight_compute.py": "import argparse",
    "check_f25_capacity.py": "import argparse",
    "comprehensive_ablation_multidamage.py": "import argparse",
    "compare_generation_releases.py": "import argparse",
    "qualification_receipt_inventory.py": "import argparse",
    "dispatch_authorization.py": "import argparse",
    "build_f25_bundles.py": "import argparse",
    "build_stage_bundles.py": "import argparse",
    "make_micro_smoke.py": "import argparse",
    "check_contact_closure_gate.py": "import argparse",
    "analyze_cross_rung_contrasts.py": "import argparse",
    "check_capacity_preflight.py": "from copy import deepcopy",
    "check_raw_parity.py": "import numpy",
    "training/f25_executor.py": "import argparse",
}
ENTRY_BODY_MARKER = "REVIEWED_ENTRY_RAN"
FAKE_GUARD_MARKER = "FAKE_GUARD_RAN"
STARTUP_MARKER = "STARTUP_CUSTOMIZER_RAN"
PRELOADED_SCIENTIFIC_MARKER = "PRELOADED_SCIENTIFIC_MODULE_WON"

FAILURES = 0


def check(label: str, condition: bool) -> None:
    global FAILURES
    passed = bool(condition)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    FAILURES += int(not passed)


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def _entry_source(pre_guard: str = "") -> str:
    return f'''from __future__ import annotations
import os as _bootstrap_os
import sys as _bootstrap_sys
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{{_unsafe_python_path_variable}} must be absent before imports"
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
{pre_guard}
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
from campaign_import_guard import enforce_import_boundary
enforce_import_boundary()
print("{ENTRY_BODY_MARKER}")
'''


def _make_fixture(root: Path, *, pre_guard: str = "") -> Path:
    guard = root / "campaign_import_guard"
    guard.mkdir(parents=True)
    shutil.copy2(GUARD_SOURCE, guard / "__init__.py")
    for relative in PROJECT_INITIALIZERS:
        target = root / relative
        target.parent.mkdir(parents=True)
        shutil.copy2(ROOT / relative, target)
    reviewed_sources = [
        Path("campaign_import_guard", "__init__.py").as_posix(),
        *(relative.as_posix() for relative in PROJECT_INITIALIZERS),
    ]
    (root / SOURCE_MANIFEST_NAME).write_text(
        "\n".join(sorted(reviewed_sources)) + "\n",
        encoding="utf-8",
    )
    entry = root / "probe_entry.py"
    entry.write_text(_entry_source(pre_guard), encoding="utf-8")
    return entry


def _run_entry(
    entry: Path,
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(entry)],
        cwd=entry.parent,
        env=_clean_environment() if environment is None else environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )


def _combined(process: subprocess.CompletedProcess[str]) -> str:
    return process.stdout + "\n" + process.stderr


def _rejected_before_body(
    process: subprocess.CompletedProcess[str],
    diagnostic: str,
) -> bool:
    output = _combined(process)
    return (
        process.returncode != 0
        and ENTRY_BODY_MARKER not in output
        and diagnostic in output
    )


def _exercise_source_candidate(
    label: str,
    relative: Path,
    *,
    directory: bool = False,
    paired_module: bool = False,
    diagnostic: str,
) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        entry = _make_fixture(root)
        candidate = root / relative
        if directory:
            candidate.mkdir(parents=True)
            (candidate / "__init__.py").write_text(
                "SHADOW = True\n", encoding="utf-8"
            )
            if paired_module:
                candidate.with_suffix(".py").write_text(
                    "REVIEWED = True\n", encoding="utf-8"
                )
        else:
            candidate.write_bytes(b"# competing import candidate\n")
        process = _run_entry(entry)
        check(label, _rejected_before_body(process, diagnostic))


def _make_junction(alias: Path, target: Path) -> bool:
    process = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process.returncode == 0


def _test_static_entrypoint_order() -> None:
    for name, scientific_marker in ENTRYPOINT_MARKERS.items():
        source = (ROOT / name).read_text(encoding="utf-8")
        try:
            environment_guard = source.index(
                '("PYTHONPATH", "PYTHONHOME")'
            )
            boundary_call = source.index("_enforce_import_boundary()")
            scientific_import = source.index(scientific_marker)
        except ValueError:
            ordered = False
        else:
            ordered = environment_guard < boundary_call < scientific_import
        tree = ast.parse(source, filename=name)
        boundary_lines = [
            node.lineno
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_enforce_import_boundary"
        ]
        allowed_before = {
            "__future__",
            "os",
            "sys",
            "campaign_import_guard",
        }
        unsafe_imports: list[str] = []
        if len(boundary_lines) == 1:
            boundary_line = boundary_lines[0]
            for node in tree.body:
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                if node.lineno >= boundary_line:
                    continue
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                unsafe_imports.extend(
                    imported
                    for imported in names
                    if imported.split(".", 1)[0] not in allowed_before
                )
        else:
            unsafe_imports.append("<missing/duplicate-boundary-call>")
        check(
            f"{name} has no substantive pre-boundary import",
            ordered and not unsafe_imports,
        )


def _test_live_guard_origin() -> None:
    spec = importlib.machinery.PathFinder.find_spec(
        "campaign_import_guard", sys.path
    )
    locations = tuple(
        Path(path).resolve()
        for path in (spec.submodule_search_locations or ())
    ) if spec is not None else ()
    check(
        "live guard resolves from the exact reviewed regular package",
        Path(guard_module.__file__).resolve() == GUARD_SOURCE.resolve()
        and spec is not None
        and spec.origin is not None
        and Path(spec.origin).resolve() == GUARD_SOURCE.resolve()
        and locations == (GUARD_SOURCE.parent.resolve(),),
    )


def _test_positive_and_source_shadows() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        entry = _make_fixture(Path(temporary))
        process = _run_entry(entry)
        check(
            "clean reviewed fixture reaches its entry body",
            process.returncode == 0
            and _combined(process).count(ENTRY_BODY_MARKER) == 1,
        )

    _exercise_source_candidate(
        "mixed-case stdlib file shadow is rejected",
        Path("JsOn.PY"),
        diagnostic="import shadow for 'json'",
    )
    _exercise_source_candidate(
        "source-root sitecustomize startup hook is rejected",
        Path("sitecustomize.py"),
        diagnostic="import shadow for 'sitecustomize'",
    )
    _exercise_source_candidate(
        "source-root usercustomize startup hook is rejected",
        Path("usercustomize.py"),
        diagnostic="import shadow for 'usercustomize'",
    )
    compiled_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    _exercise_source_candidate(
        "ABI-tagged installed-extension shadow is rejected",
        Path(f"numpy{compiled_suffix}"),
        diagnostic="import shadow for 'numpy'",
    )
    _exercise_source_candidate(
        "compiled guard initializer alternative is rejected",
        Path("campaign_import_guard", f"__init__{compiled_suffix}"),
        diagnostic="reviewed campaign import guard package is absent or ambiguous",
    )
    _exercise_source_candidate(
        "core.py alternative to the reviewed package is rejected",
        Path("core.py"),
        diagnostic="competing import candidate",
    )
    _exercise_source_candidate(
        "same-stem guard module cannot replace the reviewed package",
        Path("campaign_import_guard.py"),
        diagnostic="competing import candidate",
    )
    _exercise_source_candidate(
        "package alternative to a reviewed root module is rejected",
        Path("dispatch_authorization"),
        directory=True,
        paired_module=True,
        diagnostic="competing import candidate",
    )
    _exercise_source_candidate(
        "package alternative to a reviewed core module is rejected",
        Path("core", "environment"),
        directory=True,
        paired_module=True,
        diagnostic="competing module candidate",
    )
    _exercise_source_candidate(
        "unmanifested regular-package module is rejected",
        Path("TTBI_2D", "legacy_escape.py"),
        diagnostic="outside the reviewed source manifest",
    )


def _test_resolution_non_vacuity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "dispatch_authorization.py").write_text(
            "REVIEWED = True\n", encoding="utf-8"
        )
        package = root / "dispatch_authorization"
        package.mkdir()
        (package / "__init__.py").write_text(
            "SHADOW = True\n", encoding="utf-8"
        )
        spec = importlib.machinery.PathFinder.find_spec(
            "dispatch_authorization", [str(root)]
        )
        check(
            "package-alternative probe really wins normal module resolution",
            spec is not None
            and spec.origin is not None
            and Path(spec.origin).resolve() == (package / "__init__.py").resolve(),
        )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "core").mkdir()
        (root / "core.py").write_text(
            "LEGACY_SHADOW = True\n", encoding="utf-8"
        )
        spec = importlib.machinery.PathFinder.find_spec("core", [str(root)])
        check(
            "old namespace-core probe really resolved core.py",
            spec is not None
            and spec.origin is not None
            and Path(spec.origin).resolve() == (root / "core.py").resolve(),
        )

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        source = base / "source"
        site = base / "site"
        (source / "core").mkdir(parents=True)
        (site / "core").mkdir(parents=True)
        (site / "core" / "__init__.py").write_text(
            "SITE_SHADOW = True\n", encoding="utf-8"
        )
        spec = importlib.machinery.PathFinder.find_spec(
            "core", [str(source), str(site)]
        )
        check(
            "regular site core really defeats the old namespace package",
            spec is not None
            and spec.origin is not None
            and Path(spec.origin).resolve()
            == (site / "core" / "__init__.py").resolve(),
        )


def _test_external_path_rejection() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        source = base / "source"
        attacker = base / "attacker"
        source.mkdir()
        attacker.mkdir()
        entry = _make_fixture(source)
        (attacker / "numpy.py").write_text(
            "SHADOW = True\n", encoding="utf-8"
        )
        fake_guard = attacker / "campaign_import_guard"
        fake_guard.mkdir()
        (fake_guard / "__init__.py").write_text(
            f'print("{FAKE_GUARD_MARKER}")\n'
            "def enforce_import_boundary(): return {}\n",
            encoding="utf-8",
        )
        environment = _clean_environment()
        environment["PYTHONPATH"] = str(attacker)
        process = _run_entry(entry, environment=environment)
        spec = importlib.machinery.PathFinder.find_spec(
            "numpy",
            [
                str(attacker),
                str(Path(sys.prefix, "Lib", "site-packages")),
            ],
        )
        check(
            "external numpy probe really wins when its path is admitted",
            spec is not None
            and spec.origin is not None
            and Path(spec.origin).resolve() == (attacker / "numpy.py").resolve(),
        )
        check(
            "PYTHONPATH is rejected before an external guard can execute",
            _rejected_before_body(process, "PYTHONPATH must be absent")
            and FAKE_GUARD_MARKER not in _combined(process),
        )

        (attacker / "sitecustomize.py").write_text(
            "import os, sys\n"
            f'print("{STARTUP_MARKER}")\n'
            'os.environ.pop("PYTHONPATH", None)\n'
            f"sys.path.insert(0, {str(attacker)!r})\n",
            encoding="utf-8",
        )
        process = _run_entry(entry, environment=environment)
        output = _combined(process)
        check(
            "startup-path injection is non-vacuous",
            STARTUP_MARKER in output,
        )
        check(
            "external first path is rejected before the fake guard import",
            _rejected_before_body(
                process, "import-path order differs"
            )
            and FAKE_GUARD_MARKER not in output,
        )


def _test_preloaded_guard_rejection() -> None:
    attack = (
        '_fake_guard = type(_bootstrap_sys)("campaign_import_guard")\n'
        "_fake_guard.enforce_import_boundary = "
        f'lambda: print("{FAKE_GUARD_MARKER}")\n'
        '_bootstrap_sys.modules["campaign_import_guard"] = _fake_guard'
    )
    preloaded_check = '''_bootstrap_loaded_guard = _bootstrap_sys.modules.get("campaign_import_guard")
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
'''
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary)
        entry = _make_fixture(source, pre_guard=attack)
        guarded = entry.read_text(encoding="utf-8")
        unguarded = guarded.replace(preloaded_check, "", 1)
        entry.write_text(unguarded, encoding="utf-8")
        process = _run_entry(entry)
        output = _combined(process)
        check(
            "preloaded fake guard attack is non-vacuous",
            unguarded != guarded
            and process.returncode == 0
            and FAKE_GUARD_MARKER in output
            and ENTRY_BODY_MARKER in output,
        )

        entry.write_text(guarded, encoding="utf-8")
        process = _run_entry(entry)
        output = _combined(process)
        check(
            "preloaded fake guard is rejected before it can execute",
            _rejected_before_body(
                process,
                "preloaded campaign import guard is not the reviewed "
                "enforced module",
            )
            and FAKE_GUARD_MARKER not in output,
        )


def _test_preloaded_scientific_module_rejection() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        source = base / "source"
        external = base / "external"
        source.mkdir()
        external.mkdir()
        external_numpy = external / "numpy.py"
        external_numpy.write_text("EXTERNAL = True\n", encoding="utf-8")
        attack = (
            '_fake_numpy = type(_bootstrap_sys)("numpy")\n'
            f"_fake_numpy.__file__ = {str(external_numpy)!r}\n"
            "_fake_numpy.ATTACK_MARKER = "
            f'"{PRELOADED_SCIENTIFIC_MARKER}"\n'
            '_bootstrap_sys.modules["numpy"] = _fake_numpy'
        )
        entry = _make_fixture(source, pre_guard=attack)
        entry_source = entry.read_text(encoding="utf-8").replace(
            f'print("{ENTRY_BODY_MARKER}")',
            "import numpy\n"
            "print(numpy.ATTACK_MARKER)\n"
            f'print("{ENTRY_BODY_MARKER}")',
        )
        entry.write_text(entry_source, encoding="utf-8")

        fixture_guard = source / "campaign_import_guard" / "__init__.py"
        guarded_source = fixture_guard.read_text(encoding="utf-8")
        validation_call = (
            "    _validate_loaded_module_origins(source_root, site_packages)\n"
        )
        unguarded_source = guarded_source.replace(validation_call, "", 1)
        fixture_guard.write_text(unguarded_source, encoding="utf-8")
        process = _run_entry(entry)
        output = _combined(process)
        check(
            "preloaded external scientific module attack is non-vacuous",
            unguarded_source != guarded_source
            and process.returncode == 0
            and PRELOADED_SCIENTIFIC_MARKER in output
            and ENTRY_BODY_MARKER in output,
        )

        fixture_guard.write_text(guarded_source, encoding="utf-8")
        process = _run_entry(entry)
        output = _combined(process)
        check(
            "preloaded external scientific module is rejected",
            _rejected_before_body(
                process, "outside the authorized campaign roots"
            )
            and PRELOADED_SCIENTIFIC_MARKER not in output,
        )


def _test_exact_search_path_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        source = base / "source"
        external = base / "external-after-site"
        source.mkdir()
        external.mkdir()
        entry = _make_fixture(
            source,
            pre_guard=(
                f"_bootstrap_sys.path.append({str(external)!r})"
            ),
        )
        process = _run_entry(entry)
        check(
            "external path after site-packages is rejected",
            _rejected_before_body(process, "import-path order differs"),
        )

    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary)
        duplicate_code = """for _bootstrap_index, _bootstrap_entry in enumerate(
        tuple(_bootstrap_sys.path)
):
    if _bootstrap_os.path.normcase(_bootstrap_entry).endswith(
        _bootstrap_os.path.normcase(_bootstrap_os.path.join("", "Lib"))
    ):
        _bootstrap_sys.path.insert(
            _bootstrap_index + 1, _bootstrap_entry
        )
        break"""
        entry = _make_fixture(source, pre_guard=duplicate_code)
        process = _run_entry(entry)
        check(
            "duplicate CPython Lib path is rejected",
            _rejected_before_body(process, "import-path order differs"),
        )


def _test_site_package_and_reparse_boundaries() -> None:
    with mock.patch.object(
        guard_module.os.path, "islink", return_value=True
    ):
        check(
            "symbolic-link branch is fail-closed",
            guard_module._has_reparse_point(str(ROOT)),
        )

    with tempfile.TemporaryDirectory() as temporary:
        site_root = Path(temporary)
        (site_root / "core").mkdir()
        try:
            validate_source_tree(str(ROOT), str(site_root))
        except ImportBoundaryError as exc:
            rejected = "competing project package" in str(exc)
        else:
            rejected = False
        check("installed regular core competitor is rejected", rejected)

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        target = base / "site-target"
        alias = base / "site-junction"
        target.mkdir()
        created = _make_junction(alias, target)
        try:
            junction_shape = (
                created
                and not os.path.islink(alias)
                and bool(getattr(os.path, "isjunction", lambda _p: False)(alias))
            )
            try:
                validate_source_tree(str(ROOT), str(alias))
            except ImportBoundaryError as exc:
                rejected = "non-reparse directory" in str(exc)
            else:
                rejected = False
            check("site-packages junction probe is a real junction", junction_shape)
            check("site-packages junction is rejected", created and rejected)
        finally:
            if alias.exists():
                os.rmdir(alias)

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        target = base / "source-target"
        alias = base / "source-junction"
        target.mkdir()
        entry = _make_fixture(target)
        created = _make_junction(alias, target)
        try:
            process = _run_entry(alias / entry.name) if created else None
            check(
                "source-root junction is rejected before guard import",
                process is not None
                and _rejected_before_body(
                    process, "canonical first import path"
                ),
            )
        finally:
            if alias.exists():
                os.rmdir(alias)


def main() -> int:
    print("Import boundary behavioral audit")
    _test_static_entrypoint_order()
    _test_live_guard_origin()
    _test_positive_and_source_shadows()
    _test_resolution_non_vacuity()
    _test_external_path_rejection()
    _test_preloaded_guard_rejection()
    _test_preloaded_scientific_module_rejection()
    _test_exact_search_path_contract()
    _test_site_package_and_reparse_boundaries()
    print(f"\nIMPORT BOUNDARY: {FAILURES} FAILURE(S)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())

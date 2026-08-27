"""Behavioral checks for the portable project-import boundary.

The boundary protects the campaign from accidentally importing another copy
of ``core``, ``training`` or ``TTBI_2D``. It deliberately does not prescribe
one virtual environment, one absolute checkout path or an exact ``sys.path``.
"""

from __future__ import annotations

import importlib.machinery
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from campaign_import_guard import (
    ImportBoundaryError,
    enforce_import_boundary,
)


ROOT = Path(__file__).resolve().parent
PROJECT_PACKAGES = ("core", "training", "TTBI_2D")
FAILURES = 0


def check(label: str, condition: bool) -> None:
    global FAILURES
    passed = bool(condition)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    FAILURES += int(not passed)


def within(path: str | Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            (os.path.realpath(path), os.path.realpath(root))
        ) == os.path.realpath(root)
    except (OSError, ValueError):
        return False


def package_origins(search_path: list[str]) -> dict[str, str]:
    origins: dict[str, str] = {}
    for package in PROJECT_PACKAGES:
        spec = importlib.machinery.PathFinder.find_spec(package, search_path)
        origins[package] = "" if spec is None else str(spec.origin or "")
    return origins


def make_relocated_fixture(root: Path) -> Path:
    fixture = root / "relocated campaign workspace"
    fixture.mkdir()
    shutil.copytree(ROOT / "campaign_import_guard", fixture / "campaign_import_guard")
    for package in PROJECT_PACKAGES:
        package_dir = fixture / package
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text(
            f'"""Relocated {package} fixture."""\n',
            encoding="utf-8",
        )
    return fixture


def run_probe(
    fixture: Path,
    source: str,
    *,
    pythonpath: list[Path] = (),
) -> subprocess.CompletedProcess[str]:
    probe = fixture / "portable_import_probe.py"
    probe.write_text(source, encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    if pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in pythonpath)
    else:
        environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(probe)],
        cwd=fixture,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def accepted_probe(extra_path: Path, alternate_venv: Path) -> str:
    return f"""
import importlib.machinery
import json
import os
from pathlib import Path
import sys

workspace = Path(__file__).resolve().parent
sys.path.append({str(extra_path)!r})
sys.prefix = {str(alternate_venv)!r}
sys.base_prefix = {str(alternate_venv.parent / 'alternate-base')!r}
os.environ['VIRTUAL_ENV'] = {str(alternate_venv)!r}

from campaign_import_guard import enforce_import_boundary
result = enforce_import_boundary()
origins = {{}}
for name in ('core', 'training', 'TTBI_2D'):
    spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
    origins[name] = spec.origin
print('ACCEPTED=' + json.dumps({{
    'workspace': str(workspace),
    'origins': origins,
    'result': result,
    'prefix': sys.prefix,
    'pythonpath': os.environ.get('PYTHONPATH'),
    'extra_path_present': {str(extra_path)!r} in sys.path,
}}, sort_keys=True))
"""


def rejected_probe(external_root: Path, package: str) -> str:
    return f"""
import sys
from campaign_import_guard import ImportBoundaryError, enforce_import_boundary
sys.path.insert(0, {str(external_root)!r})
try:
    enforce_import_boundary()
except ImportBoundaryError as exc:
    print('REJECTED={package}:' + str(exc))
else:
    print('PROBE_BODY_RAN_UNEXPECTEDLY')
    raise SystemExit(7)
"""


def main() -> int:
    print("PORTABLE IMPORT BOUNDARY")

    result = enforce_import_boundary()
    origins = package_origins(sys.path)
    check(
        "live guard resolves core/training/TTBI_2D from this workspace",
        result["project_packages"] == 3
        and all(origin and within(origin, ROOT) for origin in origins.values()),
    )

    original_path = list(sys.path)
    original_pythonpath = os.environ.get("PYTHONPATH")
    try:
        with tempfile.TemporaryDirectory(prefix="ttbi-import-extra-") as temp_name:
            extra = Path(temp_name).resolve()
            os.environ["PYTHONPATH"] = str(extra)
            sys.path.extend([str(extra), str(extra / "another-search-root")])
            portable_result = enforce_import_boundary()
        check(
            "live guard accepts PYTHONPATH and additional search paths",
            portable_result["project_packages"] == 3
            and portable_result["pythonpath_present"] == 1,
        )
    finally:
        sys.path[:] = original_path
        if original_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_pythonpath

    with tempfile.TemporaryDirectory(prefix="ttbi-relocation-") as temp_name:
        temp_root = Path(temp_name).resolve()
        fixture = make_relocated_fixture(temp_root)
        pythonpath_extra = temp_root / "pythonpath extra"
        appended_extra = temp_root / "appended extra"
        alternate_venv = temp_root / "a different venv"
        pythonpath_extra.mkdir()
        appended_extra.mkdir()
        alternate_venv.mkdir()
        process = run_probe(
            fixture,
            accepted_probe(appended_extra, alternate_venv),
            pythonpath=(pythonpath_extra,),
        )
        payload = None
        for line in process.stdout.splitlines():
            if line.startswith("ACCEPTED="):
                payload = json.loads(line.removeprefix("ACCEPTED="))
        relocated_ok = (
            process.returncode == 0
            and payload is not None
            and Path(payload["workspace"]).resolve() == fixture.resolve()
            and payload["prefix"] == str(alternate_venv)
            and payload["pythonpath"] == str(pythonpath_extra)
            and payload["extra_path_present"] is True
            and all(
                origin and within(origin, fixture)
                for origin in payload["origins"].values()
            )
        )
        check(
            "relocated workspace accepts another venv, PYTHONPATH and extras",
            relocated_ok,
        )
        if not relocated_ok:
            print(process.stdout)
            print(process.stderr)

        for package in PROJECT_PACKAGES:
            external = temp_root / f"external-{package}"
            external_package = external / package
            external_package.mkdir(parents=True)
            (external_package / "__init__.py").write_text(
                '"""Unauthorized competing project copy."""\n',
                encoding="utf-8",
            )
            process = run_probe(
                fixture,
                rejected_probe(external, package),
            )
            check(
                f"{package} resolving outside the workspace is rejected",
                process.returncode == 0
                and f"REJECTED={package}:" in process.stdout
                and "PROBE_BODY_RAN_UNEXPECTEDLY" not in process.stdout,
            )

    # The exception type remains part of the public guard contract.
    check(
        "guard exposes a dedicated import-boundary failure type",
        issubclass(ImportBoundaryError, RuntimeError),
    )

    if FAILURES:
        print(f"\nIMPORT BOUNDARY: {FAILURES} FAILURE(S)")
        return 1
    print("\nIMPORT BOUNDARY: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

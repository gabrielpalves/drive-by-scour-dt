"""Adversarial checks for content-addressed generator/runtime source roots."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from core.source_provenance import (
    DRIVER,
    REPO,
    SOURCE_MANIFEST,
    SourceProvenanceError,
    generator_source_root,
    python_runtime_source_root,
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
    preset_only = original_driver.replace(
        b'STAGE = "s0_scour"',
        b'STAGE = "s23_all4"',
        1,
    ).replace(
        b"SENSOR_NOISE = None",
        b'SENSOR_NOISE = {"mode": "all_mult", "desvio": 0.05}',
        1,
    )
    driver.write_bytes(preset_only)
    check(
        "legitimate bundle STAGE/noise preset rewrites are canonicalised",
        python_runtime_source_root(fixture).sha256 == py0.sha256,
    )

    driver.write_bytes(
        preset_only.replace(
            b'STAGE = "s23_all4"',
            b'STAGE = "s23_all4"; hidden_effect = True',
            1,
        )
    )
    rejects(
        "executable suffix on STAGE preset cannot disappear from the root",
        lambda: python_runtime_source_root(fixture),
    )
    driver.write_bytes(
        preset_only.replace(
            b'SENSOR_NOISE = {"mode": "all_mult", "desvio": 0.05}',
            b'SENSOR_NOISE = {"mode": "all_mult", "desvio": 0.05}; '
            b'hidden_effect = True',
            1,
        )
    )
    rejects(
        "executable suffix on SENSOR_NOISE cannot disappear from the root",
        lambda: python_runtime_source_root(fixture),
    )
    driver.write_bytes(
        preset_only.replace(
            b'SENSOR_NOISE = {"mode": "all_mult", "desvio": 0.05}',
            b'SENSOR_NOISE = {"mode": side_effect(), "desvio": 0.05}',
            1,
        )
    )
    rejects(
        "non-literal SENSOR_NOISE preset fails closed",
        lambda: python_runtime_source_root(fixture),
    )

    driver.write_bytes(
        preset_only.replace(
            b"EPOCHS         = 50",
            b"EPOCHS         = 49",
            1,
        )
    )
    check(
        "one non-preset driver byte moves the Python runtime root",
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

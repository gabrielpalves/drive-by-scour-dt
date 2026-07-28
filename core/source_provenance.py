"""Content-addressed provenance for reviewed generator and Python sources.

The tracked ``bundle_source_files.txt`` file is the explicit boundary of the
campaign package.  These helpers turn the relevant regular-file bytes into
canonical SHA-256 roots:

* ``generator_source_root`` covers every listed ``scour_MATLAB/`` file,
  including MATLAB code and numerical assets;
* ``python_runtime_source_root`` covers every listed Python file plus the
  Python/MATLAB environment lock and pinned requirements.

The bundle builder legitimately rewrites only the driver's ``STAGE`` and
``SENSOR_NOISE`` preset lines.  Those two values already enter the protocol
descriptor as executable inputs, so the runtime-source hash canonicalises only
those two lines.  Every other byte remains identity-bearing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import unicodedata


REPO = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = "bundle_source_files.txt"
ENVIRONMENT_LOCK = "environment/campaign-py313-cu128.json"
REQUIREMENTS_LOCK = "requirements-campaign-py313-cu128.txt"
DRIVER = "comprehensive_ablation_multidamage.py"


class SourceProvenanceError(RuntimeError):
    """The reviewed source boundary is absent, unsafe, or incomplete."""


@dataclass(frozen=True)
class SourceRoot:
    sha256: str
    file_count: int
    files: tuple[str, ...]
    digest_lines: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_names(root: Path) -> tuple[str, ...]:
    manifest = root / SOURCE_MANIFEST
    if (
        not manifest.is_file()
        or manifest.is_symlink()
        or not stat.S_ISREG(manifest.stat().st_mode)
    ):
        raise SourceProvenanceError(
            f"missing regular reviewed source manifest: {manifest}"
        )
    try:
        text = manifest.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceProvenanceError(
            f"{manifest} is not canonical UTF-8"
        ) from exc

    windows_forbidden = set('<>:"|?*')
    windows_reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    names: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        # Canonical grammar shared with the bundle builder and MATLAB helper:
        # only a truly empty line or a column-zero comment is ignorable.
        # Whitespace-only and indented-comment lines are rejected below.
        if line == "" or line.startswith("#"):
            continue
        if line != line.strip():
            raise SourceProvenanceError(
                f"non-canonical whitespace in {manifest}:{line_number}"
            )
        name = line
        parts = name.split("/")
        bad_component = any(
            not part
            or part in {".", ".."}
            or part != part.strip()
            or part.endswith(".")
            or any(
                char in windows_forbidden
                or ord(char) < 32
                or ord(char) == 127
                for char in part
            )
            or part.split(".", 1)[0].upper() in windows_reserved
            for part in parts
        )
        if (
            "\\" in name
            or name.startswith("/")
            or PureWindowsPath(name).is_absolute()
            or PurePosixPath(name).as_posix() != name
            or unicodedata.normalize("NFC", name) != name
            or bad_component
        ):
            raise SourceProvenanceError(
                f"unsafe/non-canonical source entry "
                f"{manifest}:{line_number}: {name!r}"
            )
        names.append(name)

    if names != sorted(names):
        raise SourceProvenanceError(
            f"{manifest} entries must be sorted"
        )
    if (
        not names
        or len(names) != len(set(names))
        or len(names) != len({name.casefold() for name in names})
    ):
        raise SourceProvenanceError(
            f"{manifest} is empty or contains duplicate/case-colliding entries"
        )
    return tuple(names)


def _regular_bytes(root: Path, name: str) -> bytes:
    path = root.joinpath(*name.split("/"))
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise SourceProvenanceError(
            f"reviewed source entry is missing: {name}"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise SourceProvenanceError(
            f"reviewed source entry is not a regular file: {name}"
        )
    return path.read_bytes()


def _normalise_driver_presets(payload: bytes) -> bytes:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceProvenanceError(
            f"{DRIVER} is not UTF-8"
        ) from exc

    lines = text.splitlines(keepends=True)
    stage_count = 0
    noise_count = 0
    normalised: list[str] = []
    for line in lines:
        ending = ""
        body = line
        if body.endswith("\r\n"):
            body, ending = body[:-2], "\r\n"
        elif body.endswith("\n") or body.endswith("\r"):
            body, ending = body[:-1], body[-1:]
        stage_match = re.fullmatch(
            r'(STAGE = )"([^"\r\n]+)"([ \t]*(?:#[^\r\n]*)?)',
            body,
        )
        if stage_match:
            stage_count += 1
            # Preserve all non-value bytes (including the reviewed comment);
            # only the builder-controlled literal is canonicalised.
            body = (
                f'{stage_match.group(1)}"<BUNDLE_STAGE_PRESET>"'
                f"{stage_match.group(3)}"
            )
        elif body.startswith("STAGE = "):
            raise SourceProvenanceError(
                f"{DRIVER} has a non-canonical STAGE preset line"
            )
        elif body.startswith("SENSOR_NOISE = "):
            noise_match = re.fullmatch(
                r"SENSOR_NOISE = (None|\{.*\})",
                body,
            )
            if not noise_match:
                raise SourceProvenanceError(
                    f"{DRIVER} has a non-canonical SENSOR_NOISE preset line"
                )
            expression = noise_match.group(1)
            try:
                value = ast.literal_eval(expression)
            except (SyntaxError, ValueError) as exc:
                raise SourceProvenanceError(
                    f"{DRIVER} has a non-literal SENSOR_NOISE preset"
                ) from exc
            if value is not None and not isinstance(value, dict):
                raise SourceProvenanceError(
                    f"{DRIVER} SENSOR_NOISE preset must be None or a literal dict"
                )
            noise_count += 1
            body = "SENSOR_NOISE = <BUNDLE_SENSOR_NOISE_PRESET>"
        normalised.append(body + ending)
    if stage_count != 1 or noise_count != 1:
        raise SourceProvenanceError(
            f"{DRIVER} must contain exactly one STAGE and one SENSOR_NOISE "
            f"preset (found {stage_count}/{noise_count})"
        )
    return "".join(normalised).encode("utf-8")


def _root_for(
    root: Path,
    names: tuple[str, ...],
    *,
    normalise_driver: bool,
) -> SourceRoot:
    lines: list[str] = []
    for name in names:
        payload = _regular_bytes(root, name)
        if normalise_driver and name == DRIVER:
            payload = _normalise_driver_presets(payload)
        lines.append(f"{name}:{_sha256_bytes(payload)}")
    canonical = "\n".join(lines).encode("utf-8")
    return SourceRoot(
        sha256=_sha256_bytes(canonical),
        file_count=len(names),
        files=names,
        digest_lines="\n".join(lines),
    )


def generator_source_root(
    repo: str | Path = REPO,
) -> SourceRoot:
    """Hash every reviewed MATLAB generator/runtime file and numerical asset."""
    root = Path(repo).resolve()
    names = tuple(
        name for name in _manifest_names(root)
        if name.startswith("scour_MATLAB/")
    )
    if not names:
        raise SourceProvenanceError(
            "source manifest contains no scour_MATLAB entries"
        )
    return _root_for(root, names, normalise_driver=False)


def python_runtime_source_root(
    repo: str | Path = REPO,
) -> SourceRoot:
    """Hash executable Python bytes and their pinned environment inputs."""
    root = Path(repo).resolve()
    selected = tuple(
        name for name in _manifest_names(root)
        if name.endswith(".py")
        or name in {ENVIRONMENT_LOCK, REQUIREMENTS_LOCK}
    )
    required = {DRIVER, ENVIRONMENT_LOCK, REQUIREMENTS_LOCK}
    missing = sorted(required - set(selected))
    if missing:
        raise SourceProvenanceError(
            f"runtime source manifest lacks required entries: {missing}"
        )
    return _root_for(root, selected, normalise_driver=True)

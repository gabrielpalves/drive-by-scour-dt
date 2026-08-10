"""Canonical path and TOCTOU guards for qualification receipt files.

This module is intentionally evidence-agnostic. It does not parse receipt
JSON or inspect MATLAB datasets; it only establishes that each explicitly
selected receipt is one canonical, regular, single-link file reached without
symlinks or junctions. It then provides an exact byte-and-identity snapshot
that the inventory checker reasserts after all numerical work finishes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Iterable


MAX_RECEIPT_BYTES = 16 * 1024 * 1024


class QualificationPathError(RuntimeError):
    """A receipt path is ambiguous, aliased, nonregular, or unstable."""


@dataclass(frozen=True)
class ReceiptSnapshot:
    """Exact file state captured around one descriptor-based read."""

    path: Path
    raw: bytes
    sha256: str
    size: int
    mtime_ns: int
    file_id: tuple[int, int]


def _is_junction(path: Path) -> bool:
    predicate = getattr(os.path, "isjunction", None)
    return bool(predicate is not None and predicate(path))


def _inspect_path_components(path: Path, owner: str) -> None:
    current = path
    while True:
        try:
            if current.is_symlink() or _is_junction(current):
                raise QualificationPathError(
                    f"{owner} traverses a symlink/junction: {current}")
        except OSError as exc:
            raise QualificationPathError(
                f"{owner} path component cannot be inspected: {current}"
            ) from exc
        if current.parent == current:
            return
        current = current.parent


def canonical_existing_path(
    raw: Path | str,
    owner: str,
    *,
    kind: str,
) -> Path:
    """Return an exact canonical file/directory path or fail closed."""

    if kind not in {"file", "directory"}:
        raise ValueError(f"unsupported canonical path kind: {kind}")
    supplied = Path(raw)
    supplied_text = str(supplied)
    if not supplied.is_absolute() or not supplied_text or "\0" in supplied_text:
        raise QualificationPathError(f"{owner} must be one absolute path")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise QualificationPathError(
            f"{owner} cannot be resolved: {supplied}") from exc
    if supplied_text != str(resolved):
        raise QualificationPathError(
            f"{owner} is not its canonical resolved path: {supplied}")
    _inspect_path_components(supplied, owner)
    try:
        info = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise QualificationPathError(
            f"{owner} cannot be inspected: {resolved}") from exc
    expected_kind = (
        stat.S_ISREG(info.st_mode)
        if kind == "file"
        else stat.S_ISDIR(info.st_mode)
    )
    if not expected_kind:
        raise QualificationPathError(
            f"{owner} must be one real {kind}: {resolved}")
    if kind == "file" and int(getattr(info, "st_nlink", 1)) != 1:
        raise QualificationPathError(
            f"{owner} must not have a hard-link alias: {resolved}")
    return resolved


def canonical_new_file_path(raw: Path | str, owner: str) -> Path:
    """Return a canonical not-yet-existing file path in a real directory."""

    supplied = Path(raw)
    supplied_text = str(supplied)
    if not supplied.is_absolute() or not supplied_text or "\0" in supplied_text:
        raise QualificationPathError(f"{owner} must be one absolute path")
    parent = canonical_existing_path(
        supplied.parent, f"{owner} parent", kind="directory")
    canonical = parent / supplied.name
    if supplied_text != str(canonical):
        raise QualificationPathError(
            f"{owner} is not its canonical output path: {supplied}")
    if os.path.lexists(canonical):
        raise QualificationPathError(
            f"{owner} is create-once and already exists: {canonical}")
    return canonical


def collect_receipt_paths(
    *,
    receipt_files: Iterable[Path | str],
    receipt_dirs: Iterable[Path | str],
) -> tuple[Path, ...]:
    """Collect only canonical immediate ``*.json`` receipt files."""

    candidates = list(receipt_files)
    for index, raw_directory in enumerate(receipt_dirs, 1):
        directory = canonical_existing_path(
            raw_directory,
            f"qualification receipt directory {index}",
            kind="directory",
        )
        candidates.extend(sorted(directory.glob("*.json")))
    if not candidates:
        raise QualificationPathError(
            "no v4 receipt files were explicitly supplied or found")

    unique: dict[str, Path] = {}
    for index, raw_path in enumerate(candidates, 1):
        path = canonical_existing_path(
            raw_path,
            f"qualification receipt {index}",
            kind="file",
        )
        key = str(path).casefold()
        if key in unique:
            raise QualificationPathError(
                f"duplicate receipt path supplied: {path} and {unique[key]}")
        unique[key] = path
    return tuple(unique[key] for key in sorted(unique))


def snapshot_receipt(path: Path, owner: str) -> ReceiptSnapshot:
    """Read one canonical receipt while binding content and file identity."""

    canonical = canonical_existing_path(path, owner, kind="file")
    flags = os.O_RDONLY
    flags |= int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(canonical, flags)
    except OSError as exc:
        raise QualificationPathError(
            f"{owner} cannot be opened securely") from exc
    try:
        before = os.fstat(descriptor)
        if before.st_size < 0 or before.st_size > MAX_RECEIPT_BYTES:
            raise QualificationPathError(
                f"{owner} exceeds the receipt snapshot limit")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise QualificationPathError(
            f"{owner} changed while being read") from exc
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        path_after = os.stat(canonical, follow_symlinks=False)
    except OSError as exc:
        raise QualificationPathError(
            f"{owner} disappeared during its snapshot") from exc

    def identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(getattr(info, "st_dev", 0)),
            int(getattr(info, "st_ino", 0)),
            int(info.st_size),
            int(info.st_mtime_ns),
            int(getattr(info, "st_nlink", 1)),
        )

    if (
        identity(before) != identity(after)
        or identity(after) != identity(path_after)
        or not stat.S_ISREG(before.st_mode)
        or identity(before)[-1] != 1
        or len(raw) != before.st_size
    ):
        raise QualificationPathError(
            f"{owner} changed/replaced while being read")
    return ReceiptSnapshot(
        path=canonical,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        size=int(before.st_size),
        mtime_ns=int(before.st_mtime_ns),
        file_id=(
            int(getattr(before, "st_dev", 0)),
            int(getattr(before, "st_ino", 0)),
        ),
    )


def assert_snapshot_unchanged(
    snapshot: ReceiptSnapshot,
    owner: str,
) -> None:
    """Require exact bytes, metadata, and file identity to remain unchanged."""

    current = snapshot_receipt(snapshot.path, owner)
    if current != snapshot:
        raise QualificationPathError(
            f"{owner} changed/replaced during inventory validation")

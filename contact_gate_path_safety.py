"""Canonical path boundary for retained contact-gate evidence.

The contact verifier accepts three classes of filesystem input: a completed
gate directory, three retained qualification datasets named by that gate, and
an external create-once authorization receipt.  Resolving an input before
validating it erases evidence that the caller supplied ``..``, a symlink, or a
Windows junction.  This module validates the supplied path first and returns
its canonical spelling only after every existing component is proven free of
reparse points.

The helpers are intentionally small and contain no gate semantics.  They are
shared by the verifier and its behavioral path-alias tests.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any


_WINDOWS_REPARSE_POINT = 0x0400


class GateError(RuntimeError):
    """Fail-closed contact-closure evidence error."""


def _is_reparse_point(path: Path) -> bool:
    """Return whether ``path`` itself is a symlink/junction/reparse point."""
    try:
        info = path.lstat()
    except OSError:
        return False
    is_junction = getattr(os.path, "isjunction", None)
    return (
        path.is_symlink()
        or bool(is_junction is not None and is_junction(path))
        or bool(
            int(getattr(info, "st_file_attributes", 0))
            & _WINDOWS_REPARSE_POINT
        )
    )


def _require_absolute_canonical_spelling(
    supplied: Path,
    resolved: Path,
    label: str,
) -> None:
    if not supplied.is_absolute():
        raise GateError(f"{label} path must be absolute: {supplied}")
    if os.path.normcase(str(supplied)) != os.path.normcase(str(resolved)):
        raise GateError(
            f"{label} path must use its canonical spelling without aliases: "
            f"{supplied} != {resolved}"
        )


def _require_no_reparse_ancestors(path: Path, label: str) -> None:
    """Reject a reparse point at ``path`` or any existing ancestor."""
    current = path
    while True:
        if _is_reparse_point(current):
            raise GateError(
                f"{label} path traverses a symlink/junction/reparse point: "
                f"{current}"
            )
        if current.parent == current:
            return
        current = current.parent


def canonical_existing_directory(raw: Any, label: str) -> Path:
    """Return one existing canonical, non-reparse directory."""
    supplied = Path(raw)
    if not supplied.is_absolute():
        raise GateError(f"{label} path must be absolute: {supplied}")
    try:
        resolved = supplied.resolve(strict=True)
        info = supplied.lstat()
    except OSError as exc:
        raise GateError(
            f"{label} directory is unavailable: {supplied}"
        ) from exc
    _require_absolute_canonical_spelling(supplied, resolved, label)
    _require_no_reparse_ancestors(supplied, label)
    if not stat.S_ISDIR(info.st_mode):
        raise GateError(f"{label} must be one real directory: {supplied}")
    return resolved


def canonical_existing_file(raw: Any, label: str) -> Path:
    """Return one existing canonical, unaliased regular file."""
    supplied = Path(raw)
    if not supplied.is_absolute():
        raise GateError(f"{label} path must be absolute: {supplied}")
    try:
        resolved = supplied.resolve(strict=True)
        info = supplied.lstat()
    except OSError as exc:
        raise GateError(f"{label} file is unavailable: {supplied}") from exc
    _require_absolute_canonical_spelling(supplied, resolved, label)
    _require_no_reparse_ancestors(supplied, label)
    if not stat.S_ISREG(info.st_mode):
        raise GateError(f"{label} must be one regular file: {supplied}")
    if int(getattr(info, "st_nlink", 1)) != 1:
        raise GateError(f"{label} must not be a hard-linked alias: {supplied}")
    return resolved


def canonical_receipt_path(raw: Any, label: str) -> Path:
    """Validate an existing or not-yet-created receipt pathname.

    The parent must already exist canonically.  If the final path exists it
    must also be an unaliased regular file; otherwise the canonical result is
    formed from the authenticated parent and the single supplied filename.
    """
    supplied = Path(raw)
    if not supplied.is_absolute():
        raise GateError(f"{label} path must be absolute: {supplied}")
    parent = canonical_existing_directory(supplied.parent, f"{label} parent")
    candidate = parent / supplied.name
    _require_absolute_canonical_spelling(supplied, candidate, label)
    if supplied.exists() or supplied.is_symlink():
        return canonical_existing_file(supplied, label)
    _require_no_reparse_ancestors(parent, label)
    return candidate

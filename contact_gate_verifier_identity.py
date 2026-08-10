"""Transitive source identity for the independent contact-gate verifier."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat

from contact_gate_path_safety import GateError, canonical_existing_file
from core.source_provenance import (
    SourceProvenanceError,
    _reject_unsafe_loader_indirection,
)


ROOT = Path(__file__).resolve().parent

# Every repository-owned Python module reachable from the verifier or its
# argument-free behavioral suite.  Generic standard-library/SciPy modules are
# authenticated by the environment lock instead.  The contact_gate_*.py
# inventory is also checked for exactness below, so a newly split verifier
# module cannot silently fall outside this root.
VERIFIER_SOURCE_FILES = (
    "campaign_import_guard/__init__.py",
    "check_contact_closure_gate.py",
    "compare_generation_releases.py",
    "contact_gate_artifacts.py",
    "contact_gate_authorization.py",
    "contact_gate_case.py",
    "contact_gate_core.py",
    "contact_gate_dataset.py",
    "contact_gate_fixtures.py",
    "contact_gate_numerics.py",
    "contact_gate_path_safety.py",
    "contact_gate_policy.py",
    "contact_gate_selftests.py",
    "contact_gate_source_contract.py",
    "contact_gate_verifier_identity.py",
    "core/__init__.py",
    "core/campaign_contract.py",
    "core/environment.py",
    "core/environment_artifacts.py",
    "core/generation_state_contract.py",
    "core/source_provenance.py",
    "make_micro_smoke.py",
)

_EXPECTED_CONTACT_MODULES = frozenset(
    Path(name).name
    for name in VERIFIER_SOURCE_FILES
    if Path(name).name.startswith("contact_gate_")
)
_VERIFIER_ENTRY_FILES = (
    "check_contact_closure_gate.py",
    "contact_gate_fixtures.py",
    "contact_gate_selftests.py",
)
_DYNAMIC_IMPORT_CALLS = frozenset({
    "__import__",
    "compile",
    "eval",
    "exec",
    "ExtensionFileLoader",
    "FileFinder",
    "create_module",
    "exec_module",
    "get_code",
    "get_source",
    "import_module",
    "load_module",
    "module_from_spec",
    "run_module",
    "run_path",
    "SourcelessFileLoader",
    "SourceFileLoader",
    "spec_from_file_location",
})
_DYNAMIC_ATTRIBUTE_LOADERS = _DYNAMIC_IMPORT_CALLS - {
    "compile",
    "eval",
    "exec",
}


@dataclass(frozen=True)
class _SourceSnapshot:
    path: Path
    raw: bytes
    identity: tuple[int, int, int, int, int]


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(getattr(info, "st_dev", 0)),
        int(getattr(info, "st_ino", 0)),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(getattr(info, "st_nlink", 1)),
    )


def _snapshot_source(relative: str) -> _SourceSnapshot:
    """Read one verifier source once, binding bytes to a stable file ID."""
    label = f"contact verifier source {relative}"
    path = canonical_existing_file(ROOT / relative, label)
    before_path = path.stat(follow_symlinks=False)
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GateError(f"cannot securely open {label}") from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise GateError(f"cannot snapshot {label}") from exc
    finally:
        os.close(descriptor)
    path_after = canonical_existing_file(ROOT / relative, label)
    after_path = path_after.stat(follow_symlinks=False)
    raw = b"".join(chunks)
    identity = _identity(before)
    if (
        path_after != path
        or identity != _identity(before_path)
        or identity != _identity(after)
        or identity != _identity(after_path)
        or not stat.S_ISREG(before.st_mode)
        or identity[-1] != 1
        or len(raw) != before.st_size
    ):
        raise GateError(f"{label} changed/relinked while being read")
    return _SourceSnapshot(path=path, raw=raw, identity=identity)


def _assert_snapshot_unchanged(relative: str, snapshot: _SourceSnapshot) -> None:
    current = _snapshot_source(relative)
    if current.identity != snapshot.identity or current.raw != snapshot.raw:
        raise GateError(
            f"contact verifier source {relative} changed during root validation"
        )


def _local_module_files(module: str) -> set[str]:
    """Resolve one absolute import name to repository-owned Python files."""
    if not module:
        return set()
    parts = module.split(".")
    candidates = (
        ROOT.joinpath(*parts).with_suffix(".py"),
        ROOT.joinpath(*parts, "__init__.py"),
    )
    found: set[str] = set()
    for candidate in candidates:
        if candidate.is_file():
            found.add(candidate.relative_to(ROOT).as_posix())
            for depth in range(1, len(parts)):
                package_init = ROOT.joinpath(
                    *parts[:depth], "__init__.py"
                )
                if package_init.is_file():
                    found.add(package_init.relative_to(ROOT).as_posix())
    return found


def _static_local_imports_from_text(relative: str, source: str) -> set[str]:
    """Parse imports and reject source-loading forms that evade the closure."""
    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError as exc:
        raise GateError(
            f"cannot parse contact verifier dependency {relative}: {exc}"
        ) from exc
    try:
        _reject_unsafe_loader_indirection(tree, relative)
    except SourceProvenanceError as exc:
        raise GateError(str(exc)) from exc
    imports: set[str] = set()
    dynamic_aliases = set(_DYNAMIC_IMPORT_CALLS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                if alias.name in _DYNAMIC_IMPORT_CALLS:
                    dynamic_aliases.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _DYNAMIC_ATTRIBUTE_LOADERS
        ):
            raise GateError(
                f"contact verifier dependency {relative} references dynamic "
                f"source loader {node.attr!r}; aliases/factories cannot be "
                "authenticated by the static closure"
            )
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in dynamic_aliases
        ):
            raise GateError(
                f"contact verifier dependency {relative} references dynamic "
                f"source loader alias {node.id!r}"
            )
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in _DYNAMIC_IMPORT_CALLS
        ):
            raise GateError(
                f"contact verifier dependency {relative} indexes dynamic "
                f"source loader {node.slice.value!r}"
            )
        if isinstance(node, ast.Call):
            called = node.func
            call_name = called.id if isinstance(called, ast.Name) else (
                called.attr if isinstance(called, ast.Attribute) else ""
            )
            is_dynamic_call = (
                isinstance(called, ast.Name)
                and call_name in dynamic_aliases
            ) or (
                isinstance(called, ast.Attribute)
                and call_name in _DYNAMIC_ATTRIBUTE_LOADERS
            )
            if is_dynamic_call:
                raise GateError(
                    f"contact verifier dependency {relative} uses dynamic "
                    f"source loading through {call_name}(); its local import "
                    "closure cannot be authenticated statically"
                )
            if (
                call_name == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in _DYNAMIC_IMPORT_CALLS
            ):
                raise GateError(
                    f"contact verifier dependency {relative} obtains "
                    f"dynamic source loader {node.args[1].value!r} through "
                    "getattr()"
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.update(_local_module_files(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = list(Path(relative).with_suffix("").parts[:-1])
                keep = len(package_parts) - (node.level - 1)
                if keep <= 0:
                    raise GateError(
                        f"contact verifier dependency {relative} has an "
                        "invalid relative import"
                    )
                module_parts = [
                    *package_parts[:keep],
                    *((node.module or "").split(".") if node.module else ()),
                ]
                module = ".".join(module_parts)
            else:
                module = node.module or ""
            imports.update(_local_module_files(module))
            for alias in node.names:
                if alias.name != "*":
                    imports.update(
                        _local_module_files(
                            f"{module}.{alias.name}" if module else alias.name
                        )
                    )
    return imports


def _static_local_imports(relative: str) -> set[str]:
    """Return every repository module named by executable import syntax."""
    snapshot = _snapshot_source(relative)
    try:
        source = snapshot.raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise GateError(
            f"cannot read contact verifier dependency {relative}: {exc}"
        ) from exc
    return _static_local_imports_from_text(relative, source)


def _static_local_import_closure(
    snapshots: dict[str, _SourceSnapshot] | None = None,
) -> frozenset[str]:
    """Discover the exact local closure, including imports inside functions."""
    observed = set(_VERIFIER_ENTRY_FILES)
    pending = list(_VERIFIER_ENTRY_FILES)
    while pending:
        relative = pending.pop()
        if snapshots is None:
            dependencies = _static_local_imports(relative)
        elif relative in snapshots:
            try:
                source = snapshots[relative].raw.decode(
                    "utf-8", errors="strict"
                )
            except UnicodeError as exc:
                raise GateError(
                    f"cannot decode contact verifier dependency {relative}"
                ) from exc
            dependencies = _static_local_imports_from_text(relative, source)
        else:
            # The undeclared dependency itself is already sufficient to make
            # the exact closure fail; never consume unauthenticated bytes from
            # it while constructing the verifier identity.
            dependencies = set()
        for dependency in dependencies:
            if dependency not in observed:
                observed.add(dependency)
                pending.append(dependency)
    return frozenset(observed)


def verifier_source_root() -> str:
    """Return the SHA-256 root over the complete reviewed verifier closure."""
    if (
        tuple(VERIFIER_SOURCE_FILES) != tuple(sorted(VERIFIER_SOURCE_FILES))
        or len(VERIFIER_SOURCE_FILES) != len(set(VERIFIER_SOURCE_FILES))
        or len(VERIFIER_SOURCE_FILES)
        != len({name.casefold() for name in VERIFIER_SOURCE_FILES})
    ):
        raise GateError(
            "contact verifier source inventory must be sorted, unique and "
            "case-collision-free"
        )
    observed_contact_modules = {
        path.name for path in ROOT.glob("contact_gate_*.py")
    }
    if observed_contact_modules != _EXPECTED_CONTACT_MODULES:
        raise GateError(
            "contact verifier module inventory differs; "
            f"missing={sorted(_EXPECTED_CONTACT_MODULES - observed_contact_modules)}, "
            f"extra={sorted(observed_contact_modules - _EXPECTED_CONTACT_MODULES)}"
        )
    declared = frozenset(VERIFIER_SOURCE_FILES)
    snapshots = {
        relative: _snapshot_source(relative)
        for relative in VERIFIER_SOURCE_FILES
    }
    observed = _static_local_import_closure(snapshots)
    if observed != declared:
        raise GateError(
            "contact verifier transitive Python closure differs; "
            f"missing={sorted(observed - declared)}, "
            f"stale={sorted(declared - observed)}"
        )
    lines: list[str] = []
    for relative in VERIFIER_SOURCE_FILES:
        digest = hashlib.sha256(snapshots[relative].raw).hexdigest()
        lines.append(f"{relative}:{digest}")
    for relative in VERIFIER_SOURCE_FILES:
        _assert_snapshot_unchanged(relative, snapshots[relative])
    final_contact_modules = {
        path.name for path in ROOT.glob("contact_gate_*.py")
    }
    final_observed = _static_local_import_closure(snapshots)
    if (
        final_contact_modules != observed_contact_modules
        or final_observed != observed
    ):
        raise GateError(
            "contact verifier inventory/import closure changed during root "
            "validation"
        )
    material = "\n".join(sorted(lines)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()

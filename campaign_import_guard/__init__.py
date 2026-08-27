"""Portable project-import boundary for campaign entrypoints.

The guard prevents the campaign's own packages from resolving to another copy.
It intentionally accepts virtualenv, Conda, system Python, additional search
paths and relocated/extracted workspaces; those are provenance, not scientific
eligibility conditions.
"""

from __future__ import annotations

import importlib.machinery
import os
import stat
import sys


class ImportBoundaryError(RuntimeError):
    """The running process cannot prove its import search boundary."""


_PROJECT_PACKAGES = {
    "core": "core",
    "training": "training",
    "ttbi_2d": "TTBI_2D",
}
_GUARD_PACKAGE = "campaign_import_guard"
_STARTUP_MODULES = frozenset({"sitecustomize", "usercustomize"})
_SOURCE_MANIFEST = "bundle_source_files.txt"
_IMPORT_SUFFIXES = tuple(
    sorted(
        set(importlib.machinery.all_suffixes()),
        key=len,
        reverse=True,
    )
)
_REPARSE_POINT = 0x0400
_BOUNDARY_ENFORCED = False


def _normal_absolute(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _normal_canonical(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _has_reparse_point(path: str) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    return os.path.islink(path) or bool(attributes & _REPARSE_POINT)


def _require_canonical_directory(path: str, label: str) -> str:
    absolute = os.path.abspath(path)
    if (
        not os.path.isdir(absolute)
        or _has_reparse_point(absolute)
        or _normal_absolute(absolute) != _normal_canonical(absolute)
    ):
        raise ImportBoundaryError(
            f"{label} is not one canonical, non-reparse directory: {absolute}"
        )
    return absolute


def _require_regular_file(path: str, label: str) -> None:
    absolute = os.path.abspath(path)
    try:
        info = os.lstat(absolute)
    except OSError as exc:
        raise ImportBoundaryError(
            f"{label} is unavailable: {absolute}"
        ) from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or _has_reparse_point(absolute)
        or _normal_absolute(absolute) != _normal_canonical(absolute)
        or int(getattr(info, "st_nlink", 1)) != 1
    ):
        raise ImportBoundaryError(
            f"{label} is not one canonical, unaliased regular file: {absolute}"
        )


def _candidate_name(entry: str, path: str) -> str | None:
    folded = entry.casefold()
    if os.path.isdir(path):
        return folded
    for suffix in _IMPORT_SUFFIXES:
        if folded.endswith(suffix.casefold()):
            return folded[:-len(suffix)].split(".", 1)[0]
    return None


def _installed_top_level_names(site_packages: str) -> set[str]:
    site_root = _require_canonical_directory(
        site_packages, "campaign site-packages"
    )
    names: set[str] = set()
    for entry in os.listdir(site_root):
        folded = entry.casefold()
        if (
            folded == "__pycache__"
            or folded.endswith(".dist-info")
            or folded.endswith(".data")
        ):
            continue
        candidate = _candidate_name(entry, os.path.join(site_root, entry))
        if candidate is not None:
            names.add(candidate.split(".", 1)[0])
    return names


def _root_python_modules(source_root: str) -> dict[str, str]:
    """Inventory regular top-level Python files without authorizing them."""
    modules: dict[str, str] = {}
    for entry in os.listdir(source_root):
        if not entry.casefold().endswith(".py"):
            continue
        path = os.path.join(source_root, entry)
        if not os.path.isfile(path):
            continue
        _require_regular_file(path, f"reviewed source module {entry}")
        name = entry[:-3].casefold()
        previous = modules.setdefault(name, entry)
        if previous != entry:
            raise ImportBoundaryError(
                f"case-colliding reviewed modules exist: {previous}, {entry}"
            )
    return modules


def _validate_package_module_candidates(
    package: str,
    package_name: str,
) -> None:
    reviewed = _root_python_modules(package)
    for entry in os.listdir(package):
        path = os.path.join(package, entry)
        candidate = _candidate_name(entry, path)
        expected = None if candidate is None else reviewed.get(candidate)
        if expected is not None and entry != expected:
            raise ImportBoundaryError(
                f"reviewed package {package_name} has a competing module "
                f"candidate: {path}"
            )


def _validate_project_packages(source_root: str) -> None:
    for folded, entry in _PROJECT_PACKAGES.items():
        package = os.path.join(source_root, entry)
        _require_canonical_directory(package, f"reviewed package {entry}")
        init_file = os.path.join(package, "__init__.py")
        _require_regular_file(init_file, f"reviewed package initializer {entry}")
        if entry.casefold() != folded:
            raise ImportBoundaryError(
                f"reviewed package has an invalid canonical name: {entry}"
            )
        _validate_package_module_candidates(package, entry)


def _reviewed_source_names(source_root: str) -> set[str]:
    manifest = os.path.join(source_root, _SOURCE_MANIFEST)
    _require_regular_file(manifest, "reviewed source manifest")
    try:
        with open(manifest, "rb") as handle:
            text = handle.read().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ImportBoundaryError(
            "reviewed source manifest is not UTF-8"
        ) from exc
    names: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if line == "" or line.startswith("#"):
            continue
        if (
            line != line.strip()
            or "\\" in line
            or line.startswith("/")
            or any(part in {"", ".", ".."} for part in line.split("/"))
        ):
            raise ImportBoundaryError(
                f"reviewed source manifest has an unsafe entry on line "
                f"{line_number}"
            )
        names.append(line)
    if (
        not names
        or names != sorted(names)
        or len(names) != len(set(names))
        or len(names) != len({name.casefold() for name in names})
    ):
        raise ImportBoundaryError(
            "reviewed source manifest is empty, unsorted, or ambiguous"
        )
    return set(names)


def _validate_package_source_inventory(source_root: str) -> None:
    """Bind every importable project-package source to the source manifest."""
    reviewed = _reviewed_source_names(source_root)
    for _folded, package_name in {
        **_PROJECT_PACKAGES,
        _GUARD_PACKAGE: _GUARD_PACKAGE,
    }.items():
        package = os.path.join(source_root, package_name)
        pending = [package]
        while pending:
            directory = pending.pop()
            for entry in os.scandir(directory):
                path = entry.path
                relative = os.path.relpath(path, source_root).replace(
                    os.sep, "/"
                )
                if _has_reparse_point(path):
                    raise ImportBoundaryError(
                        f"reviewed package contains a reparse path: {relative}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    if entry.name != "__pycache__":
                        pending.append(path)
                    continue
                candidate = _candidate_name(entry.name, path)
                if candidate is None:
                    continue
                if not entry.name.casefold().endswith(".py"):
                    raise ImportBoundaryError(
                        "reviewed package contains a compiled/untracked import "
                        f"candidate: {relative}"
                    )
                _require_regular_file(
                    path, f"reviewed package source {relative}"
                )
                if relative not in reviewed:
                    raise ImportBoundaryError(
                        "importable project-package source is outside the "
                        f"reviewed source manifest: {relative}"
                    )


def _validate_guard_origin(source_root: str) -> None:
    expected_package = os.path.join(source_root, _GUARD_PACKAGE)
    expected_init = os.path.join(expected_package, "__init__.py")
    _require_canonical_directory(expected_package, "campaign import guard")
    _require_regular_file(expected_init, "campaign import guard initializer")
    _validate_package_module_candidates(
        expected_package, _GUARD_PACKAGE
    )
    if _normal_canonical(__file__) != _normal_canonical(expected_init):
        raise ImportBoundaryError(
            "campaign import guard was imported from an unauthorized origin: "
            f"{__file__}"
        )


def _validate_project_resolution(source_root: str) -> None:
    for _folded, entry in _PROJECT_PACKAGES.items():
        spec = importlib.machinery.PathFinder.find_spec(entry, sys.path)
        expected_dir = os.path.join(source_root, entry)
        expected_init = os.path.join(expected_dir, "__init__.py")
        locations = tuple(
            _normal_canonical(path)
            for path in (spec.submodule_search_locations or ())
        ) if spec is not None else ()
        if (
            spec is None
            or spec.origin is None
            or _normal_canonical(spec.origin)
            != _normal_canonical(expected_init)
            or locations != (_normal_canonical(expected_dir),)
        ):
            raise ImportBoundaryError(
                f"reviewed package {entry} resolves from another origin"
            )


def validate_source_tree(
    source_root: str,
    site_packages: str,
) -> dict[str, int]:
    """Validate source candidates without consulting the process search path."""
    root = _require_canonical_directory(source_root, "campaign source root")
    site_root = _require_canonical_directory(
        site_packages, "campaign site-packages"
    )
    _validate_guard_origin(root)
    _validate_project_packages(root)
    _validate_package_source_inventory(root)
    root_modules = _root_python_modules(root)
    installed_names = _installed_top_level_names(site_root)
    stdlib_names = {
        name.casefold()
        for name in sys.stdlib_module_names
    }
    protected_names = stdlib_names | installed_names | _STARTUP_MODULES
    reviewed_packages = {
        **_PROJECT_PACKAGES,
        _GUARD_PACKAGE: _GUARD_PACKAGE,
    }

    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        candidate = _candidate_name(entry, path)
        if candidate is None:
            continue
        expected_module = root_modules.get(candidate)
        expected_package = reviewed_packages.get(candidate)
        if expected_module is not None and entry != expected_module:
            raise ImportBoundaryError(
                "reviewed module has a competing import candidate: "
                f"{path}"
            )
        if expected_package is not None and entry != expected_package:
            raise ImportBoundaryError(
                "reviewed package has a competing import candidate: "
                f"{path}"
            )
        if candidate in protected_names and expected_package is None:
            raise ImportBoundaryError(
                "authorized source root contains an import shadow for "
                f"{candidate!r}: {path}"
            )
        if candidate in installed_names and expected_package is not None:
            raise ImportBoundaryError(
                "campaign site-packages contains a competing project package: "
                f"{candidate!r}"
            )

    return {
        "stdlib_names": len(stdlib_names),
        "installed_top_level_names": len(installed_names),
        "root_python_modules": len(root_modules),
    }


def enforce_import_boundary() -> dict[str, int]:
    """Ensure project packages resolve from this workspace.

    Dependency availability is checked by :mod:`core.environment`.  Exact
    ``sys.path`` layout, environment manager and physical path aliases are not
    prescribed because they do not define the physics or learning protocol.
    """
    global _BOUNDARY_ENFORCED
    source_root = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
    if not any(
        isinstance(entry, str)
        and _normal_canonical(entry if entry else os.getcwd())
        == _normal_canonical(source_root)
        for entry in sys.path
    ):
        sys.path.insert(0, source_root)
    _validate_project_resolution(source_root)
    result = {
        "project_packages": len(_PROJECT_PACKAGES),
        "search_paths": len(sys.path),
        "pythonpath_present": int("PYTHONPATH" in os.environ),
        "pythonhome_present": int("PYTHONHOME" in os.environ),
    }
    _BOUNDARY_ENFORCED = True
    return result

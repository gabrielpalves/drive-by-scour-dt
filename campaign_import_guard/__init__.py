"""Fail-closed import boundary for campaign and evidence entrypoints.

The package intentionally lives at the repository top level and contains a
regular ``__init__.py``.  Python therefore resolves it before a same-name
module file, unlike the former guard below the namespace package ``core``.
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


def _expected_search_paths(
    source_root: str,
    site_packages: str,
) -> list[str]:
    base = _require_canonical_directory(
        sys.base_prefix, "CPython base prefix"
    )
    prefix = _require_canonical_directory(
        sys.prefix, "campaign virtual-environment prefix"
    )
    python_tag = f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    return [
        _normal_canonical(source_root),
        _normal_canonical(os.path.join(base, python_tag)),
        _normal_canonical(os.path.join(base, "DLLs")),
        _normal_canonical(os.path.join(base, "Lib")),
        _normal_canonical(base),
        _normal_canonical(prefix),
        _normal_canonical(site_packages),
    ]


def _validate_import_search_path(
    source_root: str,
    site_packages: str,
) -> None:
    resolved: list[str] = []
    for entry in sys.path:
        if not isinstance(entry, str):
            raise ImportBoundaryError("sys.path contains a non-text entry")
        selected = entry if entry else os.getcwd()
        if _normal_absolute(selected) != _normal_canonical(selected):
            raise ImportBoundaryError(
                f"sys.path contains a reparse/aliased entry: {entry!r}"
            )
        resolved.append(_normal_canonical(selected))

    expected = _expected_search_paths(source_root, site_packages)
    if resolved != expected:
        raise ImportBoundaryError(
            "Python import-path order differs from the campaign contract: "
            f"{resolved!r}"
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


def _path_is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath(
            (_normal_canonical(path), _normal_canonical(root))
        ) == _normal_canonical(root)
    except (OSError, ValueError):
        return False


def _originless_child_has_authenticated_parent(
    module_name: str,
    required_root: str,
    *,
    allow_special_origin: bool,
) -> bool:
    """Recognize extension-created child modules through their loaded parent."""
    parent_name = module_name
    while "." in parent_name:
        parent_name = parent_name.rsplit(".", 1)[0]
        parent = sys.modules.get(parent_name)
        if parent is None:
            continue
        spec = getattr(parent, "__spec__", None)
        spec_origin = getattr(spec, "origin", None)
        if allow_special_origin and spec_origin in {"built-in", "frozen"}:
            return True
        for origin in (getattr(parent, "__file__", None), spec_origin):
            if (
                isinstance(origin, str)
                and os.path.isabs(origin)
                and os.path.isfile(origin)
                and _normal_absolute(origin) == _normal_canonical(origin)
                and _path_is_within(origin, required_root)
            ):
                return True
    return False


def _validate_loaded_module_origins(
    source_root: str,
    site_packages: str,
) -> None:
    """Reject modules loaded before the guard from unauthorized origins."""
    base = os.path.abspath(sys.base_prefix)
    prefix = os.path.abspath(sys.prefix)
    authorized_roots = (source_root, base, prefix)
    root_modules = _root_python_modules(source_root)
    reviewed_sources = _reviewed_source_names(source_root)
    manifested_root_modules = {
        folded: entry
        for folded, entry in root_modules.items()
        if entry in reviewed_sources
    }
    installed_names = _installed_top_level_names(site_packages)
    stdlib_names = {name.casefold() for name in sys.stdlib_module_names}
    project_packages = {
        **_PROJECT_PACKAGES,
        _GUARD_PACKAGE: _GUARD_PACKAGE,
    }
    protected_names = (
        set(manifested_root_modules)
        | set(project_packages)
        | installed_names
        | stdlib_names
        | _STARTUP_MODULES
    )

    for module_name, module in tuple(sys.modules.items()):
        top_name = module_name.split(".", 1)[0]
        folded_top = top_name.casefold()
        if module is None:
            if folded_top in protected_names:
                raise ImportBoundaryError(
                    f"protected module is a preloaded failure sentinel: "
                    f"{module_name!r}"
                )
            continue
        if folded_top in _STARTUP_MODULES:
            raise ImportBoundaryError(
                f"external Python startup customizer was preloaded: "
                f"{module_name!r}"
            )
        required_root = None
        if folded_top in project_packages:
            required_root = os.path.join(
                source_root, project_packages[folded_top]
            )
        elif folded_top in manifested_root_modules:
            required_root = source_root
        elif folded_top in stdlib_names:
            required_root = base
        elif folded_top in installed_names:
            required_root = site_packages
        authenticated_parent = (
            required_root is not None
            and _originless_child_has_authenticated_parent(
                module_name,
                required_root,
                allow_special_origin=folded_top in stdlib_names,
            )
        )

        spec = getattr(module, "__spec__", None)
        spec_origin = getattr(spec, "origin", None)
        file_origin = getattr(module, "__file__", None)
        special_origin = spec_origin in {"built-in", "frozen"}
        origins: list[str] = []
        for label, origin in (
            ("__file__", file_origin),
            ("spec.origin", spec_origin),
        ):
            if origin is None or origin in {"built-in", "frozen"}:
                continue
            if not isinstance(origin, str) or not os.path.isabs(origin):
                if authenticated_parent and spec is None:
                    continue
                raise ImportBoundaryError(
                    f"preloaded module {module_name!r} has invalid "
                    f"{label}: {origin!r}"
                )
            if (
                not os.path.isfile(origin)
                or _normal_absolute(origin) != _normal_canonical(origin)
            ):
                raise ImportBoundaryError(
                    f"preloaded module {module_name!r} has a noncanonical "
                    f"{label}: {origin!r}"
                )
            origins.append(origin)

        locations = getattr(spec, "submodule_search_locations", None)
        location_paths: list[str] = []
        if locations is not None:
            for location in locations:
                if not isinstance(location, str) or not os.path.isabs(location):
                    raise ImportBoundaryError(
                        f"preloaded package {module_name!r} has an invalid "
                        f"search location: {location!r}"
                    )
                if (
                    not os.path.isdir(location)
                    or _normal_absolute(location)
                    != _normal_canonical(location)
                ):
                    raise ImportBoundaryError(
                        f"preloaded package {module_name!r} has a "
                        f"noncanonical search location: {location!r}"
                    )
                location_paths.append(location)

        paths = origins + location_paths
        if any(
            not any(_path_is_within(path, root) for root in authorized_roots)
            for path in paths
        ):
            raise ImportBoundaryError(
                f"preloaded module {module_name!r} came from outside the "
                "authorized campaign roots"
            )
        if (
            folded_top in root_modules
            and folded_top not in manifested_root_modules
        ):
            raise ImportBoundaryError(
                f"unmanifested repository-root module was preloaded: "
                f"{module_name!r}"
            )
        if folded_top not in protected_names:
            continue
        authenticated_originless_child = (
            not paths
            and authenticated_parent
        )
        if not paths and not (
            (special_origin and folded_top in stdlib_names)
            or authenticated_originless_child
        ):
            raise ImportBoundaryError(
                f"protected preloaded module has no authenticated origin: "
                f"{module_name!r}"
            )

        if folded_top in project_packages:
            canonical_top = project_packages[folded_top]
            project_root = os.path.join(source_root, canonical_top)
            if top_name != canonical_top or any(
                not _path_is_within(path, project_root)
                for path in paths
            ):
                raise ImportBoundaryError(
                    f"preloaded project package {module_name!r} came from "
                    "another origin"
                )
            if module_name == top_name:
                expected_init = os.path.join(project_root, "__init__.py")
                if not origins or any(
                    _normal_canonical(origin)
                    != _normal_canonical(expected_init)
                    for origin in origins
                ):
                    raise ImportBoundaryError(
                        f"preloaded project package {module_name!r} does not "
                        "use its reviewed initializer"
                    )
        elif folded_top in manifested_root_modules:
            expected_name = manifested_root_modules[folded_top][:-3]
            expected_file = os.path.join(
                source_root, manifested_root_modules[folded_top]
            )
            if (
                top_name != expected_name
                or module_name != top_name
                or not origins
                or any(
                    _normal_canonical(origin)
                    != _normal_canonical(expected_file)
                    for origin in origins
                )
            ):
                raise ImportBoundaryError(
                    f"preloaded reviewed module {module_name!r} came from "
                    "another origin"
                )
        elif folded_top in stdlib_names:
            if paths and any(
                not _path_is_within(path, base)
                for path in paths
            ):
                raise ImportBoundaryError(
                    f"preloaded standard-library module {module_name!r} "
                    "came from another origin"
                )
        elif folded_top in installed_names and any(
            not _path_is_within(path, site_packages)
            for path in paths
        ):
            raise ImportBoundaryError(
                f"preloaded installed module {module_name!r} came from "
                "another origin"
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
    """Enforce the complete boundary before scientific imports occur."""
    global _BOUNDARY_ENFORCED
    for variable in ("PYTHONPATH", "PYTHONHOME"):
        if variable in os.environ:
            raise ImportBoundaryError(
                f"{variable} must be absent before campaign imports"
            )
    source_root = os.path.dirname(os.path.dirname(__file__))
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    _validate_import_search_path(source_root, site_packages)
    result = validate_source_tree(source_root, site_packages)
    _validate_loaded_module_origins(source_root, site_packages)
    _validate_project_resolution(source_root)
    _BOUNDARY_ENFORCED = True
    return result

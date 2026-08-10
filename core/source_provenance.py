"""Content-addressed provenance for reviewed generator and Python sources.

The tracked ``bundle_source_files.txt`` file is the explicit boundary of the
campaign package.  These helpers turn the relevant regular-file bytes into
canonical SHA-256 roots:

* ``generator_source_root`` covers every listed ``scour_MATLAB/`` file,
  including MATLAB code and numerical assets;
* ``python_runtime_source_root`` covers every listed Python file plus the
  Python/MATLAB environment lock and pinned requirements.

Training bundles no longer rewrite scientific presets in the Python driver.
They carry an external, content-addressed ``TTBI_TRAINING_JOB_MANIFEST`` whose
exact jobs are validated against ``core.paper1_dispatch``.  Every driver byte
is therefore identity-bearing; there is no bundle-local source-normalisation
exception.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import unicodedata


REPO = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = "bundle_source_files.txt"
ENVIRONMENT_LOCK = "environment/campaign-py313-cu128.json"
REQUIREMENTS_LOCK = "requirements-campaign-py313-cu128.txt"
DRIVER = "comprehensive_ablation_multidamage.py"
REGULAR_PROJECT_PACKAGES = (
    "campaign_import_guard",
    "core",
    "training",
    "TTBI_2D",
)
_REPARSE_POINT = 0x0400
_GENERATED_MICRO = re.compile(
    r"^micro_A00_qualification_[A-Za-z0-9_]+\.m$"
)
_IMPORT_CACHE: dict[tuple[str, str], frozenset[str]] = {}
_DYNAMIC_NAMESPACE_ROOTS = frozenset({"builtins", "importlib", "runpy"})
_DYNAMIC_NAMESPACE_PREFIXES = (
    "builtins",
    "importlib",
    "importlib.machinery",
    "importlib.util",
    "runpy",
)
_SOURCE_LOADER_MEMBERS = frozenset(
    {
        "ExtensionFileLoader",
        "FileFinder",
        "SourcelessFileLoader",
        "SourceFileLoader",
        "create_module",
        "exec_module",
        "get_code",
        "get_source",
        "load_module",
        "module_from_spec",
        "run_module",
        "run_path",
        "spec_from_file_location",
    }
)
_DYNAMIC_BUILTINS = frozenset({"compile", "eval", "exec"})
_AST_HARNESS_EXEC_ALLOWLIST = frozenset(
    {
        "check_campaign_controls.py",
        "check_execution_blocking.py",
        "check_generation_contract.py",
        "check_generation_release_comparison.py",
        "check_statistical_inference.py",
    }
)
_REFLECTIVE_LOADER_MEMBERS = (
    _SOURCE_LOADER_MEMBERS
    | _DYNAMIC_BUILTINS
    | frozenset({"__import__", "import_module"})
)


class SourceProvenanceError(RuntimeError):
    """The reviewed source boundary is absent, unsafe, or incomplete."""


@dataclass(frozen=True)
class SourceRoot:
    sha256: str
    file_count: int
    files: tuple[str, ...]
    digest_lines: str


@dataclass(frozen=True)
class _FileSnapshot:
    """Exact bytes and pathname identity consumed by one source-root pass."""

    path: Path
    raw: bytes
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class RepositorySourceSnapshot:
    """One coherent capture of all source bytes used by current policy.

    This is a static reviewed-source snapshot, not a process sandbox.  It
    gives policy consumers the exact bytes used for both source roots and for
    any source literal they parse, then lets them reassert those pathname
    identities immediately before returning an evidentiary result.
    """

    repo: Path
    manifest_names: tuple[str, ...]
    manifest_snapshot: _FileSnapshot
    generator: SourceRoot
    python_runtime: SourceRoot
    snapshots: tuple[tuple[str, _FileSnapshot], ...]

    def source_bytes(self, name: str) -> bytes:
        """Return captured bytes for one exact manifest-relative name."""
        for observed, snapshot in self.snapshots:
            if observed == name:
                return snapshot.raw
        raise SourceProvenanceError(
            f"source is outside the coherent policy capture: {name}"
        )

    def assert_unchanged(self) -> None:
        """Fail if source bytes, identities, manifest, or inventories drift."""
        _assert_repository_snapshot_unchanged(self)


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(getattr(info, "st_dev", 0)),
        int(getattr(info, "st_ino", 0)),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(getattr(info, "st_nlink", 1)),
    )


def _validate_regular_package_inventory(
    root: Path,
    manifest_names: set[str],
) -> None:
    """Require every importable project-package source to be authenticated."""
    for package_name in REGULAR_PROJECT_PACKAGES:
        package = root / package_name
        package_info = _checked_tree_entry(root, package, package_name)
        if not stat.S_ISDIR(package_info.st_mode):
            raise SourceProvenanceError(
                f"missing regular reviewed package: {package_name}"
            )
        for path in package.rglob("*"):
            relative = path.relative_to(root)
            try:
                info = path.lstat()
            except OSError as exc:
                raise SourceProvenanceError(
                    f"reviewed package path is unavailable: "
                    f"{relative.as_posix()}"
                ) from exc
            if (
                path.is_symlink()
                or int(getattr(info, "st_file_attributes", 0))
                & _REPARSE_POINT
            ):
                raise SourceProvenanceError(
                    f"reviewed package contains a reparse path: "
                    f"{relative.as_posix()}"
                )
            if (
                path.is_file()
                and path.suffix.casefold() == ".py"
                and relative.as_posix() not in manifest_names
            ):
                raise SourceProvenanceError(
                    "importable project-package source is outside the "
                    f"reviewed manifest: {relative.as_posix()}"
                )


def _checked_tree_entry(root: Path, path: Path, label: str):
    """Return ``lstat`` only for a canonical, non-reparse tree entry."""
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise SourceProvenanceError(
            f"reviewed source path is unavailable: {label}"
        ) from exc
    if (
        path.is_symlink()
        or int(getattr(info, "st_file_attributes", 0)) & _REPARSE_POINT
        or os.path.normcase(str(absolute))
        != os.path.normcase(str(resolved))
    ):
        raise SourceProvenanceError(
            f"reviewed source path is a symlink/reparse/canonical alias: {label}"
        )

    current = root
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError as exc:
        raise SourceProvenanceError(
            f"reviewed source escaped its repository root: {label}"
        ) from exc
    for part in relative_parts:
        current = current / part
        component = current.lstat()
        if (
            current.is_symlink()
            or int(getattr(component, "st_file_attributes", 0))
            & _REPARSE_POINT
        ):
            raise SourceProvenanceError(
                f"reviewed source contains a reparse component: {label}"
            )
    return info


def _validate_matlab_source_inventory(
    root: Path,
    manifest_names: set[str],
) -> None:
    """Close the MATLAB filesystem except reviewed generated-output paths."""
    matlab_root = root / "scour_MATLAB"
    root_info = _checked_tree_entry(root, matlab_root, "scour_MATLAB")
    if not stat.S_ISDIR(root_info.st_mode):
        raise SourceProvenanceError("scour_MATLAB is not a regular directory")

    observed: set[str] = set()
    pending = [matlab_root]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            relative = path.relative_to(root).as_posix()
            matlab_relative = path.relative_to(matlab_root)
            if (
                len(matlab_relative.parts) == 1
                and matlab_relative.name
                in {"Results", "Results_sensitivity"}
            ):
                continue
            info = _checked_tree_entry(root, path, relative)
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise SourceProvenanceError(
                    f"MATLAB source entry is not a regular file: {relative}"
                )
            if (
                len(matlab_relative.parts) == 1
                and (
                    matlab_relative.name == "micro_A00_smoke.m"
                    or _GENERATED_MICRO.fullmatch(matlab_relative.name)
                )
            ):
                continue
            observed.add(relative)

    reviewed = {
        name for name in manifest_names
        if name.startswith("scour_MATLAB/")
    }
    missing = sorted(observed - reviewed)
    stale = sorted(reviewed - observed)
    if missing or stale:
        raise SourceProvenanceError(
            "MATLAB filesystem/manifest closure failed; "
            f"unmanifested={missing}, missing={stale}"
        )


def _module_file_candidates(module: str) -> tuple[str, ...]:
    parts = module.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        return ()
    base = "/".join(parts)
    candidates: list[str] = []
    for index in range(1, len(parts)):
        candidates.append("/".join(parts[:index]) + "/__init__.py")
    candidates.extend((base + ".py", base + "/__init__.py"))
    return tuple(dict.fromkeys(candidates))


def _static_string(node: ast.AST) -> str | None:
    """Fold only literal string concatenation used in attribute probes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left)
        right = _static_string(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        pieces: list[str] = []
        for value in node.values:
            if not isinstance(value, ast.Constant) or not isinstance(
                value.value, str
            ):
                return None
            pieces.append(value.value)
        return "".join(pieces)
    return None


def _dynamic_namespace_aliases(tree: ast.AST) -> dict[str, str]:
    """Map explicit import aliases to importlib/builtins/runpy namespaces."""
    aliases: dict[str, str] = {
        "builtins": "builtins",
        "importlib": "importlib",
        "runpy": "runpy",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                root = item.name.split(".", 1)[0]
                if root not in _DYNAMIC_NAMESPACE_ROOTS:
                    continue
                local = item.asname or root
                aliases[local] = item.name if item.asname else root
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root not in _DYNAMIC_NAMESPACE_ROOTS:
                continue
            for item in node.names:
                full = f"{module}.{item.name}" if module else item.name
                if full in _DYNAMIC_NAMESPACE_PREFIXES:
                    aliases[item.asname or item.name] = full
    return aliases


def _namespace_name(
    node: ast.AST,
    aliases: dict[str, str],
) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _namespace_name(node.value, aliases)
        return f"{base}.{node.attr}" if base else None
    return None


def _dangerous_namespace_expression(
    node: ast.AST,
    aliases: dict[str, str],
) -> str | None:
    namespace = _namespace_name(node, aliases)
    if namespace and namespace.split(".", 1)[0] in _DYNAMIC_NAMESPACE_ROOTS:
        return namespace
    return None


def _reject_unsafe_loader_indirection(
    tree: ast.AST,
    source_name: str,
) -> None:
    """Reject source-loading forms outside the reviewed static import model.

    This is deliberately a *static reviewed import-closure* guard, not a
    general Python sandbox.  Direct literal importlib.import_module() and
    __import__() calls are resolved below.  Loader factories, reflective
    namespace access, runpy/importlib file loaders, and indirect
    exec/eval/compile references fail closed because their target bytes cannot
    be placed in the manifest closure from syntax alone.  The only direct
    compile/exec exception is an exact nominal allowlist of five authenticated
    audit harnesses that execute reviewed AST slices/mutants; it is not
    available to production/runtime modules.  ``eval`` has no exception.
    """
    namespace_aliases = _dynamic_namespace_aliases(tree)
    import_module_aliases: set[str] = set()
    builtin_import_aliases = {"__import__"}
    forbidden_aliases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 0:
            continue
        module = node.module or ""
        for item in node.names:
            local = item.asname or item.name
            if module == "importlib" and item.name == "import_module":
                import_module_aliases.add(local)
            elif module == "builtins" and item.name == "__import__":
                builtin_import_aliases.add(local)
            elif (
                item.name in _SOURCE_LOADER_MEMBERS
                or item.name in _DYNAMIC_BUILTINS
            ):
                forbidden_aliases.add(local)

    direct_call_functions = {
        id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }

    def call_module_name(call: ast.Call) -> str | None:
        positional = call.args[0] if call.args else None
        keyword_values = [
            keyword.value
            for keyword in call.keywords
            if keyword.arg in {"name", "module"}
        ]
        if positional is not None and keyword_values:
            return None
        value = positional if positional is not None else (
            keyword_values[0] if len(keyword_values) == 1 else None
        )
        return _static_string(value) if value is not None else None

    def returned_namespace(node: ast.AST) -> str | None:
        direct = _dangerous_namespace_expression(node, namespace_aliases)
        if direct:
            return direct
        if isinstance(node, ast.Subscript):
            key = _static_string(node.slice)
            if (
                key in _DYNAMIC_NAMESPACE_PREFIXES
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "modules"
            ):
                return key
        if not isinstance(node, ast.Call):
            return None
        name = call_module_name(node)
        if name not in _DYNAMIC_NAMESPACE_PREFIXES:
            return None
        function = node.func
        is_import_module = (
            isinstance(function, ast.Name)
            and function.id in import_module_aliases
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
            and _dangerous_namespace_expression(
                function.value, namespace_aliases
            ) == "importlib"
        )
        is_builtin_import = (
            isinstance(function, ast.Name)
            and function.id in builtin_import_aliases
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "__import__"
            and _dangerous_namespace_expression(
                function.value, namespace_aliases
            ) == "builtins"
        )
        return name if is_import_module or is_builtin_import else None

    def reject(detail: str) -> None:
        raise SourceProvenanceError(
            f"{source_name} uses {detail}; the static reviewed import closure "
            "cannot authenticate its target bytes"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in (
                _DYNAMIC_BUILTINS
            ):
                builtin = node.func.id
                allowed_harness_call = (
                    builtin in {"compile", "exec"}
                    and source_name in _AST_HARNESS_EXEC_ALLOWLIST
                )
                if not allowed_harness_call:
                    reject(
                        f"bare {builtin}() outside the authenticated AST-"
                        "harness allowlist"
                    )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"attrgetter", "methodcaller"}
            ):
                requested = (
                    _static_string(node.args[0]) if node.args else None
                )
                if (
                    requested is None
                    or any(
                        part in _REFLECTIVE_LOADER_MEMBERS
                        for part in requested.split(".")
                    )
                ):
                    reject(
                        f"reflective loader factory {node.func.attr}()"
                    )
            if isinstance(node.func, ast.Name) and node.func.id in {
                "globals",
                "locals",
            }:
                reject(f"reflective namespace access {node.func.id}()")
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_aliases:
                reject(f"aliased source loader {node.func.id}()")
            if isinstance(node.func, ast.Attribute):
                namespace = returned_namespace(node.func.value)
                member = node.func.attr
                if member == "import_module" and namespace != "importlib":
                    reject("indirect import_module() dispatch")
                if member == "__import__" and namespace != "builtins":
                    reject("indirect __import__() dispatch")
                if member in _SOURCE_LOADER_MEMBERS:
                    reject(f"source loader {member}()")
                if (
                    namespace
                    and namespace.split(".", 1)[0] == "builtins"
                    and member in _DYNAMIC_BUILTINS
                ):
                    reject(f"dynamic builtin {member}()")
                if (
                    namespace
                    and namespace.split(".", 1)[0] == "runpy"
                    and member in {"run_module", "run_path"}
                ):
                    reject(f"runpy loader {member}()")
            if isinstance(node.func, ast.Name) and node.func.id in {
                "getattr",
                "vars",
            }:
                first = node.args[0] if node.args else None
                namespace = (
                    returned_namespace(first)
                    if first is not None
                    else None
                )
                if namespace:
                    requested = (
                        _static_string(node.args[1])
                        if node.func.id == "getattr" and len(node.args) > 1
                        else None
                    )
                    suffix = f" for {requested!r}" if requested else ""
                    reject(
                        f"reflective {node.func.id}({namespace}, ...){suffix}"
                    )
                if (
                    node.func.id == "getattr"
                    and len(node.args) > 1
                    and _static_string(node.args[1])
                    in _REFLECTIVE_LOADER_MEMBERS
                ):
                    reject("reflective getattr() of a source-loader member")

        if isinstance(node, ast.Attribute):
            namespace = returned_namespace(node.value)
            if namespace and node.attr == "__dict__":
                reject(f"reflective {namespace}.__dict__ access")
            if (
                namespace
                and node.attr
                in {"__class__", "__getattribute__", "__loader__"}
            ):
                reject(f"reflective namespace attribute {node.attr!r}")
            if (
                node.attr in _SOURCE_LOADER_MEMBERS | {"import_module"}
                and id(node) not in direct_call_functions
            ):
                reject(f"source-loader factory/reference {node.attr!r}")

        if isinstance(node, ast.Subscript):
            namespace = returned_namespace(node.value)
            # Indexing a namespace mapping itself is reflective dispatch and
            # remains forbidden.  A public data attribute below that
            # namespace can legitimately be a sequence, however; for example
            # check_import_path_guard indexes
            # importlib.machinery.EXTENSION_SUFFIXES.  Treating every such
            # attribute as a namespace made that reviewed positive control
            # impossible without improving loader closure.
            if namespace in _DYNAMIC_NAMESPACE_PREFIXES:
                key = _static_string(node.slice)
                suffix = f"[{key!r}]" if key is not None else "[...]"
                reject(f"reflective namespace subscript {namespace}{suffix}")
            key = _static_string(node.slice)
            reflective_mapping = (
                isinstance(node.value, ast.Name)
                and node.value.id == "__builtins__"
            ) or (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in {"globals", "locals", "vars"}
            )
            if reflective_mapping and (
                key is None or key in _REFLECTIVE_LOADER_MEMBERS
            ):
                reject("reflective mapping subscript for a source loader")
            if (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "__dict__"
                and (key is None or key in _REFLECTIVE_LOADER_MEMBERS)
            ):
                reject("reflective __dict__ subscript for a source loader")

        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            is_direct_function = id(node) in direct_call_functions
            if (
                node.id == "__builtins__"
                or node.id in forbidden_aliases
                or (
                    node.id in _DYNAMIC_BUILTINS
                    and not is_direct_function
                )
                or (
                    node.id in import_module_aliases | builtin_import_aliases
                    and not is_direct_function
                )
            ):
                reject(f"indirect source-loader reference {node.id!r}")

        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            value = node.value
            namespace = returned_namespace(value)
            if namespace:
                reject(f"namespace alias/factory assignment from {namespace}")


def _literal_call_argument(
    node: ast.Call,
    position: int,
    keyword: str,
    source_name: str,
) -> str:
    """Return one unambiguous literal string call argument."""
    positional = node.args[position] if len(node.args) > position else None
    keywords = [item.value for item in node.keywords if item.arg == keyword]
    if positional is not None and keywords:
        raise SourceProvenanceError(
            f"{source_name} repeats dynamic-import argument {keyword!r}"
        )
    value = positional if positional is not None else (
        keywords[0] if len(keywords) == 1 else None
    )
    if (
        len(keywords) > 1
        or not isinstance(value, ast.Constant)
        or not isinstance(value.value, str)
        or not value.value
    ):
        raise SourceProvenanceError(
            f"{source_name} has a nonliteral dynamic import"
        )
    return value.value


def _resolve_import_module_call(node: ast.Call, source_name: str) -> str:
    """Resolve importlib.import_module(name[, package]) without executing it."""
    if len(node.args) > 2 or any(
        keyword.arg not in {"name", "package"} for keyword in node.keywords
    ):
        raise SourceProvenanceError(
            f"{source_name} has a non-canonical import_module call"
        )
    name = _literal_call_argument(node, 0, "name", source_name)
    if not name.startswith("."):
        if len(node.args) > 1 or any(
            keyword.arg == "package" for keyword in node.keywords
        ):
            raise SourceProvenanceError(
                f"{source_name} gives an unused package to an absolute "
                "dynamic import"
            )
        resolved = name
    else:
        package = _literal_call_argument(node, 1, "package", source_name)
        try:
            resolved = importlib.util.resolve_name(name, package)
        except (ImportError, ValueError) as exc:
            raise SourceProvenanceError(
                f"{source_name} has an invalid relative dynamic import"
            ) from exc
    if not resolved or any(
        not part.isidentifier() for part in resolved.split(".")
    ):
        raise SourceProvenanceError(
            f"{source_name} has a non-canonical dynamic module name"
        )
    return resolved


def _dynamic_import_aliases(
    tree: ast.AST,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Collect explicit aliases of importlib/builtins and their loaders."""
    importlib_aliases = {"importlib"}
    builtins_aliases: set[str] = set()
    import_module_aliases: set[str] = set()
    builtin_loader_aliases = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                if alias.name == "importlib":
                    importlib_aliases.add(local)
                elif alias.name == "builtins":
                    builtins_aliases.add(local)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local = alias.asname or alias.name
                if node.level == 0 and (
                    node.module == "importlib"
                    and alias.name == "import_module"
                ):
                    import_module_aliases.add(local)
                elif node.level == 0 and (
                    node.module == "builtins" and alias.name == "__import__"
                ):
                    builtin_loader_aliases.add(local)

    # Preserve loader kind while following simple assignment aliases.  The
    # distinction matters for relative import_module(name, package) calls;
    # treating that alias as __import__ would silently apply the wrong rules.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            target_names = {
                target.id for target in targets if isinstance(target, ast.Name)
            }
            if not target_names:
                continue
            import_module_value = (
                isinstance(value, ast.Name)
                and value.id in import_module_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id in importlib_aliases
                and value.attr == "import_module"
            )
            builtin_loader_value = (
                isinstance(value, ast.Name)
                and value.id in builtin_loader_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_aliases
                and value.attr == "__import__"
            )
            if import_module_value:
                previous = len(import_module_aliases)
                import_module_aliases.update(target_names)
                changed |= len(import_module_aliases) != previous
            if builtin_loader_value:
                previous = len(builtin_loader_aliases)
                builtin_loader_aliases.update(target_names)
                changed |= len(builtin_loader_aliases) != previous
    return (
        importlib_aliases,
        builtins_aliases,
        import_module_aliases,
        builtin_loader_aliases,
    )


def _imported_modules(tree: ast.AST, source_name: str) -> set[str]:
    """Collect ordinary and statically resolvable dynamic imports."""
    _reject_unsafe_loader_indirection(tree, source_name)
    package_parts = source_name.split("/")[:-1]
    imported: set[str] = set()
    (
        importlib_aliases,
        builtins_aliases,
        import_module_aliases,
        builtin_loader_aliases,
    ) = _dynamic_import_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - (node.level - 1)
                if keep <= 0:
                    raise SourceProvenanceError(
                        f"{source_name} has an invalid relative import"
                    )
                prefix = package_parts[:keep]
                base_parts = [
                    *prefix,
                    *((node.module or "").split(".") if node.module else ()),
                ]
            else:
                base_parts = (
                    node.module.split(".") if node.module else []
                )
            base = ".".join(base_parts)
            if base:
                imported.add(base)
            for alias in node.names:
                candidate = ".".join((*base_parts, alias.name))
                if candidate:
                    imported.add(candidate)
        elif isinstance(node, ast.Call):
            import_module_call = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in importlib_aliases
                and node.func.attr == "import_module"
            ) or (
                isinstance(node.func, ast.Name)
                and node.func.id in import_module_aliases
            )
            builtin_import_call = (
                isinstance(node.func, ast.Name)
                and node.func.id in builtin_loader_aliases
            ) or (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in builtins_aliases
                and node.func.attr == "__import__"
            )
            if import_module_call:
                imported.add(_resolve_import_module_call(node, source_name))
            elif builtin_import_call:
                if len(node.args) > 1 or any(
                    keyword.arg != "name" for keyword in node.keywords
                ):
                    raise SourceProvenanceError(
                        f"{source_name} has a non-canonical __import__ call"
                    )
                name = _literal_call_argument(node, 0, "name", source_name)
                if name.startswith(".") or any(
                    not part.isidentifier() for part in name.split(".")
                ):
                    raise SourceProvenanceError(
                        f"{source_name} has an invalid/non-canonical "
                        "__import__ call"
                    )
                imported.add(name)
    return imported


def _validate_python_import_closure(
    root: Path,
    manifest_names: set[str],
    source_snapshots: dict[str, _FileSnapshot],
) -> None:
    """Require every statically resolvable project import to be manifested."""
    reviewed_python = sorted(
        name for name in manifest_names if name.endswith(".py")
    )
    for source_name in reviewed_python:
        payload = source_snapshots[source_name].raw
        cache_key = (source_name, _sha256_bytes(payload))
        imported = _IMPORT_CACHE.get(cache_key)
        if imported is None:
            try:
                tree = ast.parse(payload, filename=source_name)
            except SyntaxError as exc:
                raise SourceProvenanceError(
                    f"reviewed Python source cannot be parsed: {source_name}"
                ) from exc
            imported = frozenset(_imported_modules(tree, source_name))
            _IMPORT_CACHE[cache_key] = imported
        for module in sorted(imported):
            for candidate in _module_file_candidates(module):
                target = root.joinpath(*candidate.split("/"))
                if target.exists() and candidate not in manifest_names:
                    raise SourceProvenanceError(
                        f"{source_name} imports project source outside the "
                        f"reviewed manifest: {candidate}"
                    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_names(root: Path) -> tuple[tuple[str, ...], _FileSnapshot]:
    manifest = root / SOURCE_MANIFEST
    manifest_snapshot = _regular_snapshot(root, SOURCE_MANIFEST)
    try:
        text = manifest_snapshot.raw.decode("utf-8")
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
    manifest_names = set(names)
    _validate_regular_package_inventory(root, manifest_names)
    _validate_matlab_source_inventory(root, manifest_names)
    return tuple(names), manifest_snapshot


def _regular_snapshot(root: Path, name: str) -> _FileSnapshot:
    """Read one single-link file once and bind bytes to its pathname."""
    path = root.joinpath(*name.split("/"))
    info = _checked_tree_entry(root, path, name)
    if not stat.S_ISREG(info.st_mode) or int(info.st_nlink) != 1:
        raise SourceProvenanceError(
            f"reviewed source entry is not one single-link regular file: {name}"
        )
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceProvenanceError(
            f"cannot securely open reviewed source entry: {name}"
        ) from exc
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
        raise SourceProvenanceError(
            f"cannot snapshot reviewed source entry: {name}"
        ) from exc
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    path_after = _checked_tree_entry(root, path, name)
    identity = _stat_identity(before)
    if (
        identity != _stat_identity(info)
        or identity != _stat_identity(after)
        or identity != _stat_identity(path_after)
        or not stat.S_ISREG(before.st_mode)
        or identity[-1] != 1
        or len(raw) != before.st_size
    ):
        raise SourceProvenanceError(
            f"reviewed source entry changed while being read: {name}"
        )
    return _FileSnapshot(path=path, raw=raw, identity=identity)


def _assert_snapshot_unchanged(
    root: Path,
    name: str,
    snapshot: _FileSnapshot,
) -> None:
    current = _regular_snapshot(root, name)
    if current.identity != snapshot.identity or current.raw != snapshot.raw:
        raise SourceProvenanceError(
            f"reviewed source entry changed during root calculation: {name}"
        )


def _regular_bytes(root: Path, name: str) -> bytes:
    """Compatibility helper for callers that need one authenticated read."""
    return _regular_snapshot(root, name).raw


def _authenticated_driver_bytes(payload: bytes) -> bytes:
    """Validate the driver text boundary without changing a single byte."""

    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceProvenanceError(f"{DRIVER} is not UTF-8") from exc
    return payload


def _root_for(
    names: tuple[str, ...],
    snapshots: dict[str, _FileSnapshot],
    *,
    authenticate_driver: bool,
) -> SourceRoot:
    lines: list[str] = []
    for name in names:
        payload = snapshots[name].raw
        if authenticate_driver and name == DRIVER:
            payload = _authenticated_driver_bytes(payload)
        lines.append(f"{name}:{_sha256_bytes(payload)}")
    canonical = "\n".join(lines).encode("utf-8")
    return SourceRoot(
        sha256=_sha256_bytes(canonical),
        file_count=len(names),
        files=names,
        digest_lines="\n".join(lines),
    )


def _canonical_repo_root(repo: str | Path) -> Path:
    """Validate the caller-supplied root before resolving any alias away."""
    candidate = Path(repo)
    info = _checked_tree_entry(candidate, candidate, "repository root")
    if not stat.S_ISDIR(info.st_mode):
        raise SourceProvenanceError("repository root is not a directory")
    return candidate.resolve(strict=True)


def repository_source_snapshot(
    repo: str | Path = REPO,
) -> RepositorySourceSnapshot:
    """Capture every byte needed by generator and Python current policy once."""
    root = _canonical_repo_root(repo)
    all_names, manifest_snapshot = _manifest_names(root)
    generator_names = tuple(
        name for name in all_names
        if name.startswith("scour_MATLAB/")
    )
    if not generator_names:
        raise SourceProvenanceError(
            "source manifest contains no scour_MATLAB entries"
        )
    runtime_names = tuple(
        name for name in all_names
        if name.endswith(".py")
        or name in {ENVIRONMENT_LOCK, REQUIREMENTS_LOCK}
    )
    required = {DRIVER, ENVIRONMENT_LOCK, REQUIREMENTS_LOCK}
    missing = sorted(required - set(runtime_names))
    if missing:
        raise SourceProvenanceError(
            f"runtime source manifest lacks required entries: {missing}"
        )
    captured_names = tuple(
        sorted(set(generator_names) | set(runtime_names))
    )
    snapshots = {
        name: _regular_snapshot(root, name) for name in captured_names
    }
    _validate_python_import_closure(root, set(all_names), snapshots)
    generator = _root_for(
        generator_names, snapshots, authenticate_driver=False
    )
    python_runtime = _root_for(
        runtime_names, snapshots, authenticate_driver=True
    )
    _validate_regular_package_inventory(root, set(all_names))
    _validate_matlab_source_inventory(root, set(all_names))
    _validate_python_import_closure(root, set(all_names), snapshots)
    _assert_snapshot_unchanged(root, SOURCE_MANIFEST, manifest_snapshot)
    for name in captured_names:
        _assert_snapshot_unchanged(root, name, snapshots[name])
    return RepositorySourceSnapshot(
        repo=root,
        manifest_names=all_names,
        manifest_snapshot=manifest_snapshot,
        generator=generator,
        python_runtime=python_runtime,
        snapshots=tuple((name, snapshots[name]) for name in captured_names),
    )


def _assert_repository_snapshot_unchanged(
    snapshot: RepositorySourceSnapshot,
) -> None:
    root = snapshot.repo
    names = set(snapshot.manifest_names)
    captured = dict(snapshot.snapshots)
    _assert_snapshot_unchanged(
        root, SOURCE_MANIFEST, snapshot.manifest_snapshot
    )
    _validate_regular_package_inventory(root, names)
    _validate_matlab_source_inventory(root, names)
    _validate_python_import_closure(root, names, captured)
    _assert_snapshot_unchanged(
        root, SOURCE_MANIFEST, snapshot.manifest_snapshot
    )
    for name, source in snapshot.snapshots:
        _assert_snapshot_unchanged(root, name, source)


def generator_source_root(
    repo: str | Path = REPO,
) -> SourceRoot:
    """Hash every reviewed MATLAB generator/runtime file and numerical asset."""
    return repository_source_snapshot(repo).generator


def python_runtime_source_root(
    repo: str | Path = REPO,
) -> SourceRoot:
    """Hash executable Python bytes and their pinned environment inputs."""
    return repository_source_snapshot(repo).python_runtime

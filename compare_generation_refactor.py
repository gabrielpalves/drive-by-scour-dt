"""Compare two completed MATLAB micros across a source-only refactor.

This audit tool is intentionally narrower than ``compare_generation_releases``.
The release comparator requires equal generator-source identities, which is
correct for host qualification but makes it unsuitable for checking a refactor
whose purpose is to move code between files.

Here both datasets must authenticate their own complete MAT inventories.  Every
scientific value is then compared exactly.  The only excluded values are the
explicit generator-source identity, the fingerprint derived from that identity,
and the human timestamp.  ``generation_config_json`` is parsed and compared
field by field after removing only ``generator_source_root_sha256``; the whole
JSON blob is never waived.

This is working-tree refactor evidence, not release/host qualification and not
campaign authorization.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import numpy as np

import compare_generation_releases as release_comparison
from core.generation_state_contract import (
    STATE_DATA_FIELDS,
    STATE_TOP_LEVEL_FIELDS,
    require_canonical_state_names,
    require_damage_table_shapes,
    require_exact_fields,
)
from core.source_provenance import repository_source_snapshot


_ROOT = Path(__file__).resolve().parent
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STATE_RE = re.compile(r"^\d{4}\.mat$")
_WINDOWS_REPARSE_POINT = 0x0400

# These are source/provenance fields, not scientific degrees of freedom.
_MANIFEST_SOURCE_FIELDS = frozenset(
    {
        "gen_fingerprint",
        "generator_source_root_sha256",
        "generator_source_digest_lines",
        "generator_source_file_count",
        "timestamp",
    }
)
_TOP_SOURCE_FIELDS = frozenset(
    {
        "data",
        "file_gen_fingerprint",
        "file_generator_source_root_sha256",
    }
)
_DATA_SOURCE_FIELDS = frozenset(
    {
        "gen_fingerprint",
        "generator_source_root_sha256",
        "generator_source_digest_lines",
        "generator_source_file_count",
    }
)


class RefactorComparisonError(RuntimeError):
    """A retained dataset or refactor-equivalence claim is invalid."""


@dataclass(frozen=True)
class _DirectoryIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    links: int


@dataclass(frozen=True)
class DatasetSnapshot:
    path: Path
    directory_identity: _DirectoryIdentity
    manifest: dict[str, Any]
    generation_config: dict[str, Any]
    damage_states: dict[str, Any]
    states: dict[str, dict[str, Any]]
    state_names: tuple[str, ...]
    state_digests: dict[str, str]
    dataset_content_root_sha256: str
    generator_source_root_sha256: str
    generator_source_file_count: int
    gen_fingerprint: str
    stage: str
    n_states: int
    passages_per_state: int


@dataclass(frozen=True)
class RefactorComparisonResult:
    verdict: str
    stage: str
    n_states: int
    passages_per_state: int
    before_generator_source_root_sha256: str
    after_generator_source_root_sha256: str
    before_dataset_content_root_sha256: str
    after_dataset_content_root_sha256: str
    compared_leaves: int
    compared_numeric_values: int
    compared_signal_values: int
    exact_leaves: int
    raw_byte_identical_state_files: int
    excluded_generation_config_fields: tuple[str, ...]
    source_binding_mode: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RefactorComparisonError(
            f"cannot inspect dataset path component: {path}"
        ) from exc
    is_junction = getattr(os.path, "isjunction", None)
    return (
        path.is_symlink()
        or bool(is_junction is not None and is_junction(path))
        or bool(
            int(getattr(info, "st_file_attributes", 0))
            & _WINDOWS_REPARSE_POINT
        )
    )


def _directory_identity(path: Path, owner: str) -> _DirectoryIdentity:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RefactorComparisonError(
            f"{owner}: dataset directory cannot be inspected"
        ) from exc
    if not stat.S_ISDIR(info.st_mode):
        raise RefactorComparisonError(
            f"{owner}: dataset endpoint is not one real directory"
        )
    return _DirectoryIdentity(
        device=int(getattr(info, "st_dev", 0)),
        inode=int(getattr(info, "st_ino", 0)),
        size=int(info.st_size),
        mtime_ns=int(info.st_mtime_ns),
        links=int(getattr(info, "st_nlink", 1)),
    )


def _canonical_dataset_directory(
    raw: Path | str,
    owner: str,
) -> tuple[Path, _DirectoryIdentity]:
    supplied = Path(raw)
    if not supplied.is_absolute():
        raise RefactorComparisonError(
            f"{owner}: dataset endpoint path must be absolute"
        )
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RefactorComparisonError(
            f"{owner}: dataset endpoint is unavailable: {supplied}"
        ) from exc
    if str(supplied) != str(resolved):
        raise RefactorComparisonError(
            f"{owner}: dataset endpoint must use exact canonical spelling "
            f"({supplied} != {resolved})"
        )
    current = supplied
    while True:
        if _is_reparse(current):
            raise RefactorComparisonError(
                f"{owner}: dataset endpoint traverses a symlink/junction/"
                f"reparse point: {current}"
            )
        if current.parent == current:
            break
        current = current.parent
    return resolved, _directory_identity(resolved, owner)


def _assert_directory_unchanged(snapshot: DatasetSnapshot) -> None:
    current, identity = _canonical_dataset_directory(
        snapshot.path, str(snapshot.path)
    )
    if current != snapshot.path or identity != snapshot.directory_identity:
        raise RefactorComparisonError(
            f"{snapshot.path}: dataset directory identity changed during "
            "comparison"
        )


def _scalar(value: Any, owner: str) -> Any:
    array = np.asarray(value, dtype=object)
    if array.size != 1:
        raise RefactorComparisonError(
            f"{owner} must be scalar, got shape {array.shape}"
        )
    return array.reshape(-1)[0]


def _text(mapping: dict[str, Any], key: str, owner: str) -> str:
    if key not in mapping:
        raise RefactorComparisonError(f"{owner} lacks {key!r}")
    value = _scalar(mapping[key], f"{owner}.{key}")
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RefactorComparisonError(
                f"{owner}.{key} is not UTF-8 text"
            ) from exc
    text = str(value)
    if not text or text != text.strip() or "\r" in text:
        raise RefactorComparisonError(
            f"{owner}.{key} is not canonical nonempty text"
        )
    return text


def _sha(mapping: dict[str, Any], key: str, owner: str) -> str:
    value = _text(mapping, key, owner)
    if not _SHA256_RE.fullmatch(value):
        raise RefactorComparisonError(
            f"{owner}.{key} is not one lowercase SHA-256"
        )
    return value


def _positive_int(mapping: dict[str, Any], key: str, owner: str) -> int:
    if key not in mapping:
        raise RefactorComparisonError(f"{owner} lacks {key!r}")
    value = _scalar(mapping[key], f"{owner}.{key}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RefactorComparisonError(
            f"{owner}.{key} is not numeric"
        ) from exc
    if not np.isfinite(number) or number <= 0 or number != int(number):
        raise RefactorComparisonError(
            f"{owner}.{key} must be a positive integer"
        )
    return int(number)


def _strict_json_object(text: str, owner: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite constant {token}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RefactorComparisonError(
            f"{owner} is not one strict JSON object: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RefactorComparisonError(f"{owner} must be a JSON object")
    return value


def _validate_source_descriptor(
    manifest: dict[str, Any],
    owner: str,
) -> tuple[str, int]:
    root = _sha(manifest, "generator_source_root_sha256", owner)
    lines = _text(manifest, "generator_source_digest_lines", owner)
    count = _positive_int(manifest, "generator_source_file_count", owner)

    rows = lines.split("\n")
    names: list[str] = []
    for row in rows:
        if row.count(":") != 1:
            raise RefactorComparisonError(
                f"{owner}.generator_source_digest_lines has a malformed row"
            )
        name, digest = row.split(":", 1)
        if (
            not name.startswith("scour_MATLAB/")
            or name != name.strip()
            or "\\" in name
            or not _SHA256_RE.fullmatch(digest)
        ):
            raise RefactorComparisonError(
                f"{owner}.generator_source_digest_lines has an unsafe row"
            )
        names.append(name)
    if (
        len(rows) != count
        or names != sorted(names)
        or len(names) != len(set(names))
        or len(names) != len({name.casefold() for name in names})
        or _sha256_text(lines) != root
    ):
        raise RefactorComparisonError(
            f"{owner} has an inconsistent generator-source descriptor"
        )
    return root, count


def _reassert_content(
    snapshot: DatasetSnapshot,
) -> None:
    """Re-read the digest table, every member and the completion marker."""
    _assert_directory_unchanged(snapshot)
    try:
        per_file, root = release_comparison._read_digest_table(snapshot.path)
    except Exception as exc:
        raise RefactorComparisonError(
            f"{snapshot.path}: digest table changed during comparison: {exc}"
        ) from exc
    expected_names = {
        "case_info.mat",
        "damage_states.mat",
        *snapshot.state_names,
    }
    if set(per_file) != expected_names:
        raise RefactorComparisonError(
            f"{snapshot.path}: digest inventory changed during comparison"
        )
    for name, expected in sorted(per_file.items()):
        target = snapshot.path / name
        try:
            raw = release_comparison._read_single_link_regular_bytes(
                target, f"{snapshot.path}/{name}"
            )
        except Exception as exc:
            raise RefactorComparisonError(
                f"{snapshot.path}: {name} became unsafe/unreadable: {exc}"
            ) from exc
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise RefactorComparisonError(
                f"{snapshot.path}: {name} changed during comparison"
            )
    if root != snapshot.dataset_content_root_sha256:
        raise RefactorComparisonError(
            f"{snapshot.path}: dataset content root changed during comparison"
        )

    marker = snapshot.path / "_GENERATION_COMPLETE"
    expected_marker = (
        f"{_text(snapshot.manifest, 'gen_schema', str(snapshot.path))}\n"
        f"{snapshot.gen_fingerprint}\n"
        f"{snapshot.dataset_content_root_sha256}\n"
    )
    try:
        marker_text = release_comparison._read_single_link_regular_bytes(
            marker, str(marker)
        ).decode("utf-8", errors="strict")
    except Exception as exc:
        raise RefactorComparisonError(
            f"{snapshot.path}: completion marker is unsafe/unreadable: {exc}"
        ) from exc
    if marker_text != expected_marker:
        raise RefactorComparisonError(
            f"{snapshot.path}: completion marker is absent or changed"
        )
    _assert_directory_unchanged(snapshot)


def _load_snapshot(directory: Path | str) -> DatasetSnapshot:
    path, directory_identity = _canonical_dataset_directory(
        directory, "dataset endpoint"
    )
    try:
        per_file, dataset_root = release_comparison._read_digest_table(path)
        case_mat = release_comparison._verified_mat(
            path, "case_info.mat", per_file
        )
    except Exception as exc:
        raise RefactorComparisonError(
            f"{path}: cannot authenticate the dataset header: {exc}"
        ) from exc

    manifest = case_mat.get("case_info")
    if set(case_mat) != {"case_info"} or not isinstance(manifest, dict):
        raise RefactorComparisonError(
            f"{path}: case_info.mat must contain one scalar case_info struct"
        )
    owner = f"{path}.case_info"
    n_states = _positive_int(manifest, "n_states", owner)
    n_passages = _positive_int(manifest, "passages_per_state", owner)
    n_supports = _positive_int(manifest, "num_supports", owner)
    stage = _text(manifest, "stage", owner)
    stage_policy = release_comparison._STAGE_POLICIES.get(stage)
    if stage_policy is None:
        raise RefactorComparisonError(
            f"{path}: stage {stage!r} is outside the reviewed campaign"
        )
    state_names = tuple(f"{index:04d}.mat" for index in range(1, n_states + 1))
    expected_digests = {"case_info.mat", "damage_states.mat", *state_names}
    if set(per_file) != expected_digests:
        raise RefactorComparisonError(
            f"{path}: digest table does not cover exactly the canonical "
            "manifest, damage table and numbered states"
        )
    require_canonical_state_names(
        (candidate.name for candidate in path.iterdir()),
        n_states,
        str(path),
        error_type=RefactorComparisonError,
    )

    source_root, source_count = _validate_source_descriptor(manifest, owner)
    fingerprint = _sha(manifest, "gen_fingerprint", owner)
    config_text = _text(manifest, "generation_config_json", owner)
    if _sha256_text(config_text) != fingerprint:
        raise RefactorComparisonError(
            f"{path}: generation_config_json does not reproduce gen_fingerprint"
        )
    generation_config = _strict_json_object(
        config_text, f"{owner}.generation_config_json"
    )
    if generation_config.get("generator_source_root_sha256") != source_root:
        raise RefactorComparisonError(
            f"{path}: generation config is not bound to its source root"
        )

    try:
        damage_states = release_comparison._verified_mat(
            path, "damage_states.mat", per_file
        )
        states = {
            name: release_comparison._verified_mat(path, name, per_file)
            for name in state_names
        }
    except Exception as exc:
        raise RefactorComparisonError(
            f"{path}: cannot authenticate one canonical MAT member: {exc}"
        ) from exc

    require_damage_table_shapes(
        damage_states,
        n_states=n_states,
        n_passages=n_passages,
        n_supports=n_supports,
        n_scour_supports=len(stage_policy["scour_supports"]),
        n_state_streams=len(release_comparison._STATE_STREAM_NAMES),
        n_passage_streams=len(release_comparison._PASSAGE_STREAM_NAMES),
        owner=f"{path}/damage_states.mat",
        error_type=RefactorComparisonError,
    )
    if np.asarray(damage_states["StateSeedID"]).dtype != np.dtype("uint32"):
        raise RefactorComparisonError(
            f"{path}/damage_states.mat: StateSeedID must retain MATLAB "
            "uint32 storage"
        )

    for name, state in states.items():
        try:
            release_comparison._require_uint32_state_seed_storage(
                state, f"{path}/{name}"
            )
        except release_comparison.QualificationInputError as exc:
            raise RefactorComparisonError(str(exc)) from exc
        require_exact_fields(
            state,
            STATE_TOP_LEVEL_FIELDS,
            f"{path}/{name}",
            error_type=RefactorComparisonError,
        )
        data = state.get("data")
        if not isinstance(data, dict):
            raise RefactorComparisonError(f"{path}/{name}: missing data struct")
        require_exact_fields(
            data,
            STATE_DATA_FIELDS,
            f"{path}/{name}.data",
            error_type=RefactorComparisonError,
        )
        if _sha(state, "file_gen_fingerprint", name) != fingerprint:
            raise RefactorComparisonError(
                f"{path}/{name}: top-level fingerprint differs from case_info"
            )
        if _sha(data, "gen_fingerprint", f"{name}.data") != fingerprint:
            raise RefactorComparisonError(
                f"{path}/{name}: nested fingerprint differs from case_info"
            )
        if (
            _sha(state, "file_generator_source_root_sha256", name)
            != source_root
            or _sha(
                data,
                "generator_source_root_sha256",
                f"{name}.data",
            )
            != source_root
        ):
            raise RefactorComparisonError(
                f"{path}/{name}: state is not bound to the manifest source root"
            )

    snapshot = DatasetSnapshot(
        path=path,
        directory_identity=directory_identity,
        manifest=manifest,
        generation_config=generation_config,
        damage_states=damage_states,
        states=states,
        state_names=state_names,
        state_digests={name: per_file[name] for name in state_names},
        dataset_content_root_sha256=dataset_root,
        generator_source_root_sha256=source_root,
        generator_source_file_count=source_count,
        gen_fingerprint=fingerprint,
        stage=stage,
        n_states=n_states,
        passages_per_state=n_passages,
    )
    _reassert_content(snapshot)
    return snapshot


def compare_refactor_outputs(
    before_directory: Path | str,
    after_directory: Path | str,
    *,
    expected_before_source_root_sha256: str,
    expected_after_source_root_sha256: str | None = None,
    historical_retained: bool = False,
) -> RefactorComparisonResult:
    """Return an exact scientific comparison or raise on any mismatch.

    The default mode binds the after endpoint to the live reviewed tree.
    ``historical_retained=True`` is an explicit evidence-replay mode: both
    source roots must be supplied, but neither is asserted to be the live tree.
    """
    if not _SHA256_RE.fullmatch(expected_before_source_root_sha256):
        raise RefactorComparisonError(
            "expected before-source root must be one lowercase SHA-256"
        )
    if historical_retained and expected_after_source_root_sha256 is None:
        raise RefactorComparisonError(
            "historical-retained mode requires an explicit after-source root"
        )
    try:
        current_sources = (
            None
            if historical_retained
            else repository_source_snapshot(_ROOT)
        )
    except Exception as exc:
        raise RefactorComparisonError(
            "cannot authenticate one stable current reviewed source tree"
        ) from exc
    current_root = (
        None if current_sources is None else current_sources.generator.sha256
    )
    expected_after = expected_after_source_root_sha256 or current_root
    if expected_after is None:  # narrowed above, kept explicit for type safety
        raise RefactorComparisonError("after-source root is unavailable")
    if not _SHA256_RE.fullmatch(expected_after):
        raise RefactorComparisonError(
            "expected after-source root must be one lowercase SHA-256"
        )
    if not historical_retained and expected_after != current_root:
        raise RefactorComparisonError(
            "the expected after-source root is not the current reviewed "
            f"generator root ({expected_after} != {current_root})"
        )

    before = _load_snapshot(before_directory)
    after = _load_snapshot(after_directory)
    if (
        before.generator_source_root_sha256
        != expected_before_source_root_sha256
    ):
        raise RefactorComparisonError(
            "the before dataset has the wrong registered source root"
        )
    if after.generator_source_root_sha256 != expected_after:
        raise RefactorComparisonError(
            "the after dataset does not declare/authenticate the expected "
            "current source root"
        )
    if before.generator_source_root_sha256 == after.generator_source_root_sha256:
        raise RefactorComparisonError(
            "source roots are equal; this is not a cross-refactor comparison"
        )
    for attribute in (
        "stage",
        "n_states",
        "passages_per_state",
        "state_names",
    ):
        if getattr(before, attribute) != getattr(after, attribute):
            raise RefactorComparisonError(
                f"datasets differ in {attribute}: "
                f"{getattr(before, attribute)!r} != "
                f"{getattr(after, attribute)!r}"
            )

    stats = release_comparison.ComparisonStats()
    manifest_before = {
        key: value
        for key, value in before.manifest.items()
        if key not in _MANIFEST_SOURCE_FIELDS
        and key != "generation_config_json"
    }
    manifest_after = {
        key: value
        for key, value in after.manifest.items()
        if key not in _MANIFEST_SOURCE_FIELDS
        and key != "generation_config_json"
    }
    release_comparison._compare_node(
        manifest_before,
        manifest_after,
        path="case_info",
        rtol=0.0,
        atol=0.0,
        stats=stats,
        force_exact=True,
    )

    config_before = dict(before.generation_config)
    config_after = dict(after.generation_config)
    config_before.pop("generator_source_root_sha256", None)
    config_after.pop("generator_source_root_sha256", None)
    release_comparison._compare_node(
        config_before,
        config_after,
        path="generation_config_json",
        rtol=0.0,
        atol=0.0,
        stats=stats,
        force_exact=True,
    )
    release_comparison._compare_node(
        before.damage_states,
        after.damage_states,
        path="damage_states",
        rtol=0.0,
        atol=0.0,
        stats=stats,
        force_exact=True,
    )

    raw_identical = 0
    for name in before.state_names:
        if before.state_digests[name] == after.state_digests[name]:
            raw_identical += 1
        state_before = before.states[name]
        state_after = after.states[name]
        top_before = {
            key: value
            for key, value in state_before.items()
            if key not in _TOP_SOURCE_FIELDS
        }
        top_after = {
            key: value
            for key, value in state_after.items()
            if key not in _TOP_SOURCE_FIELDS
        }
        release_comparison._compare_node(
            top_before,
            top_after,
            path=f"{name}.top",
            rtol=0.0,
            atol=0.0,
            stats=stats,
            force_exact=True,
        )
        data_before = {
            key: value
            for key, value in state_before["data"].items()
            if key not in _DATA_SOURCE_FIELDS
        }
        data_after = {
            key: value
            for key, value in state_after["data"].items()
            if key not in _DATA_SOURCE_FIELDS
        }
        release_comparison._compare_node(
            data_before,
            data_after,
            path=f"{name}.data",
            rtol=0.0,
            atol=0.0,
            stats=stats,
            force_exact=True,
        )

    if stats.compared_leaves == 0 or stats.compared_signal_values == 0:
        raise RefactorComparisonError(
            "comparison reached zero leaves or zero vehicle-signal values"
        )
    _reassert_content(before)
    _reassert_content(after)
    if stats.mismatches:
        detail = "\n".join(stats.mismatches)
        raise RefactorComparisonError(
            "scientific payload changed across the refactor:\n" + detail
        )
    if stats.numerical_difference:
        raise RefactorComparisonError(
            "unexpected numerical-difference flag in an exact comparison"
        )
    if not historical_retained:
        if current_sources is None:  # narrowed above; explicit for reviewers
            raise RefactorComparisonError(
                "current reviewed generator snapshot is unavailable"
            )
        try:
            current_sources.assert_unchanged()
        except Exception as exc:
            raise RefactorComparisonError(
                "current reviewed generator bytes/identities changed during "
                "the default comparison; rerun against one stable source tree"
            ) from exc
        if current_sources.generator.sha256 != expected_after:
            raise RefactorComparisonError(
                "retained current generator snapshot no longer binds the "
                "expected after-source root"
            )

    return RefactorComparisonResult(
        verdict="EXACT-SCIENTIFIC-PAYLOAD-EQUIVALENCE",
        stage=before.stage,
        n_states=before.n_states,
        passages_per_state=before.passages_per_state,
        before_generator_source_root_sha256=(
            before.generator_source_root_sha256
        ),
        after_generator_source_root_sha256=(
            after.generator_source_root_sha256
        ),
        before_dataset_content_root_sha256=(
            before.dataset_content_root_sha256
        ),
        after_dataset_content_root_sha256=(
            after.dataset_content_root_sha256
        ),
        compared_leaves=stats.compared_leaves,
        compared_numeric_values=stats.compared_numeric_values,
        compared_signal_values=stats.compared_signal_values,
        exact_leaves=stats.exact_leaves,
        raw_byte_identical_state_files=raw_identical,
        excluded_generation_config_fields=(
            "generator_source_root_sha256",
        ),
        source_binding_mode=(
            "historical-retained"
            if historical_retained
            else "current-reviewed-tree"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument(
        "--expected-before-source-root",
        required=True,
        help="registered generator source root embedded in the before dataset",
    )
    parser.add_argument(
        "--expected-after-source-root",
        help=(
            "optional current generator source root; if omitted it is derived "
            "from the reviewed working tree"
        ),
    )
    parser.add_argument(
        "--historical-retained",
        action="store_true",
        help=(
            "replay two retained historical endpoints using both explicitly "
            "supplied source roots; default mode remains bound to the live tree"
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = compare_refactor_outputs(
            args.before,
            args.after,
            expected_before_source_root_sha256=(
                args.expected_before_source_root
            ),
            expected_after_source_root_sha256=(
                args.expected_after_source_root
            ),
            historical_retained=args.historical_retained,
        )
    except RefactorComparisonError as exc:
        parser.exit(1, f"REFACTOR EQUIVALENCE: FAIL: {exc}\n")
    print(json.dumps(asdict(result), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

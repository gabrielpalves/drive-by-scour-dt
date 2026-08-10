"""Behavioral tests for the cross-refactor MATLAB payload comparator.

The fixtures are genuine MATLAB-v5 files.  A positive pair differs coherently
in generator-source identity and the derived generation fingerprint, while its
scientific payload is unchanged.  Two adversarial pairs then prove that a state
value and a setting hidden inside ``generation_config_json`` are both rejected
even after all affected digests/fingerprints are rebuilt.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

import check_generation_release_comparison as fixture_builder
import compare_generation_refactor as refactor_comparison
from core.source_provenance import generator_source_root


_CURRENT_STAGES = ("F40-S", "F40-M", "L99-S", "L99-M")
_CURRENT_CHANNEL_SCHEMA_ID = "physical8_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_mat(path: Path) -> dict:
    loaded = loadmat(path, simplify_cells=True, mat_dtype=True)
    return {
        str(key): value
        for key, value in loaded.items()
        if not str(key).startswith("__")
    }


def _source_variant(replacement_digest: str = "0") -> tuple[str, str, int]:
    """Return one coherent synthetic pre-refactor source descriptor."""
    if replacement_digest not in {"0", "1"}:
        raise ValueError("fixture digest selector must be '0' or '1'")
    current = generator_source_root(Path(__file__).resolve().parent)
    rows = current.digest_lines.split("\n")
    replacement_index = next(
        index
        for index, row in enumerate(rows)
        if row.startswith("scour_MATLAB/A00_Run.m:")
    )
    name, _ = rows[replacement_index].split(":", 1)
    rows[replacement_index] = f"{name}:{replacement_digest * 64}"
    lines = "\n".join(rows)
    return (
        hashlib.sha256(lines.encode("utf-8")).hexdigest(),
        lines,
        len(rows),
    )


def _write_digest_table(path: Path, fingerprint: str, schema: str) -> None:
    state_names = sorted(
        candidate.name
        for candidate in path.iterdir()
        if candidate.is_file()
        and refactor_comparison._STATE_RE.fullmatch(candidate.name)
    )
    names = [*state_names, "case_info.mat", "damage_states.mat"]
    per_file = {name: _sha256(path / name) for name in names}
    lines = "\n".join(
        f"{name}:{per_file[name]}" for name in sorted(per_file)
    )
    root = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    savemat(
        path / "file_digests.mat",
        {
            "file_digests": {
                "schema": "source-digests-v2",
                "scope": "NNNN.mat+case_info.mat+damage_states.mat",
                "digest_lines": lines,
                "root": root,
            }
        },
        long_field_names=True,
    )
    (path / "_GENERATION_COMPLETE").write_text(
        f"{schema}\n{fingerprint}\n{root}\n",
        encoding="utf-8",
        newline="\n",
    )


def _rewrite_identity(
    path: Path,
    *,
    source_root: str,
    source_lines: str,
    source_count: int,
    mutate_config: bool = False,
) -> None:
    case_info = deepcopy(_public_mat(path / "case_info.mat")["case_info"])
    config = json.loads(case_info["generation_config_json"])
    config["generator_source_root_sha256"] = source_root
    if mutate_config:
        config["pad_p_fail"] = float(config["pad_p_fail"]) + 0.001
    config_json = json.dumps(
        config,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    fingerprint = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    case_info.update(
        {
            "generation_config_json": config_json,
            "gen_fingerprint": fingerprint,
            "generator_source_root_sha256": source_root,
            "generator_source_digest_lines": source_lines,
            "generator_source_file_count": source_count,
        }
    )
    savemat(
        path / "case_info.mat",
        {"case_info": case_info},
        long_field_names=True,
    )

    for state_path in sorted(path.glob("[0-9][0-9][0-9][0-9].mat")):
        state = deepcopy(_public_mat(state_path))
        data = state["data"]
        state["file_gen_fingerprint"] = fingerprint
        state["file_generator_source_root_sha256"] = source_root
        data["gen_fingerprint"] = fingerprint
        data["generator_source_root_sha256"] = source_root
        data["generator_source_digest_lines"] = source_lines
        data["generator_source_file_count"] = source_count
        # ``simplify_cells=True`` turns MATLAB 1x1 numeric values into Python
        # scalars.  Reassert the production storage type before reserialising;
        # otherwise this identity-only fixture silently changes uint32 seeds.
        state["file_state_seed_id"] = np.uint32(
            np.asarray(state["file_state_seed_id"]).reshape(-1)[0]
        )
        data["state_seed_id"] = np.uint32(
            np.asarray(data["state_seed_id"]).reshape(-1)[0]
        )
        savemat(state_path, state, long_field_names=True)
    _write_digest_table(path, fingerprint, str(case_info["gen_schema"]))


def _mutate_scientific_value(path: Path) -> None:
    state_path = path / "0001.mat"
    state = deepcopy(_public_mat(state_path))
    velocity = np.asarray(state["data"]["Velocidade"], dtype=float).copy()
    velocity.reshape(-1)[0] += 0.25
    state["data"]["Velocidade"] = velocity
    savemat(state_path, state, long_field_names=True)
    case_info = _public_mat(path / "case_info.mat")["case_info"]
    _write_digest_table(
        path,
        str(case_info["gen_fingerprint"]),
        str(case_info["gen_schema"]),
    )


def _expect_rejected(
    label: str,
    action,
    *,
    diagnostic: str | None = None,
) -> None:
    try:
        action()
    except refactor_comparison.RefactorComparisonError as exc:
        if diagnostic is not None and diagnostic not in str(exc):
            raise AssertionError(
                f"{label}: rejected by the wrong guard; expected "
                f"{diagnostic!r} in {str(exc)!r}"
            ) from exc
        print(f"  [PASS] {label}")
    else:
        raise AssertionError(f"mutation escaped: {label}")


def main() -> None:
    current = generator_source_root(Path(__file__).resolve().parent)
    before_root, before_lines, before_count = _source_variant()
    release_contract = refactor_comparison.release_comparison
    if tuple(release_contract.STAGE_ORDER) != _CURRENT_STAGES:
        raise AssertionError("refactor fixture stage inventory drifted")
    if (
        release_contract.EXPECTED_CHANNEL_SCHEMA_ID
        != _CURRENT_CHANNEL_SCHEMA_ID
    ):
        raise AssertionError("refactor fixture channel schema drifted")
    with tempfile.TemporaryDirectory(
        prefix="generation-refactor-check-"
    ) as temporary:
        root = Path(temporary).resolve(strict=True)
        stage_pairs: dict[str, tuple[Path, Path]] = {}
        for stage in _CURRENT_STAGES:
            fixture_tag = stage.lower().replace("-", "_")
            stage_before = root / f"{fixture_tag}_before"
            stage_after = root / f"{fixture_tag}_after"
            fixture_builder._write_dataset(
                stage_before,
                release="R2025b",
                n_states=2,
                n_passages=2,
                stage=stage,
            )
            fixture_builder._write_dataset(
                stage_after,
                release="R2025b",
                n_states=2,
                n_passages=2,
                stage=stage,
            )
            for dataset in (stage_before, stage_after):
                manifest = _public_mat(dataset / "case_info.mat")["case_info"]
                if str(manifest["stage"]) != stage:
                    raise AssertionError(
                        f"{stage}: fixture manifest stage drifted"
                    )
                if (
                    str(manifest["channel_schema_id"])
                    != _CURRENT_CHANNEL_SCHEMA_ID
                ):
                    raise AssertionError(
                        f"{stage}: fixture channel schema drifted"
                    )
            _rewrite_identity(
                stage_before,
                source_root=before_root,
                source_lines=before_lines,
                source_count=before_count,
            )

            result = refactor_comparison.compare_refactor_outputs(
                stage_before,
                stage_after,
                expected_before_source_root_sha256=before_root,
                expected_after_source_root_sha256=current.sha256,
            )
            if (
                result.verdict
                != "EXACT-SCIENTIFIC-PAYLOAD-EQUIVALENCE"
                or result.compared_signal_values <= 0
                or result.source_binding_mode != "current-reviewed-tree"
            ):
                raise AssertionError(
                    f"{stage}: positive cross-refactor control did not pass"
                )
            print(
                "  [PASS] source-only identity change: "
                f"{stage}/{_CURRENT_CHANNEL_SCHEMA_ID}"
            )
            stage_pairs[stage] = (stage_before, stage_after)

        # Retain one current block for the adversarial endpoint, source-drift,
        # scientific-payload, and hidden-configuration checks below.
        before, after = stage_pairs["F40-S"]

        relative_before = Path(before.name)
        _expect_rejected(
            "relative dataset endpoint",
            lambda: refactor_comparison.compare_refactor_outputs(
                relative_before,
                after,
                expected_before_source_root_sha256=before_root,
                expected_after_source_root_sha256=current.sha256,
            ),
            diagnostic="absolute",
        )
        lexical_alias = before / ".." / before.name
        _expect_rejected(
            "absolute but noncanonical dataset spelling",
            lambda: refactor_comparison.compare_refactor_outputs(
                lexical_alias,
                after,
                expected_before_source_root_sha256=before_root,
                expected_after_source_root_sha256=current.sha256,
            ),
            diagnostic="canonical",
        )

        alias = root / "before-alias"
        if os.name == "nt":
            alias_created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(alias), str(before)],
                check=False,
                capture_output=True,
                text=True,
            ).returncode == 0
        else:
            alias.symlink_to(before, target_is_directory=True)
            alias_created = True
        if alias_created:
            _expect_rejected(
                "dataset junction/symlink alias",
                lambda: refactor_comparison.compare_refactor_outputs(
                    alias,
                    after,
                    expected_before_source_root_sha256=before_root,
                    expected_after_source_root_sha256=current.sha256,
                ),
                diagnostic="canonical",
            )
        else:
            print("  [N/A] junction fixture unavailable on this host")

        original_compare_node = refactor_comparison.release_comparison._compare_node
        identity_probe = {"fired": False}

        def mutate_directory_identity(*args, **kwargs):
            if not identity_probe["fired"]:
                identity_probe["fired"] = True
                (before / "late-directory-entry.tmp").write_bytes(b"late\n")
            return original_compare_node(*args, **kwargs)

        refactor_comparison.release_comparison._compare_node = (
            mutate_directory_identity
        )
        try:
            _expect_rejected(
                "dataset directory identity drift during comparison",
                lambda: refactor_comparison.compare_refactor_outputs(
                    before,
                    after,
                    expected_before_source_root_sha256=before_root,
                    expected_after_source_root_sha256=current.sha256,
                ),
                diagnostic="identity changed",
            )
        finally:
            refactor_comparison.release_comparison._compare_node = (
                original_compare_node
            )
            (before / "late-directory-entry.tmp").unlink(missing_ok=True)
        if not identity_probe["fired"]:
            raise AssertionError("directory-identity drift probe never fired")

        original_source_snapshot = (
            refactor_comparison.repository_source_snapshot
        )
        stable_snapshot = original_source_snapshot(
            Path(__file__).resolve().parent
        )
        source_reassertions = {"count": 0}

        class DriftingSourceSnapshot:
            generator = stable_snapshot.generator

            @staticmethod
            def assert_unchanged() -> None:
                source_reassertions["count"] += 1
                raise RuntimeError("synthetic retained source-identity drift")

        def drifting_source_snapshot(_repo: Path):
            return DriftingSourceSnapshot()

        refactor_comparison.repository_source_snapshot = (
            drifting_source_snapshot
        )
        try:
            _expect_rejected(
                "default mode reasserts retained live source identities",
                lambda: refactor_comparison.compare_refactor_outputs(
                    before,
                    after,
                    expected_before_source_root_sha256=before_root,
                    expected_after_source_root_sha256=current.sha256,
                ),
                diagnostic="bytes/identities changed",
            )
        finally:
            refactor_comparison.repository_source_snapshot = (
                original_source_snapshot
            )
        if source_reassertions["count"] != 1:
            raise AssertionError(
                "live-source drift probe never reached final reassertion"
            )

        scientific = root / "scientific_mutation"
        shutil.copytree(after, scientific)
        _mutate_scientific_value(scientific)
        _expect_rejected(
            "coherently re-digested scientific-value mutation",
            lambda: refactor_comparison.compare_refactor_outputs(
                before,
                scientific,
                expected_before_source_root_sha256=before_root,
                expected_after_source_root_sha256=current.sha256,
            ),
        )

        hidden_config = root / "hidden_config_mutation"
        shutil.copytree(before, hidden_config)
        _rewrite_identity(
            hidden_config,
            source_root=before_root,
            source_lines=before_lines,
            source_count=before_count,
            mutate_config=True,
        )
        _expect_rejected(
            "coherently re-fingerprinted hidden generation-config mutation",
            lambda: refactor_comparison.compare_refactor_outputs(
                hidden_config,
                after,
                expected_before_source_root_sha256=before_root,
                expected_after_source_root_sha256=current.sha256,
            ),
        )

        historical_after = root / "historical_after"
        shutil.copytree(after, historical_after)
        historical_root, historical_lines, historical_count = (
            _source_variant("1")
        )
        _rewrite_identity(
            historical_after,
            source_root=historical_root,
            source_lines=historical_lines,
            source_count=historical_count,
        )
        _expect_rejected(
            "historical after-root cannot bypass default current-tree binding",
            lambda: refactor_comparison.compare_refactor_outputs(
                before,
                historical_after,
                expected_before_source_root_sha256=before_root,
                expected_after_source_root_sha256=historical_root,
            ),
        )
        historical_result = (
            refactor_comparison.compare_refactor_outputs(
                before,
                historical_after,
                expected_before_source_root_sha256=before_root,
                expected_after_source_root_sha256=historical_root,
                historical_retained=True,
            )
        )
        if (
            historical_result.source_binding_mode != "historical-retained"
            or historical_result.compared_signal_values <= 0
        ):
            raise AssertionError(
                "explicit historical-retained comparison did not pass"
            )
        print("  [PASS] explicit historical-retained source binding")

    print("GENERATION REFACTOR EQUIVALENCE CHECKS: ALL PASS")


if __name__ == "__main__":
    main()

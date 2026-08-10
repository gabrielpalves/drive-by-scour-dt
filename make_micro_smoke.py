"""Build MICRO integration and MATLAB release-qualification environments.

The generated MATLAB scripts are guarded text patches of the real
``scour_MATLAB/A00_Run.m``.  They therefore exercise the reviewed generator
rather than reimplementing it.

Supported release qualification
-------------------------------
The only supported invocation is
``python make_micro_smoke.py --qualification --stage STAGE``.  The former
no-argument ``micro_A00_smoke.m`` and ``--dryrun`` paths are retired: their
reduced counts are intentionally invalid under the production stage-count
contract, so every public and CLI entry point rejects them before writing.

Release qualification:

* selects one of the four Paper-1 blocks;
* sets ``qualification_run=true`` explicitly, independent of inherited shell
  environment;
* segregates output below
  ``Results/release_qualification/<source-hash>/<environment-hash>/R<release>/``.

Generate all four blocks on each intended release/host.  The release comparator
authenticates each segregated qualification directory independently.
"""
from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{_unsafe_python_path_variable} must be absent before evidence "
            "imports"
        )
_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.dirname(__file__)
)
_bootstrap_first_path = _bootstrap_sys.path[0] or _bootstrap_os.getcwd()
if (
    _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    or _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_source_root
    ))
):
    raise RuntimeError(
        "reviewed repository root must be the canonical first import path"
    )
_bootstrap_guard_dir = _bootstrap_os.path.join(
    _bootstrap_source_root, "campaign_import_guard"
)
_bootstrap_guard_init = _bootstrap_os.path.join(
    _bootstrap_guard_dir, "__init__.py"
)
if (
    not _bootstrap_os.path.isfile(_bootstrap_guard_init)
    or _bootstrap_os.path.islink(_bootstrap_guard_init)
    or _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_guard_dir
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_dir
    ))
    or any(
        entry.casefold().startswith("__init__.")
        and entry != "__init__.py"
        for entry in _bootstrap_os.listdir(_bootstrap_guard_dir)
    )
):
    raise RuntimeError(
        "reviewed campaign import guard package is absent or ambiguous"
    )
_bootstrap_loaded_guard = _bootstrap_sys.modules.get("campaign_import_guard")
if _bootstrap_loaded_guard is not None and (
    _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        getattr(_bootstrap_loaded_guard, "__file__", "")
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_init
    ))
    or getattr(_bootstrap_loaded_guard, "_BOUNDARY_ENFORCED", False) is not True
):
    raise RuntimeError(
        "preloaded campaign import guard is not the reviewed enforced module"
    )
from campaign_import_guard import (  # noqa: E402
    enforce_import_boundary as _enforce_import_boundary,
)
_enforce_import_boundary()

import argparse
import hashlib
import os
import re
import shutil
import sys


REPO = os.path.dirname(os.path.abspath(__file__))
MICRO_DS = "F40-S_L40_st31"  # marked qualification families: 3+4+8+6+10
QUALIFICATION_STAGES = (
    "F40-S",
    "F40-M",
    "L99-S",
    "L99-M",
)
_QUALIFICATION_SHA_PLACEHOLDER = "<QUALIFICATION_SOURCE_SHA256>"
_QUALIFICATION_FOLDER_PLACEHOLDER = "<QUALIFICATION_SOURCE_FOLDER>"


def _patch_once(source: str, pattern: str, replacement: str, label: str) -> str:
    patched, count = re.subn(pattern, replacement, source, count=1, flags=re.M)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one A00 match, found {count}; "
            "A00 changed, so no micro script was emitted"
        )
    return patched


def _render_micro_template(
    source: str,
    *,
    qualification: bool = False,
    stage: str = "F40-S",
) -> str:
    """Return a guarded micro script, with identity placeholders if qualified."""
    if not qualification:
        raise ValueError(
            "unmarked micro generation is retired; use --qualification "
            "--stage with one of F40-S, F40-M, L99-S, or L99-M"
        )
    if stage not in QUALIFICATION_STAGES:
        raise ValueError(
            f"unknown stage {stage!r}; choose one of {QUALIFICATION_STAGES}"
        )

    patches = (
        (
            r"^n_states_multi   = 0;.*$",
            "n_states_multi   = 10;     % MICRO-SMOKE",
            "joint-state count",
        ),
        (
            r"^Npass = 50;.*$",
            "Npass = 3;                  % MICRO-SMOKE",
            "passage count",
        ),
        (
            r"^n_healthy_states  = 5;.*$",
            "n_healthy_states  = 3;     % MICRO-SMOKE",
            "healthy-state count",
        ),
        (
            r"^n_anchor_levels  = 60;.*$",
            "n_anchor_levels  = 2;      % MICRO-SMOKE",
            "anchor-level count",
        ),
        (
            r"^n_anchor_reps     = 5;.*$",
            "n_anchor_reps     = 2;     % MICRO-SMOKE",
            "anchor-replica count",
        ),
        (
            r"^n_nuisance_states = 0;.*$",
            "n_nuisance_states = 6;     % MICRO-SMOKE",
            "nuisance-state count",
        ),
    )
    for pattern, replacement, label in patches:
        source = _patch_once(source, pattern, replacement, label)

    source = _patch_once(
        source,
        r"^STAGE\s*=\s*'[^']+';",
        f"STAGE = '{stage}';               % RELEASE-QUALIFICATION",
        "stage selector",
    )

    # Support both the former environment-derived declaration and the hardened
    # R11 declaration. The emitted script itself is explicit, so a forgotten
    # shell variable cannot silently change its role.
    qualification_patterns = (
        r"^qualification_run\s*=\s*~isempty\(getenv\("
        r"'A00_RELEASE_QUALIFICATION'\)\);",
        r"^qualification_run\s*=\s*false;",
    )
    qualification_matches = 0
    for pattern in qualification_patterns:
        candidate, count = re.subn(
            pattern,
            "qualification_run = true;  % RELEASE-QUALIFICATION ONLY",
            source,
            count=1,
            flags=re.M,
        )
        if count:
            source = candidate
            qualification_matches += count
            break
    if qualification_matches != 1:
        raise RuntimeError(
            "qualification declaration was not found exactly once; refusing "
            "to emit a script whose qualification marker is uncertain"
        )

    # Qualification identity is defined over this canonical template, before its
    # own digest and derived folder token are substituted.  This avoids a
    # self-referential hash while binding every other byte of the emitted script,
    # including the selected stage and all guarded micro-size patches.
    source = _patch_once(
        source,
        r"^qualification_source_sha256\s*=\s*'PRODUCTION';",
        "qualification_source_sha256 = "
        f"'{_QUALIFICATION_SHA_PLACEHOLDER}';"
        "  % GENERATED QUALIFICATION SOURCE ID",
        "qualification-source identity",
    )
    qualification_folder = (
        "run_folder = fullfile('Results', 'release_qualification', "
        f"'{_QUALIFICATION_FOLDER_PLACEHOLDER}', "
        "['env-' actual_matlab_environment_sha256(1:16)], "
        "['R' matlab_release], case_name);"
    )
    source = _patch_once(
        source,
        r"^run_folder\s*=\s*fullfile\(results_root,\s*relative_run_folder\);",
        qualification_folder,
        "run-folder declaration",
    )
    return source


def _qualification_template(
    stage: str,
    source: str | None = None,
) -> str:
    """Canonical preimage used for a qualification script's source identity."""
    if source is None:
        source_path = os.path.join(REPO, "scour_MATLAB", "A00_Run.m")
        with open(source_path, encoding="utf-8") as handle:
            source = handle.read()
    return _render_micro_template(
        source,
        qualification=True,
        stage=stage,
    )


def qualification_source_sha256(
    stage: str,
    source: str | None = None,
) -> str:
    """Full SHA-256 stamped into a stage's deterministic micro script.

    Passing ``source`` lets evidence code derive the stamp from the exact
    authenticated A00 byte snapshot it already owns, without reopening the
    pathname and accidentally combining two repository states.
    """
    template = _qualification_template(stage, source)
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def canonicalize_qualification_source_bytes(
    emitted: bytes,
    stamped_sha256: str,
) -> bytes:
    """Recover the exact self-reference-free qualification preimage.

    The generated MATLAB file differs from its hashed template in exactly two
    derived ASCII tokens: the full embedded SHA-256 and its 16-character output
    folder prefix.  Replacing anything less specific would permit a mutation to
    hide behind broad text normalisation; replacing anything more would make the
    source identity self-referential.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", stamped_sha256):
        raise ValueError("stamped_sha256 must be one lowercase SHA-256")
    full = stamped_sha256.encode("ascii")
    folder = stamped_sha256[:16].encode("ascii")
    if emitted.count(full) != 1:
        raise ValueError(
            "qualification source must contain its full stamped SHA exactly once"
        )
    canonical = emitted.replace(
        full,
        _QUALIFICATION_SHA_PLACEHOLDER.encode("ascii"),
        1,
    )
    if canonical.count(folder) != 1:
        raise ValueError(
            "qualification source must contain its derived folder token exactly once"
        )
    return canonical.replace(
        folder,
        _QUALIFICATION_FOLDER_PLACEHOLDER.encode("ascii"),
        1,
    )


def verify_qualification_source_bytes(
    emitted: bytes,
    stamped_sha256: str,
) -> str:
    """Fail unless emitted executable bytes authenticate their embedded stamp."""
    canonical = canonicalize_qualification_source_bytes(emitted, stamped_sha256)
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != stamped_sha256:
        raise ValueError(
            "qualification executable bytes do not match the embedded "
            f"qualification_source_sha256 ({actual} != {stamped_sha256})"
        )
    return hashlib.sha256(emitted).hexdigest()


def render_micro_a00(
    source: str,
    *,
    qualification: bool = False,
    stage: str = "F40-S",
) -> str:
    """Return the guarded A00 text patch without writing anything."""
    if not qualification:
        raise ValueError(
            "unmarked micro generation is retired; use qualification=True "
            "with one of F40-S, F40-M, L99-S, or L99-M"
        )
    template = _render_micro_template(
        source,
        qualification=qualification,
        stage=stage,
    )
    source_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if template.count(_QUALIFICATION_SHA_PLACEHOLDER) != 1:
        raise RuntimeError(
            "qualification template must contain exactly one source-SHA "
            "placeholder"
        )
    if template.count(_QUALIFICATION_FOLDER_PLACEHOLDER) != 1:
        raise RuntimeError(
            "qualification template must contain exactly one source-folder "
            "placeholder"
        )
    emitted = template.replace(
        _QUALIFICATION_SHA_PLACEHOLDER,
        source_sha256,
    ).replace(
        _QUALIFICATION_FOLDER_PLACEHOLDER,
        source_sha256[:16],
    )
    verify_qualification_source_bytes(emitted.encode("utf-8"), source_sha256)
    return emitted


def write_micro_a00(
    *,
    qualification: bool = False,
    stage: str = "F40-S",
) -> str:
    """Patch A00 size knobs and write one generated MATLAB script."""
    if not qualification:
        raise ValueError(
            "unmarked micro generation is retired; use qualification=True "
            "with one of F40-S, F40-M, L99-S, or L99-M"
        )
    source_path = os.path.join(REPO, "scour_MATLAB", "A00_Run.m")
    with open(source_path, encoding="utf-8") as handle:
        source = handle.read()
    source = render_micro_a00(
        source, qualification=qualification, stage=stage
    )
    filename = f"micro_A00_qualification_{stage}.m"
    output = os.path.join(REPO, "scour_MATLAB", filename)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(source)
    return output


def render_toy_driver(source: str) -> str:
    """Return an explicitly isolated micro driver from the production driver.

    The production ladder is now derived from ``campaign_contract`` rather than
    containing editable dataset literals.  Override the resolved dataset at its
    single semantic assignment point, and narrowly relax rung validation only
    for this one generated s0 micro dataset.

    This is deliberately a LEGACY, non-publication mechanics fixture.  Its
    configs omit every campaign identity/runtime field and carry the conspicuous
    ``qualification_mode='toy_nonpublication_legacy'`` marker.  It therefore
    cannot enter the registered anchor/frozen HPO path or masquerade as campaign
    output.  The real production policy object is neither mutated nor shadowed.
    """
    raise RuntimeError(
        "the legacy toy dry-run is retired; use four-stage release "
        "qualification evidence instead"
    )
    source = _patch_once(
        source,
        (
            r"^(DATASET,\s*TARGET_SUPPORTS,\s*BEARING_TARGETS\s*=\s*"
            r"_LADDER\[STAGE\].*)$"
        ),
        (
            r"\g<0>"
            f'\nDATASET = "{MICRO_DS}"  # GENERATED MICRO-DRYRUN OVERRIDE'
        ),
        "resolved s0 dataset assignment",
    )
    provenance_override = f"""optuna.logging.set_verbosity(optuna.logging.WARNING)

# GENERATED MICRO-DRYRUN ONLY. Production code never enables this override.
# The source still passes all byte/provenance checks, but its deliberately
# reduced state/pass inventory is not allowed to claim the production rung.
import core.protocol as _micro_protocol
_micro_read_dataset_provenance = _micro_protocol.read_dataset_provenance

def _read_micro_provenance(dataset_dir, **requested):
    expected = {{
        "expected_stage": "F40-S",
        "expected_dataset": "{MICRO_DS}",
        "expected_target_supports": [2],
        "expected_bearing_targets": None,
    }}
    if os.path.basename(os.path.normpath(dataset_dir)) != "{MICRO_DS}":
        raise RuntimeError("micro provenance override refused a foreign dataset")
    for key, wanted in expected.items():
        if key in requested and requested[key] != wanted:
            raise RuntimeError(
                f"micro provenance override refused {{key}}={{requested[key]!r}}"
            )
    return _micro_read_dataset_provenance(dataset_dir)

_micro_protocol.read_dataset_provenance = _read_micro_provenance
"""
    source = _patch_once(
        source,
        r"^optuna\.logging\.set_verbosity\(optuna\.logging\.WARNING\)$",
        provenance_override,
        "micro provenance-isolation hook",
    )
    patches = (
        (
            (
                r'^N_TRIALS\s*=\s*HYPERPARAMETER_POLICY'
                r'\["anchor_hpo"\]\["n_trials"\]$'
            ),
            "N_TRIALS       = 2            # DRY-RUN",
            "trial count",
        ),
        (
            r"^EPOCHS         = 50$",
            "EPOCHS         = 2            # DRY-RUN",
            "epoch count",
        ),
        (
            (
                r"^USE_PRUNER\s*=\s*HYPERPARAMETER_POLICY\[\n"
                r'\s*"anchor_hpo"\n'
                r'\]\["use_registered_pruner"\]$'
            ),
            "USE_PRUNER     = False        # DRY-RUN",
            "pruner policy",
        ),
        (
            r"^SEEDS\s*=\s*list\(HPO_SEEDS\)$",
            "SEEDS          = [42]         # DRY-RUN",
            "seed list",
        ),
        (
            r"^import csv$",
            (
                "import sys\n"
                f"sys.path.insert(0, {REPO.replace(chr(92), '/')!r})\n"
                "import csv"
            ),
            "repository import path",
        ),
    )
    for pattern, replacement, label in patches:
        source = _patch_once(source, pattern, replacement, label)

    toy_make_config = '''def make_config(arch: dict, dofs: list[int], seed: int) -> dict:
    """Build one conspicuously non-publication LEGACY mechanics config."""
    dof_str = "_".join(str(d) for d in dofs)
    return {
        "name": (
            f"TOY_NONPUBLICATION_LEGACY_"
            f"{arch['name_short']}_DOFs_{dof_str}_seed{seed}"
        ),
        "seed": seed,
        "sensor_noise": SENSOR_NOISE,
        "name_short": arch["name_short"],
        "method": arch["method"],
        "dofs": list(dofs),
        "discretization": DISCRETIZATION,
        "use_space2vec": arch["use_space2vec"],
        "use_lstm": arch["use_lstm"],
        "use_nhits": arch["use_nhits"],
        "model_type": arch["model_type"],
        "task": TASK,
        "target_supports": TARGET_SUPPORTS,
        "bearing_targets": BEARING_TARGETS,
        "qualification_mode": "toy_nonpublication_legacy",
    }


def run_phase'''
    source = _patch_once(
        source,
        r"^def make_config\(arch: dict, dofs: list\[int\], seed: int\)"
        r" -> dict:[\s\S]*?^def run_phase",
        toy_make_config,
        "non-publication LEGACY toy config",
    )

    toy_main = '''if __name__ == "__main__":
    qualification_mode = "toy_nonpublication_legacy"
    print()
    print("*** TOY NON-PUBLICATION LEGACY MECHANICS RUN ***")
    print(
        f"dataset={DATASET}; qualification_mode={qualification_mode}; "
        f"trials={N_TRIALS}; epochs={EPOCHS}; seeds={SEEDS}"
    )
    print("No output from this generated driver is campaign evidence.")
    set_global_seed(SEEDS[0], TRAIN_PROTOCOL["determinism"])
    ARCHITECTURES = [ALL_ARCHITECTURES[0]]
    run_phase(qualification_mode, [ALL_DOFS[3]])
    print("*** TOY NON-PUBLICATION LEGACY MECHANICS RUN COMPLETE ***")
'''
    source = _patch_once(
        source,
        r'^if __name__ == "__main__":[\s\S]*\Z',
        toy_main,
        "non-publication LEGACY toy entry point",
    )
    return source


def write_dryrun(scratch: str) -> str:
    """Reject the retired toy dry-run before reading or writing any path."""
    raise RuntimeError(
        "--dryrun is retired because its no-argument micro dataset is not "
        "valid under the current Paper-1 stage-count contract"
    )



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dryrun",
        nargs="?",
        const=os.path.join(REPO, "_micro_dryrun_scratch"),
        metavar="SCRATCH",
        help="retired compatibility option; always rejected",
    )
    mode.add_argument(
        "--qualification",
        action="store_true",
        help="emit a marked, segregated MATLAB qualification script",
    )
    parser.add_argument(
        "--stage",
        choices=QUALIFICATION_STAGES,
        help="required Paper-1 qualification block",
    )
    args = parser.parse_args(argv)

    if args.dryrun is not None:
        parser.error(
            "--dryrun is retired; use --qualification --stage with one of "
            "F40-S, F40-M, L99-S, or L99-M"
        )
    if not args.qualification:
        parser.error(
            "no-argument micro generation is retired; use --qualification "
            "--stage with one of F40-S, F40-M, L99-S, or L99-M"
        )
    if args.stage is None:
        parser.error(
            "--qualification requires --stage with one of F40-S, F40-M, "
            "L99-S, or L99-M"
        )

    output = write_micro_a00(
        qualification=args.qualification,
        stage=args.stage,
    )
    print(f"qualification micro A00 -> {output}")
    print(
        "Run this script in MATLAB. Its manifest and states are explicitly "
        "marked release_qualification_run=true and are not campaign input."
    )
    print(
        "Before MATLAB, set TTBI_QUALIFICATION_HOST_ID to a stable unique "
        "label for this PC (for example home-laptop or labpc-01). The run "
        "fails closed without its authenticated host-diagnostic receipt."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

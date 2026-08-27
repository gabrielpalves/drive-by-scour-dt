"""Independent verifier for the exhaustive Paper-1 contact-closure gate.

No arguments run fast synthetic/mutation checks only.  A real authorization
requires an immutable MATLAB gate directory and an explicit create-once receipt:

    python check_contact_closure_gate.py D:/audit/contact_closure_paper1 \
        --source-commit <40-lowercase-hex> \
        --receipt D:/audit/contact_closure_authorization_receipt.json

Re-run the same command with ``--revalidate-receipt`` to authenticate a
retained receipt without replacing it.

The checker never invokes MATLAB and never selects cases.  It verifies the
pre-solve 420-case Cartesian inventory, recomputes every contact/GCI/waveform
decision from the JSON case projections, authenticates the case-artifact root,
and binds the result to clean Git HEAD and the current MATLAB source root.
"""

from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.dirname(__file__)
)
if _bootstrap_source_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_source_root)
_bootstrap_guard_dir = _bootstrap_os.path.join(
    _bootstrap_source_root, "campaign_import_guard"
)
_bootstrap_guard_init = _bootstrap_os.path.join(
    _bootstrap_guard_dir, "__init__.py"
)
if (
    not _bootstrap_os.path.isfile(_bootstrap_guard_init)
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
from pathlib import Path
import subprocess
import sys

from contact_gate_path_safety import (
    GateError,
    canonical_existing_directory,
    canonical_existing_file,
    canonical_receipt_path,
)
from contact_gate_source_contract import (
    DYNAMIC_DISPATCH_FILES,
    GATE_SOURCE_FILES,
    REVIEWED_TRACKPROP_CALL,
    REVIEWED_TRACKPROP_DISPATCH,
    SOLVER_MODULES,
    SOLVER_SOURCE_FILES,
    STUDY_HARNESS_FILES,
    TRACK_DEFINITION_FILES,
    _contact_source_set,
    _matlab_statements,
    _validate_dynamic_dispatch,
    _validate_matlab_source_contract,
)
from contact_gate_verifier_identity import verifier_source_root
from contact_gate_core import (
    CHANNELS,
    CHANNEL_SCHEMA_ID,
    CLOSURE_INTERPRETATION,
    COARSE_CORR,
    COARSE_NMAX,
    COARSE_NRMSE,
    COMMIT_RE,
    COMMON_DX_M,
    COMPARISON_WINDOW_ATOL,
    DT_MS,
    DatasetDescriptor,
    ENVIRONMENT_LOCK,
    EQUIVALENCE_ATOL,
    EQUIVALENCE_RTOL,
    EXPECTED_CASES,
    EXPECTED_FAMILIES,
    EXPECTED_L_BRIDGE_M,
    EXPECTED_MATLAB_ENVIRONMENT_SHA256,
    EXPECTED_MATLAB_RELEASE,
    EXPECTED_MATLAB_VERSION,
    EXPECTED_GEN_SCHEMA,
    EXPECTED_GENERATION_BEHAVIOR_VERSION,
    EXPECTED_PASSAGES,
    EXPECTED_STATES,
    EXPECTED_TOTAL_CASES,
    FINEST_IDENTITY_ATOL,
    FRACTION_GATE,
    GATES_N,
    GCI_FS,
    GCI_METHOD,
    GCI_P_MAX,
    GCI_P_MIN,
    MEDIUM_CORR,
    MEDIUM_NMAX,
    MEDIUM_NRMSE,
    NUMERIC_HASH_SELFCHECK,
    PLAIN_REPORT_FIELDS,
    POST_DECK_WINDOW_M,
    RECON_ATOL,
    RECON_RTOL,
    ROOT,
    SHA256_RE,
    STAGES,
    STUDY_SCHEMA,
    SUMMARY_SCHEMA,
    SelectionRow,
    TIME_GRID_ULPS,
    WAVEFORM_MONOTONIC_ATOL,
    _allclose,
    _as_list,
    _canonical_lf_file,
    _exact_keys,
    _first_json_mismatch,
    _float_list,
    _locked_matlab_environment,
    _normalised_lf,
    _same_float_list,
    _sha256_bytes,
    _sha256_file,
    _strict_integer,
    _strict_json_equivalent,
    _strict_json_file,
    _strict_json_text,
    _strict_number,
    _validate_actual_matlab_environment,
    _validate_utc_pair,
)
from contact_gate_policy import (
    _expected_policy_fields,
    _expected_policy_json,
    _expected_selection_records,
    _parse_dataset_line,
    _parse_policy,
    _parse_selection,
    _validate_inventory,
)
from contact_gate_numerics import (
    _contracts,
    _gci,
    _metric_columns,
    _validate_plain_report,
    _validate_public_gci,
)
from contact_gate_case import (
    _gate_execution_root,
    _generator_source_identity,
    _recompute_case,
    _solver_execution_identity,
    _study_harness_root,
)
from contact_gate_artifacts import (
    _dataset_digest_snapshot,
    _freeze_dataset_snapshot,
    _freeze_gate_snapshot,
    _gate_digest_snapshot,
    _git_clean_head,
    _require_clean_status,
    _validate_closure_host_attestation,
    _validate_gate_inventory,
    _validate_plan_marker,
    _validate_summary_datasets,
)
from contact_gate_dataset import (
    _descriptor_scalar,
    _expected_physical_descriptor,
    _mat_jsonish,
    _mat_scalar_text,
    _matlab_row_count,
    _passage_container_value,
    _validate_datasets_with_comparator,
    _validate_mat_sources,
)
from contact_gate_authorization import (
    _authorize_gate,
    _is_within,
    _publish_receipt,
    _revalidate_existing_receipt,
    _validate_receipt_location,
    _verify_gate_snapshot,
    verify_existing_authorization_receipt,
    verify_gate,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_dir", nargs="?")
    parser.add_argument("--source-commit")
    parser.add_argument("--receipt")
    parser.add_argument(
        "--revalidate-receipt",
        action="store_true",
        help=(
            "require and revalidate an existing create-once receipt instead "
            "of publishing a new one"
        ),
    )
    args = parser.parse_args(argv)
    if args.gate_dir is None:
        if (
            args.source_commit is not None
            or args.receipt is not None
            or args.revalidate_receipt
        ):
            parser.error(
                "--source-commit/--receipt/--revalidate-receipt "
                "require gate_dir"
            )
        # Lazy, self-test-only import: the real authorization path must
        # never load the synthetic fixture/self-test modules.
        from contact_gate_selftests import run_self_tests

        run_self_tests()
        return 0
    if args.source_commit is None or args.receipt is None:
        parser.error(
            "real verification requires --source-commit and --receipt"
        )
    try:
        receipt_path = Path(args.receipt)
        if args.revalidate_receipt:
            verify_existing_authorization_receipt(
                Path(args.gate_dir),
                args.source_commit,
                receipt_path,
            )
        else:
            _authorize_gate(
                Path(args.gate_dir),
                args.source_commit,
                receipt_path,
                revalidate_receipt=False,
            )
    except (GateError, OSError, subprocess.SubprocessError) as exc:
        print(f"CONTACT CLOSURE AUTHORIZATION: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "CONTACT CLOSURE AUTHORIZATION: PASS "
        f"({EXPECTED_TOTAL_CASES}/{EXPECTED_TOTAL_CASES}; "
        f"receipt={receipt_path.resolve()}; "
        f"mode={'revalidate' if args.revalidate_receipt else 'create-once'})"
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

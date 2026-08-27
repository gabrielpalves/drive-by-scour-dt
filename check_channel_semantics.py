"""Fail-closed executable contract for the physical8_v1 response schema.

The response-signature inventory retains ten legacy numerical-V&V diagnostics.
The pre-dispatch contact gate follows the deployed eight-channel learning
schema: its frozen ``Wheel{1,2}_Vert`` keys resolve to the constrained-wheelset
proxy, while ``AcelRodaPrimVag`` remains stored only as a legacy diagnostic.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent

CANONICAL_VV_CHANNELS = (
    "carbody_vertical_acc",
    "front_bogie_vertical_acc",
    "rear_bogie_vertical_acc",
    "rail_eulerian_vertical_acc_under_wheel_1",
    "rail_eulerian_vertical_acc_under_wheel_2",
    "rail_eulerian_vertical_acc_under_wheel_3",
    "rail_eulerian_vertical_acc_under_wheel_4",
    "carbody_pitch_rate",
    "front_bogie_pitch_rate",
    "rear_bogie_pitch_rate",
)
CANONICAL_PHYSICAL8_CHANNELS = (
    "carbody_vertical_acceleration",
    "front_bogie_vertical_acceleration",
    "rear_bogie_vertical_acceleration",
    "wheelset_1_constrained_vertical_acceleration_proxy",
    "wheelset_2_constrained_vertical_acceleration_proxy",
    "carbody_pitch_rate",
    "front_bogie_pitch_rate",
    "rear_bogie_pitch_rate",
)


class ContractError(AssertionError):
    """Raised when executable channel semantics drift."""


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise ContractError(message)


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _literal_assignment(source: str, variable: str):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == variable
               for target in targets):
            return ast.literal_eval(node.value)
    raise ContractError(f"missing literal assignment {variable}")


def _matlab_cell_strings(source: str, marker: str) -> tuple[str, ...]:
    try:
        tail = source.split(marker, 1)[1]
        cell = tail.split("};", 1)[0]
    except IndexError as exc:
        raise ContractError(f"missing MATLAB cell marker {marker!r}") from exc
    return tuple(re.findall(r"'([^']+)'", cell))


def verify(text: dict[str, str]) -> int:
    checks = Checks()

    # Contact qualification covers exactly the deployed physical8 array.
    checks.require(
        tuple(_literal_assignment(text["contact_gate_core.py"], "CHANNELS"))
        == CANONICAL_PHYSICAL8_CHANNELS,
        "Python contact-gate physical8 channel order drifted",
    )
    for filename, marker in (
        ("scour_MATLAB/contact_gate_policy_definition.m",
         "policy.expected_channels = {"),
        ("scour_MATLAB/contact_run_one.m", "run.channel_names = {"),
    ):
        checks.require(
            _matlab_cell_strings(text[filename], marker)
            == CANONICAL_PHYSICAL8_CHANNELS,
            f"{filename} physical8 channel order drifted",
        )
    checks.require(
        _matlab_cell_strings(
            text["scour_MATLAB/response_signature_run_one.m"],
            "vehicle_names = {",
        ) == CANONICAL_VV_CHANNELS,
        "response-signature V&V channel order drifted",
    )

    d01 = text["scour_MATLAB/D01_DataProcessing.m"]
    b17 = text["scour_MATLAB/B17_CalcUat.m"]
    b66 = text["scour_MATLAB/B66_ContactForce.m"]
    helper = text["scour_MATLAB/+ttbi/wheel_contact_kinematics.m"]
    checks.require(
        "N(x_w)'*A_rail" in d01
        and "AcelRodaPrimVag" in d01
        and "NOT wheelset/axle-box acceleration" in d01,
        "D01 no longer preserves/classifies the legacy Eulerian diagnostic",
    )
    checks.require(
        "AcelWheelsetPrimVag" in d01
        and "idealized model-predicted constrained-wheelset" in d01
        and "axle-box response proxy" in d01,
        "D01 lacks the qualified physical8_v1 wheelset proxy description",
    )
    checks.require(
        "Eulerian partial-time field" in b17
        and "shape_fun_at_x_t*Sol.Model.Nodal.A" in b17,
        "B17 no longer identifies/calculates the Eulerian rail field",
    )
    checks.require(
        d01.count("ttbi.wheel_contact_kinematics") >= 2
        and b66.count("ttbi.wheel_contact_kinematics") >= 2,
        "D01 and B66 must disclose and call the shared kinematics helper",
    )
    for token in (
        "active = calc_veh.elexj > 0;",
        "sol_veh.acc_under ...",
        "+ 2*vel*sol_veh.vel_under_p ...",
        "+ vel^2*sol_veh.def_under_pp ...",
        "+ calc_veh.hdd_path .* active;",
    ):
        checks.require(token in helper, f"wheelset helper lost term/mask: {token}")
    checks.require(
        "measured exactly zero mask-only delta" in helper
        and "geometries/speeds are not inferred" in helper,
        "wheelset helper overstates the measured mask impact",
    )
    forbidden_overclaims = (
        "would measure",
        "wheel/axle-box quantity",
        "measured axle-box acceleration",
    )
    checks.require(
        not any(phrase in (d01 + helper).lower()
                for phrase in forbidden_overclaims),
        "active MATLAB prose overclaims physical sensor semantics",
    )

    dataset = text["core/dataset.py"]
    campaign = text["core/campaign_contract.py"]
    protocol = text["core/protocol.py"]
    identity = text["scour_MATLAB/+ttbi/build_generation_identity.m"]
    execution_context = text["scour_MATLAB/+ttbi/build_execution_context.m"]
    case_info = text["scour_MATLAB/+ttbi/build_case_info.m"]
    sidecar_validator = text["scour_MATLAB/+ttbi/validate_generation_sidecars.m"]
    smoke = text["scour_MATLAB/smoke_audit.m"]
    checks.require(
        'EXPECTED_CHANNEL_SCHEMA_ID = "physical8_v1"' in campaign
        and '"channel_schema_id": EXPECTED_CHANNEL_SCHEMA_ID' in campaign,
        "reviewed campaign contract does not bind physical8_v1",
    )
    checks.require(
        "channel_schema_id = 'physical8_v1';" in identity
        and identity.count("'channel_schema_id', channel_schema_id") >= 2
        and "'channel_schema_id', derived_identity.channel_schema_id" in execution_context
        and "'channel_schema_id', identity.channel_schema_id" in case_info,
        "MATLAB fingerprint/worker context/case manifest does not bind channel_schema_id",
    )
    checks.require(
        "'state_design_kind', state.state_design_kind" in identity
        and "'state_design_kind', derived_identity.state_design_kind" in execution_context
        and "context.identity.state_design_kind" in sidecar_validator
        and "'state_design_kind', identity.state_design_kind" in case_info,
        "MATLAB fingerprint/worker context/case manifest does not bind state_design_kind",
    )
    checks.require(
        "3: ('AcelWheelsetPrimVag', 0)" in dataset
        and "4: ('AcelWheelsetPrimVag', 1)" in dataset
        and "3: ('AcelRodaPrimVag', 0)" not in dataset
        and "4: ('AcelRodaPrimVag', 1)" not in dataset,
        "deployed loader DOFs 3-4 are not exclusively wheelset-backed",
    )
    for token in (
        '"channel_schema_id": _required_mat_text(',
        'manifest_generation["channel_schema_id"]',
        '"channel_schema_id": manifest["channel_schema_id"]',
        "'AcelWheelsetPrimVag', 'PitchPrimVag'",
        '"AcelWheelsetPrimVag": 4',
        'CACHE_SCHEMA_TAG = "_gs9"',
        '"channel_schema_id": (',
    ):
        checks.require(token in dataset, f"loader/cache schema guard missing: {token}")
    checks.require(
        '"channel_schema_id": generation["channel_schema_id"]' in protocol
        and '"expected_channel_schema_id": _EXPECTED_CHANNEL_SCHEMA_ID' in protocol
        # Tripwire: bumping the protocol version must force a re-read of the
        # two channel-semantics conjuncts above. v8 (2026-08-26) carries the
        # capability-based environment lock, execution-block policy v3, and the
        # exclusion of the wheelset proxies from learning; physical8_v1 remains
        # the on-disk response inventory. check_protocol_hash.py pins the same
        # number independently.
        and '"protocol_version": 8' in protocol,
        "protocol descriptor/provenance does not expose physical8_v1",
    )
    for token in (
        "physical8_v1 manufactured four-term + active mask",
        "ttbi.wheel_contact_kinematics",
        "term1_ + term2_ + term3_ + term4_",
        "CalcW_.hdd_path .* activeW_expected_",
        "isequal(activeW_, activeW_expected_)",
    ):
        checks.require(token in smoke, f"mandatory manufactured smoke missing: {token}")

    utils = text["core/utils.py"]
    plots = text["plotting/aggregate_ablation.py"]
    checks.require(
        "idealized constrained-wheelset vertical acceleration" in utils
        and "axle-box response proxy" in utils,
        "core DOF descriptions still expose the legacy rail-field meaning",
    )
    checks.require(
        "Wheelset accel proxy 1 (Wheel1_Vert)" in plots
        and "Rail Eulerian accel @ wheel 1" not in plots,
        "plot labels do not reflect physical8_v1",
    )
    return checks.count


def main() -> None:
    files = (
        "contact_gate_core.py",
        "scour_MATLAB/contact_gate_policy_definition.m",
        "scour_MATLAB/contact_run_one.m",
        "scour_MATLAB/response_signature_run_one.m",
        "scour_MATLAB/D01_DataProcessing.m",
        "scour_MATLAB/B17_CalcUat.m",
        "scour_MATLAB/B66_ContactForce.m",
        "scour_MATLAB/+ttbi/wheel_contact_kinematics.m",
        "scour_MATLAB/+ttbi/build_generation_identity.m",
        "scour_MATLAB/+ttbi/build_execution_context.m",
        "scour_MATLAB/+ttbi/build_case_info.m",
        "scour_MATLAB/+ttbi/validate_generation_sidecars.m",
        "scour_MATLAB/smoke_audit.m",
        "core/campaign_contract.py",
        "core/dataset.py",
        "core/protocol.py",
        "core/utils.py",
        "plotting/aggregate_ablation.py",
    )
    text = {name: _read(name) for name in files}
    count = verify(text)

    mutations = (
        (
            "loader fallback to legacy diagnostic",
            "core/dataset.py",
            "3: ('AcelWheelsetPrimVag', 0)",
            "3: ('AcelRodaPrimVag', 0)",
        ),
        (
            "profile-inertia term removed",
            "scour_MATLAB/+ttbi/wheel_contact_kinematics.m",
            "+ calc_veh.hdd_path .* active;",
            "+ zeros(size(calc_veh.hdd_path));",
        ),
        (
            "schema omitted from generation fingerprint",
            "scour_MATLAB/+ttbi/build_generation_identity.m",
            "'channel_schema_id', channel_schema_id, ...",
            "'unrelated_schema', channel_schema_id, ...",
        ),
        (
            "schema omitted from worker context",
            "scour_MATLAB/+ttbi/build_execution_context.m",
            "'channel_schema_id', derived_identity.channel_schema_id, ...",
            "'unrelated_schema', derived_identity.channel_schema_id, ...",
        ),
        (
            "state design kind omitted from worker identity",
            "scour_MATLAB/+ttbi/build_execution_context.m",
            "'state_design_kind', derived_identity.state_design_kind, ...",
            "'unrelated_design_kind', derived_identity.state_design_kind, ...",
        ),
    )
    for name, filename, old, new in mutations:
        mutant = dict(text)
        if old not in mutant[filename]:
            raise ContractError(f"mutation anchor missing: {name}")
        mutant[filename] = mutant[filename].replace(old, new, 1)
        try:
            verify(mutant)
        except ContractError:
            pass
        else:
            raise ContractError(f"mutation self-test failed: {name}")

    print(f"CHANNEL SEMANTICS: PASS ({count} checks + {len(mutations)} mutations)")


if __name__ == "__main__":
    main()

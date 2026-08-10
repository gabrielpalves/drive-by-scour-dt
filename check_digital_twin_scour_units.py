"""Fast unit-contract checks for digital-twin to TTBI scour conversion.

This check deliberately avoids a full vehicle--track--bridge solve. It guards
the scientific unit boundary: lifecycle state is stored in percentage points,
while TTBI consumes the dimensionless support-stiffness-loss fraction ``d`` in
``k_v=(1-d)k_v0``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from digital_twin.scour_multi import MultiScourModel


def check_multi_support_conversion() -> None:
    model = MultiScourModel(
        [0.0, 20.0, 40.0, 60.0, 80.0],
        rng=np.random.default_rng(0),
    )
    model.current_X[:] = [-5.0, 0.0, 30.0, 60.0, 75.0]

    np.testing.assert_allclose(
        model.get_scour_rates(),
        [0.0, 0.0, 0.30, 0.60, 0.60],
    )
    np.testing.assert_allclose(
        model.get_normalized_severity(),
        [0.0, 0.0, 0.50, 1.00, 1.00],
    )

    retained = (1.0 - model.get_scour_rates()) * 344e6
    np.testing.assert_allclose(
        retained,
        [344e6, 344e6, 0.70 * 344e6, 0.40 * 344e6, 0.40 * 344e6],
    )


def check_evolve_units() -> None:
    model = MultiScourModel([0.0], rng=np.random.default_rng(1))
    returned = model.evolve(1.0)
    np.testing.assert_allclose(returned, model.get_scour_rates())
    assert 0.0 <= returned[0] <= 0.60


def check_physical_asset_converts_once() -> None:
    # Import here so the NumPy-only checks above stay useful even in a minimal
    # environment without the estimator stack.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "TTBI_2D"))
    from digital_twin import assets

    class Config:
        enable_shock = False
        n_veh = 5
        n_prop = 3

    physical = assets.PhysicalAsset(Config())
    physical.state_continuous = 30.0

    sentinel = np.empty((8, 0), dtype=np.float32)
    with patch.object(assets, "run_single_passage", return_value=sentinel) as run:
        observed = physical.get_observation_signal()

    assert observed is sentinel
    assert run.call_args.kwargs["damage_percent"] == 0.30


def check_legacy_physics_boundary() -> None:
    from digital_twin.physics import run_single_passage

    for invalid in (-0.01, 1.01, np.nan, np.inf):
        try:
            run_single_passage(invalid, add_signal_noise=False)
        except ValueError:
            continue
        raise AssertionError(f"invalid TTBI loss fraction escaped: {invalid!r}")


def main() -> None:
    check_multi_support_conversion()
    check_evolve_units()
    check_physical_asset_converts_once()
    check_legacy_physics_boundary()
    print("DIGITAL-TWIN SCOUR UNIT CONTRACT: ALL PASS (0/30/60% checked)")


if __name__ == "__main__":
    main()

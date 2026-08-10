"""Protocol-separated robustness and stability evaluations.

Development adjudication and sealed-test stability answer different questions
and therefore have different interfaces:

``evaluate_development_adjudication``
    Repeated stratified grouped folds over the development pool only. Split
    seeds and initialization seeds are mandatory prospective inputs. This
    phase may inform the frozen choice and cannot receive outer-test indices.

``evaluate_post_freeze_stability``
    Refit an already frozen configuration on the complete development pool and
    evaluate it on explicitly named sealed outer-test indices. This phase is
    report-only and cannot be used for selection.

Both interfaces checkpoint after each refit and bind the checkpoint to the
complete evaluation plan. The older one-split, internally generated 30-seed
routine is intentionally absent.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from tqdm import tqdm

from core.dataset import get_or_create_cache
from core.statistical_inference import (
    GroupedFold,
    repeated_stratified_group_folds,
)
from training.trainer import fit_predict_fixed_group_fold, run_single_training


FoldEvaluator = Callable[..., Mapping[str, Any]]


def run_development_adjudication(
    *,
    config: Mapping[str, Any],
    params: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    development_idx: np.ndarray,
    strata_by_state: Sequence[str],
    split_seeds: Sequence[int],
    initialization_seeds: Sequence[int],
    n_splits: int,
    n_repeats: int,
    n_epochs: int,
    max_epochs: int,
    n_scour_heads: int,
    checkpoint_path: str | None = None,
    fit_evaluate: FoldEvaluator = fit_predict_fixed_group_fold,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Adjudicate a fixed candidate using development data only.

    A distinct repeated grouped-CV schedule is constructed for every supplied
    split seed with :func:`repeated_stratified_group_folds`. Every fold is then
    refitted for every supplied initialization seed. Neither a seed schedule
    nor a train/validation split is synthesized inside this function.
    """

    X, y, groups = _validated_arrays(X, y, groups)
    development_idx = _validated_indices(
        "development_idx", development_idx, len(groups)
    )
    split_seeds = _validated_seeds("split_seeds", split_seeds)
    initialization_seeds = _validated_seeds(
        "initialization_seeds", initialization_seeds
    )
    n_splits = _positive_int("n_splits", n_splits, minimum=2)
    n_repeats = _positive_int("n_repeats", n_repeats)
    n_epochs = _positive_int("n_epochs", n_epochs)
    max_epochs = _positive_int("max_epochs", max_epochs)
    if n_epochs > max_epochs:
        raise ValueError("n_epochs cannot exceed max_epochs")
    n_scour_heads = _positive_int("n_scour_heads", n_scour_heads)

    descriptor = {
        "config": _json_value(dict(config)),
        "params": _json_value(dict(params)),
        "provenance": _json_value(dict(provenance or {})),
        "development_idx_sha256": _array_digest(development_idx),
        "groups_sha256": _array_digest(groups),
        "strata_by_state": [str(value) for value in strata_by_state],
        "split_seeds": list(split_seeds),
        "initialization_seeds": list(initialization_seeds),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_epochs": n_epochs,
        "max_epochs": max_epochs,
        "n_scour_heads": n_scour_heads,
    }
    result = _load_or_initialize(
        checkpoint_path,
        schema="ttbi-development-adjudication-v1",
        evaluation_scope="development_adjudication",
        selection_permitted=True,
        outer_test_observations_accessed=False,
        descriptor=descriptor,
    )
    completed = {_run_key(run) for run in result["runs"]}

    tasks: list[tuple[int, GroupedFold, int]] = []
    for split_seed in split_seeds:
        folds = repeated_stratified_group_folds(
            groups,
            development_idx,
            strata_by_state,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=split_seed,
        )
        for fold in folds:
            for initialization_seed in initialization_seeds:
                task = (split_seed, fold, initialization_seed)
                if _task_key(*task) not in completed:
                    tasks.append(task)

    for split_seed, fold, initialization_seed in tqdm(
        tasks, desc="Development adjudication", unit="refit"
    ):
        metrics = fit_evaluate(
            config=dict(config),
            params=dict(params),
            X=X,
            y=y,
            groups=groups,
            fold=fold,
            seed=initialization_seed,
            n_epochs=n_epochs,
            max_epochs=max_epochs,
            n_scour_heads=n_scour_heads,
        )
        result["runs"].append({
            "split_seed": split_seed,
            "repeat": int(fold.repeat),
            "fold": int(fold.fold),
            "initialization_seed": initialization_seed,
            "validation_states": fold.val_states.astype(int).tolist(),
            "metrics": _validated_metrics(metrics, fold.val_states),
        })
        _save_checkpoint(checkpoint_path, result)

    result["complete"] = True
    result["n_completed_refits"] = len(result["runs"])
    result["n_expected_refits"] = (
        len(split_seeds) * n_repeats * n_splits * len(initialization_seeds)
    )
    if result["n_completed_refits"] != result["n_expected_refits"]:
        raise RuntimeError("development-adjudication checkpoint is incomplete")
    _save_checkpoint(checkpoint_path, result)
    return result


def run_post_freeze_stability(
    *,
    config: Mapping[str, Any],
    params: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    development_idx: np.ndarray,
    sealed_outer_test_idx: np.ndarray,
    initialization_seeds: Sequence[int],
    n_epochs: int,
    max_epochs: int,
    n_scour_heads: int,
    checkpoint_path: str | None = None,
    fit_evaluate: FoldEvaluator = fit_predict_fixed_group_fold,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report post-freeze stability on the explicitly sealed outer test.

    This function has no split-seed argument and performs no candidate choice.
    It trains each prospectively supplied initialization seed on the complete
    development partition and evaluates exactly once on the sealed partition.
    """

    X, y, groups = _validated_arrays(X, y, groups)
    development_idx = _validated_indices(
        "development_idx", development_idx, len(groups)
    )
    outer_idx = _validated_indices(
        "sealed_outer_test_idx", sealed_outer_test_idx, len(groups)
    )
    initialization_seeds = _validated_seeds(
        "initialization_seeds", initialization_seeds
    )
    n_epochs = _positive_int("n_epochs", n_epochs)
    max_epochs = _positive_int("max_epochs", max_epochs)
    if n_epochs > max_epochs:
        raise ValueError("n_epochs cannot exceed max_epochs")
    n_scour_heads = _positive_int("n_scour_heads", n_scour_heads)
    fold = _sealed_partition_fold(groups, development_idx, outer_idx)

    descriptor = {
        "config": _json_value(dict(config)),
        "params": _json_value(dict(params)),
        "provenance": _json_value(dict(provenance or {})),
        "development_idx_sha256": _array_digest(development_idx),
        "sealed_outer_test_idx_sha256": _array_digest(outer_idx),
        "groups_sha256": _array_digest(groups),
        "initialization_seeds": list(initialization_seeds),
        "n_epochs": n_epochs,
        "max_epochs": max_epochs,
        "n_scour_heads": n_scour_heads,
    }
    result = _load_or_initialize(
        checkpoint_path,
        schema="ttbi-post-freeze-stability-v1",
        evaluation_scope="sealed_outer_test_post_freeze",
        selection_permitted=False,
        outer_test_observations_accessed=True,
        descriptor=descriptor,
    )
    completed = {int(run["initialization_seed"]) for run in result["runs"]}
    for initialization_seed in tqdm(
        [seed for seed in initialization_seeds if seed not in completed],
        desc="Post-freeze stability",
        unit="refit",
    ):
        metrics = fit_evaluate(
            config=dict(config),
            params=dict(params),
            X=X,
            y=y,
            groups=groups,
            fold=fold,
            seed=initialization_seed,
            n_epochs=n_epochs,
            max_epochs=max_epochs,
            n_scour_heads=n_scour_heads,
        )
        result["runs"].append({
            "initialization_seed": initialization_seed,
            "outer_test_states": fold.val_states.astype(int).tolist(),
            "metrics": _validated_metrics(metrics, fold.val_states),
        })
        _save_checkpoint(checkpoint_path, result)

    result["complete"] = True
    result["n_completed_refits"] = len(result["runs"])
    result["n_expected_refits"] = len(initialization_seeds)
    if result["n_completed_refits"] != result["n_expected_refits"]:
        raise RuntimeError("post-freeze stability checkpoint is incomplete")
    _save_checkpoint(checkpoint_path, result)
    return result


def evaluate_development_adjudication(
    study,
    config: dict,
    dataset_name: str,
    *,
    development_idx: np.ndarray,
    strata_by_state: Sequence[str],
    split_seeds: Sequence[int],
    initialization_seeds: Sequence[int],
    n_splits: int,
    n_repeats: int,
    n_epochs: int,
    max_epochs: int,
    n_scour_heads: int,
    cache_dir: str = "",
    output_dir: str = "",
) -> dict[str, Any]:
    """Cache-loading entry point for development adjudication."""

    X, y, _, groups = get_or_create_cache(config, dataset_name, cache_dir)
    return run_development_adjudication(
        config=config,
        params=study.best_params,
        X=X,
        y=y,
        groups=groups,
        development_idx=development_idx,
        strata_by_state=strata_by_state,
        split_seeds=split_seeds,
        initialization_seeds=initialization_seeds,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_epochs=n_epochs,
        max_epochs=max_epochs,
        n_scour_heads=n_scour_heads,
        checkpoint_path=os.path.join(
            output_dir, "robustness_development_adjudication.json"
        ),
        provenance={"dataset_name": dataset_name},
    )


def evaluate_post_freeze_stability(
    config: dict,
    frozen_params: Mapping[str, Any],
    dataset_name: str,
    *,
    development_idx: np.ndarray,
    sealed_outer_test_idx: np.ndarray,
    initialization_seeds: Sequence[int],
    n_epochs: int,
    max_epochs: int,
    n_scour_heads: int,
    cache_dir: str = "",
    output_dir: str = "",
) -> dict[str, Any]:
    """Cache-loading entry point for report-only sealed-test stability."""

    X, y, _, groups = get_or_create_cache(config, dataset_name, cache_dir)
    return run_post_freeze_stability(
        config=config,
        params=frozen_params,
        X=X,
        y=y,
        groups=groups,
        development_idx=development_idx,
        sealed_outer_test_idx=sealed_outer_test_idx,
        initialization_seeds=initialization_seeds,
        n_epochs=n_epochs,
        max_epochs=max_epochs,
        n_scour_heads=n_scour_heads,
        checkpoint_path=os.path.join(
            output_dir, "robustness_post_freeze_stability.json"
        ),
        provenance={"dataset_name": dataset_name},
    )


def evaluate_parametric_robustness(
    study,
    config: dict,
    dataset_name: str,
    baseline_mse: float,
    *,
    initialization_seed: int,
    n_epochs: int = 50,
    cache_dir: str = "",
    output_dir: str = "",
) -> tuple[float, float]:
    """Legacy one-at-a-time perturbation diagnostic with an explicit seed."""

    seed = _validated_seeds("initialization_seed", [initialization_seed])[0]
    X, y, _, groups = get_or_create_cache(config, dataset_name, cache_dir)
    best_params = study.best_params
    json_path = os.path.join(
        output_dir, f"robustness_sensitivity_seed{seed}.json"
    )
    multipliers = [0.90, 0.95, 1.00, 1.05, 1.10]
    candidate_params = [
        "lr", "weight_decay", "dropout_l0", "n_filters_l0", "lstm_hidden_size"
    ]
    params_to_test = {
        name: best_params[name] for name in candidate_params if name in best_params
    }
    if not params_to_test:
        raise ValueError("best_params contains no registered perturbation parameter")
    results = _load_json(json_path, default={})
    tasks = []
    for param_name, base_val in params_to_test.items():
        results.setdefault(param_name, {})
        for multiplier in multipliers:
            key = f"{multiplier * 100:.0f}%"
            if key not in results[param_name]:
                tasks.append((param_name, base_val, multiplier, key))
    for param_name, base_val, multiplier, key in tqdm(
        tasks, desc="Perturbations", unit="run"
    ):
        perturbed = copy.deepcopy(best_params)
        perturbed[param_name] = _perturb(base_val, multiplier)
        metrics = run_single_training(
            config,
            perturbed,
            X,
            y,
            seed=seed,
            n_epochs=n_epochs,
            groups=groups,
            dataset_name=dataset_name,
        )
        results[param_name][key] = {
            "mae": float(metrics["mae"]),
            "mse": float(metrics["mse"]),
        }
        _atomic_json(json_path, results)
    worst_mse = max(
        value["mse"]
        for param_results in results.values()
        for value in param_results.values()
    )
    return float(worst_mse), float(worst_mse - baseline_mse)


def _validated_arrays(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    if len(X) != len(y) or len(y) != len(groups) or len(groups) == 0:
        raise ValueError("X, y and groups must contain the same non-zero samples")
    return X, y, groups


def _validated_indices(name: str, values: np.ndarray, n_samples: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} contains duplicate sample indices")
    if result.min() < 0 or result.max() >= n_samples:
        raise IndexError(f"{name} contains an out-of-range sample index")
    return np.sort(result)


def _validated_seeds(name: str, values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of integers")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence of integers") from exc
    if not values:
        raise ValueError(f"{name} must be a non-empty sequence of integers")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must contain only integers")
        value = int(value)
        if not 0 <= value < 2**63:
            raise ValueError(f"{name} values must be in [0, 2**63)")
        result.append(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate seeds")
    return tuple(result)


def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sealed_partition_fold(
    groups: np.ndarray,
    development_idx: np.ndarray,
    outer_idx: np.ndarray,
) -> GroupedFold:
    if np.intersect1d(development_idx, outer_idx).size:
        raise ValueError("development and sealed outer-test samples overlap")
    if not np.array_equal(
        np.sort(np.concatenate([development_idx, outer_idx])),
        np.arange(len(groups), dtype=np.int64),
    ):
        raise ValueError(
            "development_idx and sealed_outer_test_idx must partition all samples"
        )
    dev_mask = np.zeros(len(groups), dtype=bool)
    dev_mask[development_idx] = True
    outer_mask = np.zeros(len(groups), dtype=bool)
    outer_mask[outer_idx] = True
    train_states, val_states = [], []
    for state in np.unique(groups):
        state_mask = groups == state
        n_state = int(state_mask.sum())
        n_dev = int(np.count_nonzero(state_mask & dev_mask))
        n_outer = int(np.count_nonzero(state_mask & outer_mask))
        if (n_dev, n_outer) == (n_state, 0):
            train_states.append(int(state))
        elif (n_dev, n_outer) == (0, n_state):
            val_states.append(int(state))
        else:
            raise ValueError(
                f"state {state!r} crosses development and sealed outer test"
            )
    if not train_states or not val_states:
        raise ValueError("both development and sealed outer test need whole states")
    return GroupedFold(
        repeat=0,
        fold=0,
        train_idx=development_idx,
        val_idx=outer_idx,
        train_states=np.asarray(train_states, dtype=np.int64),
        val_states=np.asarray(val_states, dtype=np.int64),
    )


def _validated_metrics(
    metrics: Mapping[str, Any], expected_states: np.ndarray
) -> dict[str, Any]:
    if not isinstance(metrics, Mapping) or "state" not in metrics:
        raise ValueError("fold evaluator must return a mapping with a state field")
    states = np.asarray(metrics["state"], dtype=np.int64)
    if states.ndim != 1 or len(np.unique(states)) != len(states):
        raise ValueError("fold evaluator returned invalid state identifiers")
    if not np.array_equal(np.sort(states), np.sort(expected_states)):
        raise ValueError("fold evaluator did not return exactly the validation states")
    return _json_value(dict(metrics))


def _task_key(
    split_seed: int, fold: GroupedFold, initialization_seed: int
) -> tuple[int, int, int, int]:
    return split_seed, int(fold.repeat), int(fold.fold), initialization_seed


def _run_key(run: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(run["split_seed"]),
        int(run["repeat"]),
        int(run["fold"]),
        int(run["initialization_seed"]),
    )


def _array_digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint values must be finite")
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _load_or_initialize(
    path: str | None,
    *,
    schema: str,
    evaluation_scope: str,
    selection_permitted: bool,
    outer_test_observations_accessed: bool,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema": schema,
        "evaluation_scope": evaluation_scope,
        "selection_permitted": selection_permitted,
        "outer_test_observations_accessed": outer_test_observations_accessed,
        "plan": _json_value(dict(descriptor)),
    }
    if path is None or not os.path.exists(path):
        return {**expected, "complete": False, "runs": []}
    result = _load_json(path)
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(
                f"checkpoint {path!r} does not match the requested {key}"
            )
    if not isinstance(result.get("runs"), list):
        raise RuntimeError(f"checkpoint {path!r} has no valid runs list")
    return result


def _save_checkpoint(path: str | None, result: Mapping[str, Any]) -> None:
    if path is not None:
        _atomic_json(path, result)


def _load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _atomic_json(path: str, value: Mapping[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(_json_value(dict(value)), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _perturb(base_val: float | int, multiplier: float) -> float | int:
    new_val = base_val * multiplier
    if isinstance(base_val, int):
        return max(1, round(new_val))
    return new_val

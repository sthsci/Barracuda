"""Donor-aware simulation and posterior analysis.

This module contains the numerical parts of the donor-aware notebook and web
reporting workflows without Plotly or Dash.  It deliberately distinguishes two
kinds of pairing:

* parameters belonging to the same posterior particle retain their ``chain``
  and ``draw`` labels; and
* independently fitted conditions are compared with Cartesian posterior pairs
  (or reproducible independent Monte Carlo pairs when the Cartesian product is
  too large).

The inference backend historically calls the zero-inflated fraction ``phi_0``.
Public tables from this module use the package-wide ``p_zero`` spelling while
accepting either variable name in an :class:`arviz.InferenceData` object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Final, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr


DONOR_MODEL_KEYS: Final[tuple[str, ...]] = (
    "homo",
    "z2p",
    "dis2p",
    "hetero3",
)

_MODEL_ALIASES: Final[dict[str, str]] = {
    "homo": "homo",
    "homogeneous": "homo",
    "homogeneouspoisson": "homo",
    "poisson": "homo",
    "z2p": "z2p",
    "zi": "z2p",
    "zeroinflated": "z2p",
    "zeroinflatedpoisson": "z2p",
    "dis2p": "dis2p",
    "distributed": "dis2p",
    "gamma": "dis2p",
    "gammapoisson": "dis2p",
    "hetero3": "hetero3",
    "full": "hetero3",
    "zir": "hetero3",
    "zigamma": "hetero3",
    "zeroinflatedgammapoisson": "hetero3",
}

_POPULATION_VARIABLES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("mu_lambda_population", ("mu_lambda_population",)),
    ("sigma_lambda_population", ("sigma_lambda_population",)),
    ("p_zero_population", ("p_zero_population", "phi_0_population")),
)
_DONOR_VARIABLES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("mu_lambda_donor", ("mu_lambda_donor",)),
    ("sigma_lambda_donor", ("sigma_lambda_donor",)),
    ("p_zero_donor", ("p_zero_donor", "phi_0_donor")),
)


def canonical_donor_model_key(model: object) -> str:
    """Return one lowercase key for a donor-aware event-count model.

    Spaces, underscores and hyphens are ignored, so both backend spellings
    such as ``"Dis2P"`` and descriptive names such as ``"gamma-poisson"``
    resolve to the stable public key ``"dis2p"``.
    """

    normalized = (
        str(model)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )
    try:
        return _MODEL_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown donor-aware model {model!r}; choose from: "
            + ", ".join(DONOR_MODEL_KEYS)
        ) from exc


def _finite_float(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _validated_seed(seed: int | None, name: str = "seed") -> int | None:
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    converted = int(seed)
    if converted < 0 or converted > np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must be between 0 and 2**32 - 1")
    return converted


def _donor_size_items(
    values: Mapping[object, int] | Sequence[tuple[object, int]],
) -> tuple[tuple[str, int], ...]:
    if isinstance(values, Mapping):
        items = list(values.items())
    elif isinstance(values, (Sequence, np.ndarray)) and not isinstance(
        values, (str, bytes)
    ):
        items = list(values)
    else:
        raise ValueError("donor_sizes must map donor labels to positive cell counts")
    if not items:
        raise ValueError("donor_sizes must contain at least one donor")

    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, (Sequence, np.ndarray)) or len(item) != 2:
            raise ValueError("donor_sizes entries must be (donor_id, n_cells) pairs")
        raw_label, raw_size = item
        label = str(raw_label).strip()
        if not label:
            raise ValueError("donor labels must not be empty")
        if label in seen:
            raise ValueError(f"duplicate donor label after normalization: {label!r}")
        if isinstance(raw_size, (bool, np.bool_)) or not isinstance(
            raw_size, (int, np.integer)
        ):
            raise ValueError(f"cell count for donor {label!r} must be a positive integer")
        size = int(raw_size)
        if size <= 0:
            raise ValueError(f"cell count for donor {label!r} must be a positive integer")
        normalized.append((label, size))
        seen.add(label)
    return tuple(normalized)


def _donor_parameter_values(
    values: Real | Mapping[object, Real] | Sequence[Real],
    *,
    donor_ids: Sequence[str],
    name: str,
) -> tuple[float, ...]:
    if isinstance(values, (Real, np.number)) and not isinstance(values, (bool, np.bool_)):
        raw = [values] * len(donor_ids)
    elif isinstance(values, Mapping):
        normalized_mapping: dict[str, Real] = {}
        for key, value in values.items():
            normalized_key = str(key).strip()
            if normalized_key in normalized_mapping:
                raise ValueError(f"{name} contains duplicate donor labels")
            normalized_mapping[normalized_key] = value
        missing = set(donor_ids).difference(normalized_mapping)
        extra = set(normalized_mapping).difference(donor_ids)
        if missing or extra:
            raise ValueError(
                f"{name} must name every donor exactly once "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )
        raw = [normalized_mapping[donor_id] for donor_id in donor_ids]
    elif isinstance(values, (Sequence, np.ndarray)) and not isinstance(
        values, (str, bytes)
    ):
        raw = list(values)
        if len(raw) != len(donor_ids):
            raise ValueError(
                f"{name} must contain one value per donor; "
                f"expected {len(donor_ids)}, received {len(raw)}"
            )
    else:
        raise ValueError(f"{name} must be a scalar, mapping, or sequence")
    return tuple(_finite_float(value, f"{name}[{donor}]") for donor, value in zip(donor_ids, raw))


@dataclass(frozen=True)
class DonorSimulationSpec:
    """Validated settings for one donor-aware event-count simulation.

    ``donor_sizes`` preserves insertion order and may contain unequal cell
    counts.  Each biological parameter may be a scalar shared by all donors,
    a donor-labelled mapping, or a sequence aligned to ``donor_sizes``.

    Parameters are moments among engaging cells: ``mu_lambda`` and
    ``sigma_lambda`` describe the donor's rate distribution, while ``p_zero``
    is its structural-zero probability.  Model-incompatible nonzero values are
    rejected rather than silently discarded.
    """

    donor_sizes: Mapping[object, int] | Sequence[tuple[object, int]]
    mu_lambda: Real | Mapping[object, Real] | Sequence[Real]
    model_key: str = "hetero3"
    sigma_lambda: Real | Mapping[object, Real] | Sequence[Real] = 0.0
    p_zero: Real | Mapping[object, Real] | Sequence[Real] = 0.0
    observation_time: float = 1.0
    seed: int | None = 2026

    def __post_init__(self) -> None:
        sizes = _donor_size_items(self.donor_sizes)
        donor_ids = tuple(label for label, _ in sizes)
        mu = _donor_parameter_values(
            self.mu_lambda,
            donor_ids=donor_ids,
            name="mu_lambda",
        )
        sigma = _donor_parameter_values(
            self.sigma_lambda,
            donor_ids=donor_ids,
            name="sigma_lambda",
        )
        p_zero = _donor_parameter_values(
            self.p_zero,
            donor_ids=donor_ids,
            name="p_zero",
        )
        model_key = canonical_donor_model_key(self.model_key)
        observation_time = _finite_float(self.observation_time, "observation_time")
        seed = _validated_seed(self.seed)

        if any(value <= 0 for value in mu):
            raise ValueError("every donor mu_lambda must be greater than zero")
        if any(value < 0 for value in sigma):
            raise ValueError("every donor sigma_lambda must be nonnegative")
        if any(value < 0 or value >= 1 for value in p_zero):
            raise ValueError("every donor p_zero must lie in [0, 1)")
        if observation_time <= 0:
            raise ValueError("observation_time must be greater than zero")
        if model_key in {"homo", "z2p"} and any(value != 0 for value in sigma):
            raise ValueError(f"model {model_key!r} requires sigma_lambda=0")
        if model_key in {"homo", "dis2p"} and any(value != 0 for value in p_zero):
            raise ValueError(f"model {model_key!r} requires p_zero=0")

        object.__setattr__(self, "donor_sizes", sizes)
        object.__setattr__(self, "mu_lambda", mu)
        object.__setattr__(self, "sigma_lambda", sigma)
        object.__setattr__(self, "p_zero", p_zero)
        object.__setattr__(self, "model_key", model_key)
        object.__setattr__(self, "observation_time", observation_time)
        object.__setattr__(self, "seed", seed)

    @property
    def donor_ids(self) -> tuple[str, ...]:
        """Donor labels in the order used for simulation and parameter arrays."""

        return tuple(label for label, _ in self.donor_sizes)

    @property
    def cells_per_donor(self) -> tuple[int, ...]:
        """Validated cell counts aligned to :attr:`donor_ids`."""

        return tuple(size for _, size in self.donor_sizes)

    @property
    def n_cells(self) -> int:
        """Total number of simulated cells."""

        return int(sum(self.cells_per_donor))


def _mixture_truth(spec: DonorSimulationSpec) -> dict[str, object]:
    cell_counts = np.asarray(spec.cells_per_donor, dtype=float)
    weights = cell_counts / cell_counts.sum()
    mu = np.asarray(spec.mu_lambda, dtype=float)
    sigma = np.asarray(spec.sigma_lambda, dtype=float)
    p_zero = np.asarray(spec.p_zero, dtype=float)
    active_mass = weights * (1.0 - p_zero)
    active_weights = active_mass / active_mass.sum()
    mu_population = float(np.sum(active_weights * mu))
    variance_within = float(np.sum(active_weights * sigma**2))
    variance_between = float(np.sum(active_weights * (mu - mu_population) ** 2))

    donors: dict[str, dict[str, int | float]] = {}
    for index, donor_id in enumerate(spec.donor_ids):
        donors[donor_id] = {
            "n_cells": int(cell_counts[index]),
            "donor_weight": float(weights[index]),
            "active_donor_weight": float(active_weights[index]),
            "mu_lambda": float(mu[index]),
            "sigma_lambda": float(sigma[index]),
            "p_zero": float(p_zero[index]),
        }
    return {
        "model_key": spec.model_key,
        "observation_time": spec.observation_time,
        "seed": spec.seed,
        "n_cells": spec.n_cells,
        "donor_ids": spec.donor_ids,
        "donors": donors,
        "population": {
            "mu_lambda": mu_population,
            "sigma_lambda": float(np.sqrt(max(variance_within + variance_between, 0.0))),
            "p_zero": float(np.sum(weights * p_zero)),
            "variance_within_donor": variance_within,
            "variance_between_donor": variance_between,
            "variance_total": variance_within + variance_between,
        },
    }


def simulate_donor_event_counts(
    spec: DonorSimulationSpec,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Simulate donor-labelled cell counts and return exact generating truths.

    Engaging rates are fixed for ``homo``/``z2p`` and Gamma-distributed with
    the requested mean and standard deviation for ``dis2p``/``hetero3``.
    Structural zeros are sampled independently before Poisson event counts.
    Population truths use cell-count weights and, for rate moments, the active
    donor weights used by the donor-aware inference backend.
    """

    if not isinstance(spec, DonorSimulationSpec):
        raise TypeError("spec must be a DonorSimulationSpec")
    rng = np.random.default_rng(spec.seed)
    frames: list[pd.DataFrame] = []
    for donor_id, n_cells, mu, sigma, p_zero in zip(
        spec.donor_ids,
        spec.cells_per_donor,
        spec.mu_lambda,
        spec.sigma_lambda,
        spec.p_zero,
    ):
        if sigma == 0:
            rates = np.full(n_cells, mu, dtype=float)
        else:
            variance = sigma**2
            rates = rng.gamma(shape=mu**2 / variance, scale=variance / mu, size=n_cells)
        counts = rng.poisson(rates * spec.observation_time).astype(int)
        if p_zero > 0:
            structural_zero = rng.random(n_cells) < p_zero
            counts[structural_zero] = 0
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": [f"{donor_id}_cell_{index + 1}" for index in range(n_cells)],
                    "donor_id": donor_id,
                    "count": counts,
                }
            )
        )
    return pd.concat(frames, ignore_index=True), _mixture_truth(spec)


def _posterior(source: Any) -> tuple[xr.Dataset, Any]:
    """Return a posterior and the object carrying optional result metadata."""

    if isinstance(source, xr.Dataset):
        posterior = source
        metadata_source = source
    else:
        idata = getattr(source, "idata", source)
        posterior = getattr(idata, "posterior", None)
        metadata_source = source
    if not isinstance(posterior, xr.Dataset):
        raise ValueError("source must be InferenceData, InferenceResult, or a posterior Dataset")
    if "chain" not in posterior.dims or "draw" not in posterior.dims:
        raise ValueError("posterior variables must contain chain and draw dimensions")
    return posterior, metadata_source


def _first_variable(posterior: xr.Dataset, aliases: Sequence[str]) -> str | None:
    return next((name for name in aliases if name in posterior), None)


def _paired_draw_index(posterior: xr.Dataset) -> pd.DataFrame:
    n_chain = int(posterior.sizes["chain"])
    n_draw = int(posterior.sizes["draw"])
    return pd.DataFrame(
        {
            "chain": np.repeat(np.asarray(posterior.coords["chain"]), n_draw),
            "draw": np.tile(np.asarray(posterior.coords["draw"]), n_chain),
        }
    )


def _subsample_draw_pairs(
    frame: pd.DataFrame,
    max_draws: int | None,
    *,
    random_seed: int,
) -> pd.DataFrame:
    if max_draws is None:
        return frame.reset_index(drop=True)
    if isinstance(max_draws, (bool, np.bool_)) or not isinstance(max_draws, (int, np.integer)):
        raise ValueError("max_draws must be a positive integer or None")
    limit = int(max_draws)
    if limit <= 0:
        raise ValueError("max_draws must be a positive integer or None")
    draw_keys = frame[["chain", "draw"]].drop_duplicates()
    if len(draw_keys) <= limit:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(_validated_seed(random_seed, "random_seed"))
    positions = np.sort(rng.choice(len(draw_keys), size=limit, replace=False))
    selected = draw_keys.iloc[positions]
    return frame.merge(
        selected,
        on=["chain", "draw"],
        how="inner",
        validate="many_to_one",
    ).reset_index(drop=True)


def _model_metadata(metadata_source: Any, model_key: str | None) -> str | None:
    inferred = getattr(metadata_source, "model_key", None)
    if model_key is None and inferred is None:
        return None
    requested = inferred if model_key is None else model_key
    canonical = canonical_donor_model_key(requested)
    if inferred is not None and canonical_donor_model_key(inferred) != canonical:
        raise ValueError("model_key conflicts with the InferenceResult model_key")
    return canonical


def population_posterior_frame(
    source: Any,
    *,
    condition: str | None = None,
    model_key: str | None = None,
    max_draws: int | None = None,
    random_seed: int = 307,
) -> pd.DataFrame:
    """Extract paired population draws from InferenceData or InferenceResult.

    The returned ``chain`` and ``draw`` columns identify the posterior particle
    shared by every parameter in a row.  ``phi_0_population`` is exposed as
    ``p_zero_population``.  Optional subsampling selects whole particle rows.
    """

    posterior, metadata_source = _posterior(source)
    frame = _paired_draw_index(posterior)
    variables: list[str] = []
    for public_name, aliases in _POPULATION_VARIABLES:
        stored_name = _first_variable(posterior, aliases)
        if stored_name is None:
            continue
        variable = posterior[stored_name]
        if set(variable.dims) != {"chain", "draw"}:
            raise ValueError(f"{stored_name} must be scalar for every chain and draw")
        frame[public_name] = variable.transpose("chain", "draw").values.reshape(-1)
        variables.append(public_name)
    if "mu_lambda_population" not in variables:
        raise ValueError("posterior is missing mu_lambda_population")
    finite = np.isfinite(frame[variables].to_numpy(dtype=float)).all(axis=1)
    frame = frame.loc[finite]
    frame = _subsample_draw_pairs(frame, max_draws, random_seed=random_seed)
    canonical_model = _model_metadata(metadata_source, model_key)
    if canonical_model is not None:
        frame.insert(0, "model_key", canonical_model)
    if condition is not None:
        frame.insert(0, "condition", str(condition))
    return frame.reset_index(drop=True)


def _donor_dimension(variable: xr.DataArray, name: str) -> str:
    extra = [dimension for dimension in variable.dims if dimension not in {"chain", "draw"}]
    if len(extra) != 1:
        raise ValueError(f"{name} must contain exactly one donor dimension")
    return extra[0]


def _resolved_donor_labels(
    coordinates: np.ndarray,
    donor_labels: Sequence[object] | Mapping[object, object] | None,
) -> list[str]:
    if donor_labels is None:
        return [str(value) for value in coordinates]
    if isinstance(donor_labels, Mapping):
        labels: list[str] = []
        for coordinate in coordinates:
            if coordinate not in donor_labels:
                raise ValueError("donor_labels mapping does not cover every donor coordinate")
            labels.append(str(donor_labels[coordinate]))
    else:
        labels = [str(value) for value in donor_labels]
        if len(labels) != len(coordinates):
            raise ValueError(
                f"expected {len(coordinates)} donor labels, received {len(labels)}"
            )
    if any(not label.strip() for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("donor labels must be nonempty and unique")
    return labels


def donor_posterior_frame(
    source: Any,
    *,
    condition: str | None = None,
    model_key: str | None = None,
    donor_labels: Sequence[object] | Mapping[object, object] | None = None,
    max_draws: int | None = None,
    random_seed: int = 307,
) -> pd.DataFrame:
    """Extract donor draws while retaining within-particle parameter pairing.

    Donor labels default to posterior coordinates.  For an InferenceResult,
    stored ``donor_labels`` are used automatically.  Backend
    ``phi_0_donor`` values are exposed as ``p_zero_donor``.
    """

    posterior, metadata_source = _posterior(source)
    mu_name = _first_variable(posterior, ("mu_lambda_donor",))
    if mu_name is None:
        raise ValueError("posterior is missing mu_lambda_donor")
    donor_dimension = _donor_dimension(posterior[mu_name], mu_name)
    donor_coordinates = np.asarray(posterior[mu_name].coords[donor_dimension])
    n_chain = int(posterior.sizes["chain"])
    n_draw = int(posterior.sizes["draw"])
    n_donor = int(posterior.sizes[donor_dimension])
    if donor_labels is None:
        inferred_labels = getattr(metadata_source, "donor_labels", None)
        if inferred_labels is not None and len(inferred_labels) > 0:
            donor_labels = inferred_labels
    resolved_labels = _resolved_donor_labels(donor_coordinates, donor_labels)

    frame = pd.DataFrame(
        {
            "chain": np.repeat(np.asarray(posterior.coords["chain"]), n_draw * n_donor),
            "draw": np.tile(np.repeat(np.asarray(posterior.coords["draw"]), n_donor), n_chain),
            "donor_coordinate": np.tile(donor_coordinates, n_chain * n_draw),
            "donor_id": np.tile(resolved_labels, n_chain * n_draw),
        }
    )
    variables: list[str] = []
    for public_name, aliases in _DONOR_VARIABLES:
        stored_name = _first_variable(posterior, aliases)
        if stored_name is None:
            continue
        variable = posterior[stored_name]
        variable_donor_dimension = _donor_dimension(variable, stored_name)
        if variable_donor_dimension != donor_dimension:
            raise ValueError(f"{stored_name} uses a different donor dimension")
        if int(variable.sizes[variable_donor_dimension]) != n_donor:
            raise ValueError(f"{stored_name} uses a different number of donors")
        frame[public_name] = variable.transpose(
            "chain", "draw", variable_donor_dimension
        ).values.reshape(-1)
        variables.append(public_name)
    finite = np.isfinite(frame[variables].to_numpy(dtype=float)).all(axis=1)
    frame = frame.loc[finite]
    frame = _subsample_draw_pairs(frame, max_draws, random_seed=random_seed)
    canonical_model = _model_metadata(metadata_source, model_key)
    if canonical_model is not None:
        frame.insert(0, "model_key", canonical_model)
    if condition is not None:
        frame.insert(0, "condition", str(condition))
    return frame.reset_index(drop=True)


def _normalised_donor_weights(
    values: Sequence[float] | Mapping[object, float] | np.ndarray,
    *,
    coordinates: np.ndarray,
) -> np.ndarray:
    if isinstance(values, Mapping):
        try:
            raw = [values[coordinate] for coordinate in coordinates]
        except KeyError as exc:
            raise ValueError(
                "donor_weights mapping does not cover every donor coordinate"
            ) from exc
        if len(values) != len(coordinates):
            raise ValueError("donor_weights mapping must contain exactly the posterior donors")
        weights = np.asarray(raw, dtype=float)
    else:
        weights = np.asarray(values, dtype=float)
    if weights.shape != (len(coordinates),):
        raise ValueError(f"expected {len(coordinates)} donor weights or cell counts")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("donor weights or cell counts must be finite and positive")
    return weights / weights.sum()


def _p_zero_donor(posterior: xr.Dataset) -> xr.DataArray | None:
    name = _first_variable(posterior, ("p_zero_donor", "phi_0_donor"))
    return None if name is None else posterior[name]


def _mixture_moments(
    posterior: xr.Dataset,
    donor_weights: np.ndarray,
    *,
    donor_dimension: str,
    require_sigma: bool,
) -> xr.Dataset:
    mu = posterior["mu_lambda_donor"]
    sigma = posterior.get("sigma_lambda_donor")
    if sigma is None:
        if require_sigma:
            raise ValueError("posterior is missing sigma_lambda_donor")
        sigma = xr.zeros_like(mu)
    elif _donor_dimension(sigma, "sigma_lambda_donor") != donor_dimension:
        raise ValueError("donor parameter arrays use different donor dimensions")

    weight = xr.DataArray(
        donor_weights,
        dims=(donor_dimension,),
        coords={donor_dimension: mu.coords[donor_dimension]},
    )
    p_zero = _p_zero_donor(posterior)
    if p_zero is None:
        active_weight = weight.broadcast_like(mu)
        p_zero_population = xr.zeros_like(mu.isel({donor_dimension: 0}, drop=True))
    else:
        if _donor_dimension(p_zero, p_zero.name or "p_zero_donor") != donor_dimension:
            raise ValueError("p_zero_donor uses a different donor dimension")
        active_mass = weight * (1.0 - p_zero)
        active_total = active_mass.sum(donor_dimension)
        if np.any(np.asarray(active_total) <= 0):
            raise ValueError("every posterior draw must retain positive active-cell mass")
        active_weight = active_mass / active_total
        p_zero_population = (weight * p_zero).sum(donor_dimension)

    mu_population = (active_weight * mu).sum(donor_dimension)
    variance_within = (active_weight * sigma**2).sum(donor_dimension)
    variance_between = (active_weight * (mu - mu_population) ** 2).sum(donor_dimension)
    variance_total = variance_within + variance_between
    sigma_population = np.sqrt(variance_total.clip(min=0.0))
    fraction_within = xr.where(variance_total > 0, variance_within / variance_total, np.nan)
    fraction_between = xr.where(variance_total > 0, variance_between / variance_total, np.nan)
    return xr.Dataset(
        {
            "donor_weight": weight,
            "active_donor_weight": active_weight,
            "mu_lambda_population": mu_population,
            "sigma_lambda_population": sigma_population,
            "p_zero_population": p_zero_population,
            "variance_within_donor": variance_within,
            "variance_between_donor": variance_between,
            "variance_total": variance_total,
            "fraction_within_donor": fraction_within,
            "fraction_between_donor": fraction_between,
        }
    )


def population_variance_decomposition(
    source: Any,
    donor_weights: Sequence[float] | Mapping[object, float] | np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> xr.Dataset:
    r"""Split population rate variance into within- and between-donor terms.

    Raw donor weights are normally observed cell counts.  In zero-inflated
    models the rate moments use active weights
    :math:`w_d(1-p_{0,d}) / \sum_j w_j(1-p_{0,j})`.  Every draw satisfies
    ``variance_total == variance_within_donor + variance_between_donor`` and
    ``sigma_lambda_population**2 == variance_total`` up to floating-point
    precision.  Saved backend population moments are checked within
    ``tolerance`` when available.
    """

    posterior, _ = _posterior(source)
    if "mu_lambda_donor" not in posterior:
        raise ValueError("posterior is missing mu_lambda_donor")
    mu = posterior["mu_lambda_donor"]
    donor_dimension = _donor_dimension(mu, "mu_lambda_donor")
    coordinates = np.asarray(mu.coords[donor_dimension])
    weights = _normalised_donor_weights(donor_weights, coordinates=coordinates)
    tolerance = _finite_float(tolerance, "tolerance")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    result = _mixture_moments(
        posterior,
        weights,
        donor_dimension=donor_dimension,
        require_sigma=True,
    )

    checks = {
        "mu_lambda_population": "mu_lambda_population",
        "sigma_lambda_population": "sigma_lambda_population",
    }
    p_zero_name = _first_variable(posterior, ("p_zero_population", "phi_0_population"))
    if p_zero_name is not None:
        checks[p_zero_name] = "p_zero_population"
    for saved_name, reconstructed_name in checks.items():
        if saved_name not in posterior:
            continue
        difference = np.abs(posterior[saved_name] - result[reconstructed_name])
        maximum = float(difference.max().to_numpy())
        if not np.isfinite(maximum) or maximum > tolerance:
            raise ValueError(
                f"reconstructed {saved_name} differs from the saved posterior "
                f"by {maximum:.3g} (tolerance {tolerance:g})"
            )
    return result


def leave_one_donor_out_moments(
    source: Any,
    donor_weights: Sequence[float] | Mapping[object, float] | np.ndarray,
    *,
    donor_labels: Sequence[object] | Mapping[object, object] | None = None,
) -> xr.Dataset:
    """Recompute population mixture moments after excluding each donor.

    This is a posterior sensitivity calculation, **not a model refit**.  It
    reuses every fitted donor parameter draw, removes one donor from the
    mixture, renormalizes the supplied cell weights, and recomputes population
    moments.  It therefore answers how fitted mixture summaries depend on each
    donor; it does not include the posterior changes that refitting would
    induce.
    """

    posterior, metadata_source = _posterior(source)
    if "mu_lambda_donor" not in posterior:
        raise ValueError("posterior is missing mu_lambda_donor")
    mu = posterior["mu_lambda_donor"]
    donor_dimension = _donor_dimension(mu, "mu_lambda_donor")
    coordinates = np.asarray(mu.coords[donor_dimension])
    if len(coordinates) < 2:
        raise ValueError("leave-one-donor-out moments require at least two donors")
    weights = _normalised_donor_weights(donor_weights, coordinates=coordinates)
    if donor_labels is None:
        inferred_labels = getattr(metadata_source, "donor_labels", None)
        if inferred_labels is not None and len(inferred_labels) > 0:
            donor_labels = inferred_labels
    labels = _resolved_donor_labels(coordinates, donor_labels)

    datasets: list[xr.Dataset] = []
    for donor_index, label in enumerate(labels):
        retained = weights.copy()
        retained[donor_index] = 0.0
        retained /= retained.sum()
        moments = _mixture_moments(
            posterior,
            retained,
            donor_dimension=donor_dimension,
            require_sigma=False,
        ).expand_dims(excluded_donor=[label])
        datasets.append(moments)
    result = xr.concat(datasets, dim="excluded_donor")
    result = result.assign_coords(
        donor_id=(donor_dimension, labels),
    )
    result.attrs["interpretation"] = (
        "Posterior mixture recomputation with one fitted donor excluded; no refitting."
    )
    return result


ContrastScale = Literal["absolute", "percent_of_control_mean"]


def _finite_sample_matrix(samples: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(samples, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"{name} samples must be a one- or two-dimensional array")
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    if not len(matrix):
        raise ValueError(f"{name} samples contain no finite rows")
    return matrix


def cartesian_contrast_draws(
    treatment_samples: Sequence[float] | np.ndarray,
    control_samples: Sequence[float] | np.ndarray,
    *,
    scale: ContrastScale = "absolute",
    control_mean: Sequence[float] | np.ndarray | None = None,
    max_exact_pairs: int = 500_000,
    approximate_pairs: int = 100_000,
    random_seed: int = 307,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compare independent posteriors using all or sampled Cartesian pairs.

    When ``len(treatment) * len(control)`` exceeds ``max_exact_pairs``, indices
    are sampled independently and uniformly using ``random_seed``.  Percentage
    contrasts divide by the fixed control posterior mean, never by individual
    control particles.
    """

    treatment = _finite_sample_matrix(treatment_samples, "treatment")
    control = _finite_sample_matrix(control_samples, "control")
    if treatment.shape[1] != control.shape[1]:
        raise ValueError("treatment and control samples must contain the same parameters")
    for value, name in (
        (max_exact_pairs, "max_exact_pairs"),
        (approximate_pairs, "approximate_pairs"),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    max_exact_pairs = int(max_exact_pairs)
    approximate_pairs = int(approximate_pairs)
    seed = _validated_seed(random_seed, "random_seed")

    if scale == "percent_of_control_mean":
        denominator = (
            control.mean(axis=0)
            if control_mean is None
            else np.asarray(control_mean, dtype=float)
        )
        if denominator.shape != (control.shape[1],):
            raise ValueError("control_mean must contain one value per parameter")
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ValueError("control posterior means must be finite and positive")
    elif scale == "absolute":
        denominator = None
    else:
        raise ValueError(f"unknown contrast scale: {scale!r}")

    possible_pairs = int(len(treatment) * len(control))
    if possible_pairs <= max_exact_pairs:
        differences = (treatment[:, None, :] - control[None, :, :]).reshape(
            possible_pairs, treatment.shape[1]
        )
        exact = True
    else:
        rng = np.random.default_rng(seed)
        treatment_index = rng.integers(0, len(treatment), size=approximate_pairs)
        control_index = rng.integers(0, len(control), size=approximate_pairs)
        differences = treatment[treatment_index] - control[control_index]
        exact = False
    values = differences if denominator is None else 100.0 * differences / denominator
    return values, {
        "scale": scale,
        "exact_cartesian": exact,
        "possible_pairs": possible_pairs,
        "returned_pairs": int(len(values)),
        "control_mean": None if denominator is None else denominator.copy(),
    }


def _required_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError("missing required column(s): " + ", ".join(missing))


def condition_contrast_frame(
    posterior_draws: pd.DataFrame,
    *,
    treatment: str,
    control: str,
    parameter_columns: Sequence[str] | None = None,
    group_columns: Sequence[str] | None = None,
    scale: ContrastScale = "absolute",
    max_exact_pairs: int = 500_000,
    approximate_pairs: int = 100_000,
    random_seed: int = 307,
) -> pd.DataFrame:
    """Compare independently fitted conditions in extracted posterior tables.

    Use :func:`population_posterior_frame` or :func:`donor_posterior_frame`
    with ``condition=...``, concatenate the resulting frames, then pass them
    here.  If ``group_columns`` is omitted, model keys and donor identifiers
    present in the table are retained as comparison groups.  Pass an explicit
    empty sequence only when pooling those identities is scientifically
    intended.  Chain/draw labels are intentionally *not* paired across
    independently fitted conditions.
    """

    if not isinstance(posterior_draws, pd.DataFrame) or posterior_draws.empty:
        raise ValueError("posterior_draws must be a nonempty DataFrame")
    _required_columns(posterior_draws, ("condition",))
    treatment = str(treatment)
    control = str(control)
    if treatment == control:
        raise ValueError("treatment and control conditions must be different")
    available = set(posterior_draws["condition"].astype(str))
    missing_conditions = {treatment, control}.difference(available)
    if missing_conditions:
        raise ValueError("unknown condition(s): " + ", ".join(sorted(missing_conditions)))

    groups = (
        tuple(
            column
            for column in ("model_key", "donor_id")
            if column in posterior_draws
        )
        if group_columns is None
        else tuple(group_columns)
    )
    _required_columns(posterior_draws, groups)
    if parameter_columns is None:
        candidates = [
            public_name
            for public_name, _ in (*_POPULATION_VARIABLES, *_DONOR_VARIABLES)
            if public_name in posterior_draws
        ]
        parameters = tuple(candidates)
    else:
        parameters = tuple(str(column) for column in parameter_columns)
    if not parameters:
        raise ValueError("no posterior parameter columns were selected")
    if len(parameters) != len(set(parameters)):
        raise ValueError("parameter_columns must not contain duplicates")
    _required_columns(posterior_draws, parameters)

    selected = posterior_draws.loc[
        posterior_draws["condition"].astype(str).isin((control, treatment))
    ].copy()
    if groups:
        treatment_groups = set(
            selected.loc[selected["condition"].astype(str) == treatment, list(groups)]
            .itertuples(index=False, name=None)
        )
        control_groups = set(
            selected.loc[selected["condition"].astype(str) == control, list(groups)]
            .itertuples(index=False, name=None)
        )
        if treatment_groups != control_groups:
            raise ValueError("treatment and control must contain the same comparison groups")
        group_keys: list[tuple[object, ...]] = sorted(
            control_groups,
            key=lambda row: tuple(map(str, row)),
        )
    else:
        group_keys = [()]

    prefix = "delta_" if scale == "absolute" else "percent_delta_"
    value_columns = [prefix + parameter for parameter in parameters]
    frames: list[pd.DataFrame] = []
    for group_index, group_key in enumerate(group_keys):
        group = selected
        for column, value in zip(groups, group_key):
            group = group.loc[group[column] == value]
        treatment_values = group.loc[
            group["condition"].astype(str) == treatment,
            list(parameters),
        ].to_numpy(dtype=float)
        control_values = group.loc[
            group["condition"].astype(str) == control,
            list(parameters),
        ].to_numpy(dtype=float)
        values, metadata = cartesian_contrast_draws(
            treatment_values,
            control_values,
            scale=scale,
            max_exact_pairs=max_exact_pairs,
            approximate_pairs=approximate_pairs,
            random_seed=random_seed + group_index,
        )
        frame = pd.DataFrame(values, columns=value_columns)
        for column, value in reversed(list(zip(groups, group_key))):
            frame.insert(0, column, value)
        frame.insert(0, "control", control)
        frame.insert(0, "treatment", treatment)
        frame["exact_cartesian"] = bool(metadata["exact_cartesian"])
        frame["possible_pairs"] = int(metadata["possible_pairs"])
        frame["returned_pairs"] = int(metadata["returned_pairs"])
        if metadata["control_mean"] is not None:
            for parameter, denominator in zip(
                parameters,
                np.asarray(metadata["control_mean"], dtype=float),
            ):
                frame[f"control_mean_{parameter}"] = denominator
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_contrast_draws(
    contrasts: pd.DataFrame,
    *,
    group_columns: Sequence[str] | None = None,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Summarize contrast columns with HDIs and sign probabilities.

    When ``group_columns`` is omitted, treatment, control, model, and donor
    identity columns present in the contrast table are retained.  Supply an
    explicit empty sequence to pool all rows.
    """

    if not isinstance(contrasts, pd.DataFrame) or contrasts.empty:
        raise ValueError("contrasts must be a nonempty DataFrame")
    probability = _finite_float(hdi_prob, "hdi_prob")
    if not 0 < probability < 1:
        raise ValueError("hdi_prob must lie strictly between zero and one")
    groups = (
        tuple(
            column
            for column in ("treatment", "control", "model_key", "donor_id")
            if column in contrasts
        )
        if group_columns is None
        else tuple(group_columns)
    )
    _required_columns(contrasts, groups)
    value_columns = [
        column
        for column in contrasts
        if column.startswith("delta_") or column.startswith("percent_delta_")
    ]
    if not value_columns:
        raise ValueError("contrast table contains no contrast columns")

    if groups:
        grouper: str | list[str] = groups[0] if len(groups) == 1 else list(groups)
        grouped: Any = contrasts.groupby(grouper, sort=False, dropna=False)
    else:
        grouped = [((), contrasts)]
    rows: list[dict[str, object]] = []
    for group_key, frame in grouped:
        if groups:
            keys = (group_key,) if len(groups) == 1 else tuple(group_key)
            identity = dict(zip(groups, keys))
        else:
            identity = {}
        for parameter in value_columns:
            values = pd.to_numeric(frame[parameter], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            interval = np.asarray(az.hdi(values, hdi_prob=probability), dtype=float)
            rows.append(
                {
                    **identity,
                    "parameter": parameter,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "hdi_lower": float(interval[0]),
                    "hdi_upper": float(interval[1]),
                    "probability_above_zero": float(np.mean(values > 0)),
                    "probability_below_zero": float(np.mean(values < 0)),
                    "n_draws": int(len(values)),
                    "hdi_prob": probability,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "DONOR_MODEL_KEYS",
    "ContrastScale",
    "DonorSimulationSpec",
    "canonical_donor_model_key",
    "simulate_donor_event_counts",
    "population_posterior_frame",
    "donor_posterior_frame",
    "population_variance_decomposition",
    "leave_one_donor_out_moments",
    "cartesian_contrast_draws",
    "condition_contrast_frame",
    "summarize_contrast_draws",
]

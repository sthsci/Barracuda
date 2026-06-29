#!/usr/bin/env python3
from __future__ import annotations

import os
import warnings
from ast import literal_eval
from dataclasses import dataclass
from inspect import signature
from typing import Mapping, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import gaussian_kde


@dataclass(frozen=True)
class HistoryData:
    n_cells: int
    n_contact: np.ndarray
    n_lethal: np.ndarray
    cell_idx: np.ndarray
    x_before: np.ndarray
    y_before: np.ndarray
    z: np.ndarray
    event_to_cell: np.ndarray


@dataclass(frozen=True)
class ModelSpec:
    name: str
    heterogeneous: bool
    history_dependent: bool


def prepare_data(df: pd.DataFrame) -> HistoryData:
    if "history" in df:
        histories = [_normalise_history(h) for h in df["history"]]
    else:
        histories = _histories_from_wide_frame(df)

    n_cells = len(histories)

    n_contact = np.asarray([len(h) for h in histories], dtype=int)
    n_lethal = np.asarray([sum(h) for h in histories], dtype=int)

    cell_idx = []
    x_before = []
    y_before = []
    z_obs = []

    for cell_id, history in enumerate(histories):
        x = 0
        y = 0

        for z in history:
            z = int(z)

            if z not in (0, 1):
                raise ValueError("Each history must contain only 0 and 1.")

            cell_idx.append(cell_id)
            x_before.append(x)
            y_before.append(y)
            z_obs.append(z)

            if z == 1:
                y += 1
            else:
                x += 1

    cell_idx = np.asarray(cell_idx, dtype=int)
    n_events = cell_idx.size
    event_to_cell = np.zeros((n_events, n_cells), dtype=float)

    if n_events > 0:
        event_to_cell[np.arange(n_events), cell_idx] = 1.0

    return HistoryData(
        n_cells=n_cells,
        n_contact=n_contact,
        n_lethal=n_lethal,
        cell_idx=cell_idx,
        x_before=np.asarray(x_before, dtype=float),
        y_before=np.asarray(y_before, dtype=float),
        z=np.asarray(z_obs, dtype=int),
        event_to_cell=event_to_cell,
    )


def _normalise_history(history) -> tuple[int, ...]:
    if history is None:
        return ()

    if isinstance(history, str):
        text = history.strip()

        if not text or text.lower() in {"nan", "none"}:
            return ()

        compact = "".join(text.split())

        if len(compact) > 1 and set(compact).issubset({"0", "1"}):
            history = list(compact)
        else:
            try:
                history = literal_eval(text)
            except (SyntaxError, ValueError):
                text = text.strip("[]()")

                if not text:
                    return ()

                if "," in text:
                    history = [part.strip() for part in text.split(",") if part.strip()]
                else:
                    history = list(text)

        if isinstance(history, str):
            text = text.strip("[]()")

            if not text:
                return ()

            if "," in text:
                history = [part.strip() for part in text.split(",") if part.strip()]
            else:
                history = list(text)

    elif np.isscalar(history):
        try:
            if pd.isna(history):
                return ()
        except TypeError:
            pass

    if np.isscalar(history):
        values = [history]
    else:
        values = list(history)

    out = []

    for value in values:
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass

        numeric = float(value)

        if not numeric.is_integer():
            raise ValueError("Each history must contain only 0 and 1.")

        z = int(numeric)

        if z not in (0, 1):
            raise ValueError("Each history must contain only 0 and 1.")

        out.append(z)

    return tuple(out)


def _histories_from_wide_frame(df: pd.DataFrame) -> list[tuple[int, ...]]:
    event_columns = []

    for column in df.columns:
        try:
            event_id = int(str(column))
        except ValueError:
            continue

        event_columns.append((event_id, column))

    if not event_columns:
        raise ValueError(
            "df must contain a 'history' column or wide numeric event columns."
        )

    event_columns = [column for _, column in sorted(event_columns)]

    return [
        _normalise_history(row[event_columns].to_numpy())
        for _, row in df.iterrows()
    ]


def default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("homogeneous_history_independent", False, False),
        ModelSpec("homogeneous_history_dependent", False, True),
        ModelSpec("heterogeneous_history_independent", True, False),
        ModelSpec("heterogeneous_history_dependent", True, True),
    ]


def _resolve_cores(cores=None, *, chains: int | None = None) -> int:
    cpu = os.cpu_count() or 1
    max_cores = cpu if chains is None else min(cpu, int(chains))

    if cores is None:
        return max_cores

    try:
        cores = int(cores)
    except Exception:
        return max_cores

    if cores <= 0:
        return max_cores

    return min(cores, max_cores)


def _pm_function_supports(fn, name: str) -> bool:
    try:
        return name in signature(fn).parameters
    except Exception:
        return False


def _parallel_smc_failed(exc: BaseException) -> bool:
    if isinstance(exc, (EOFError, BrokenPipeError, ConnectionResetError)):
        return True

    if exc.__class__.__name__ == "ParallelSamplingError":
        return True

    message = str(exc)
    return (
        "did not produce any results" in message
        or "No message from samplers" in message
        or "got end of file during message" in message
        or (
            "file lock" in message.lower()
            and "could not be acquired" in message.lower()
        )
        or "filelock._error.Timeout" in message
    )


def _run_pymc_sample_smc(
    *,
    draws: int,
    chains: int,
    cores: int,
    random_seed: Optional[int],
    threshold: float,
    correlation_threshold: float,
    progressbar: bool,
    mp_ctx=None,
    blas_cores=None,
    compute_convergence_checks: bool = True,
):
    kwargs = {
        "draws": int(draws),
        "chains": int(chains),
        "cores": int(cores),
        "random_seed": random_seed,
        "progressbar": progressbar,
        "return_inferencedata": False,
        "threshold": float(threshold),
        "correlation_threshold": float(correlation_threshold),
    }

    if _pm_function_supports(pm.sample_smc, "compute_convergence_checks"):
        kwargs["compute_convergence_checks"] = bool(compute_convergence_checks)

    if _pm_function_supports(pm.sample_smc, "mp_ctx"):
        kwargs["mp_ctx"] = mp_ctx
    elif mp_ctx is not None:
        warnings.warn(
            "This PyMC version does not support sample_smc(mp_ctx=...).",
            RuntimeWarning,
            stacklevel=2,
        )

    if _pm_function_supports(pm.sample_smc, "blas_cores"):
        kwargs["blas_cores"] = blas_cores
    elif blas_cores is not None:
        warnings.warn(
            "This PyMC version does not support sample_smc(blas_cores=...).",
            RuntimeWarning,
            stacklevel=2,
        )

    return pm.sample_smc(**kwargs)


def _sample_prior_predictive(*, draws: int, random_seed: Optional[int]):
    if draws is None or int(draws) <= 0:
        return None

    kwargs = {
        "random_seed": random_seed,
        "return_inferencedata": True,
    }

    if _pm_function_supports(pm.sample_prior_predictive, "draws"):
        kwargs["draws"] = int(draws)
    else:
        kwargs["samples"] = int(draws)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The effect of Potentials on other parameters is ignored during prior predictive sampling.*",
            category=UserWarning,
        )
        return pm.sample_prior_predictive(**kwargs)


def _softplus(x):
    return pt.log1p(pt.exp(-pt.abs(x))) + pt.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + pt.exp(-x))


def _logit(p):
    return pt.log(p) - pt.log1p(-p)


def _bernoulli_logit_logp(z, logit_p):
    return z * (-_softplus(-logit_p)) + (1.0 - z) * (-_softplus(logit_p))


def _hermite_quadrature(n_quad: int):
    nodes, weights = np.polynomial.hermite.hermgauss(int(n_quad))
    log_weights = np.log(weights) - 0.5 * np.log(np.pi)

    return (
        pt.as_tensor_variable(nodes.astype(float)),
        pt.as_tensor_variable(log_weights.astype(float)),
    )


def _gamma_shape_rate_from_mean_sd(mu, sd):
    sd = pt.maximum(sd, 1e-9)
    var = sd**2
    shape = mu**2 / var
    rate = mu / var

    return shape, rate


def _poisson_count_constant(n, duration: float):
    return n * pt.log(float(duration)) - pt.gammaln(n + 1.0)


def _gamma_poisson_logp(n, duration: float, mu_lambda, sigma_lambda):
    shape, rate = _gamma_shape_rate_from_mean_sd(mu_lambda, sigma_lambda)

    return (
        _poisson_count_constant(n, duration)
        + pt.gammaln(shape + n)
        - pt.gammaln(shape)
        + shape * pt.log(rate)
        - (shape + n) * pt.log(rate + float(duration))
    )


def _logit_normal_count_logp(
    n_lethal,
    n_contact,
    mu_eta,
    sigma_eta,
    *,
    n_quad: int,
):
    nodes, log_weights = _hermite_quadrature(n_quad)
    eta = mu_eta + pt.sqrt(2.0) * sigma_eta * nodes

    y = n_lethal[None, :]
    x = (n_contact - n_lethal)[None, :]
    eta = eta[:, None]

    logp = y * (-_softplus(-eta)) + x * (-_softplus(eta))

    return pt.logsumexp(log_weights[:, None] + logp, axis=0)


def _logit_normal_history_logp(
    z,
    x_before,
    y_before,
    event_start,
    event_end,
    mu_eta,
    sigma_eta,
    beta_x,
    beta_y,
    *,
    n_quad: int,
):
    nodes, log_weights = _hermite_quadrature(n_quad)
    eta = mu_eta + pt.sqrt(2.0) * sigma_eta * nodes

    z = z[None, :]
    x_before = x_before[None, :]
    y_before = y_before[None, :]
    eta = eta[:, None]

    logit_p = eta + beta_x * x_before + beta_y * y_before
    event_logp = _bernoulli_logit_logp(z, logit_p)
    cumulative_logp = pt.concatenate(
        [
            pt.zeros((event_logp.shape[0], 1), dtype=event_logp.dtype),
            pt.cumsum(event_logp, axis=1),
        ],
        axis=1,
    )
    cell_logp = cumulative_logp[:, event_end] - cumulative_logp[:, event_start]

    return pt.logsumexp(log_weights[:, None] + cell_logp, axis=0)


def build_model(
    data: HistoryData,
    spec: ModelSpec,
    *,
    duration: float = 1.0,
    lambda_prior_bounds=(-5.0, 2.0),
    sigma_lambda_prior: float = 1.0,
    p0_prior=(1.0, 1.0),
    sigma_eta_prior: float = 1.0,
    beta_prior_sd: float = 1.0,
    n_quad: int = 30,
):
    coords = {
        "cell": np.arange(data.n_cells),
        "event": np.arange(data.z.size),
    }

    n_contact = pt.as_tensor_variable(data.n_contact.astype(float))
    n_lethal = pt.as_tensor_variable(data.n_lethal.astype(float))
    z = pt.as_tensor_variable(data.z.astype(float))
    x_before = pt.as_tensor_variable(data.x_before.astype(float))
    y_before = pt.as_tensor_variable(data.y_before.astype(float))
    event_start = np.concatenate([[0], np.cumsum(data.n_contact[:-1])]).astype("int64")
    event_end = np.cumsum(data.n_contact).astype("int64")
    event_start = pt.as_tensor_variable(event_start)
    event_end = pt.as_tensor_variable(event_end)

    with pm.Model(coords=coords) as model:
        eta_lambda = pm.Uniform(
            "eta_lambda",
            lower=float(lambda_prior_bounds[0]),
            upper=float(lambda_prior_bounds[1]),
        )

        mu_lambda = pm.Deterministic("mu_lambda", 10.0**eta_lambda)

        sigma_lambda = pm.HalfNormal(
            "sigma_lambda",
            sigma=float(sigma_lambda_prior),
        )

        pm.Potential(
            "contact_loglik",
            pt.sum(
                _gamma_poisson_logp(
                    n_contact,
                    duration,
                    mu_lambda,
                    sigma_lambda,
                )
            ),
        )

        p0_centre = pm.Beta(
            "p0_centre",
            alpha=float(p0_prior[0]),
            beta=float(p0_prior[1]),
        )

        mu_eta = pm.Deterministic("mu_eta", _logit(p0_centre))
        pm.Deterministic("mu_p0", _sigmoid(mu_eta))

        if spec.heterogeneous:
            sigma_eta = pm.HalfNormal(
                "sigma_eta",
                sigma=float(sigma_eta_prior),
            )
        else:
            sigma_eta = pm.Deterministic(
                "sigma_eta",
                pt.as_tensor_variable(0.0),
            )

        if spec.history_dependent:
            beta_x = pm.Normal(
                "beta_x",
                mu=0.0,
                sigma=float(beta_prior_sd),
            )

            beta_y = pm.Normal(
                "beta_y",
                mu=0.0,
                sigma=float(beta_prior_sd),
            )
        else:
            beta_x = pm.Deterministic(
                "beta_x",
                pt.as_tensor_variable(0.0),
            )

            beta_y = pm.Deterministic(
                "beta_y",
                pt.as_tensor_variable(0.0),
            )

        pm.Deterministic("odds_ratio_x", pm.math.exp(beta_x))
        pm.Deterministic("odds_ratio_y", pm.math.exp(beta_y))

        if not spec.heterogeneous and not spec.history_dependent:
            logp = (
                n_lethal * (-_softplus(-mu_eta))
                + (n_contact - n_lethal) * (-_softplus(mu_eta))
            )

            pm.Potential("decision_loglik", pt.sum(logp))

        elif not spec.heterogeneous and spec.history_dependent:
            if data.z.size > 0:
                logit_p = mu_eta + beta_x * x_before + beta_y * y_before

                pm.Potential(
                    "decision_loglik",
                    pt.sum(_bernoulli_logit_logp(z, logit_p)),
                )

        elif spec.heterogeneous and not spec.history_dependent:
            pm.Potential(
                "decision_loglik",
                pt.sum(
                    _logit_normal_count_logp(
                        n_lethal,
                        n_contact,
                        mu_eta,
                        sigma_eta,
                        n_quad=n_quad,
                    )
                ),
            )

        elif spec.heterogeneous and spec.history_dependent:
            if data.z.size > 0:
                pm.Potential(
                    "decision_loglik",
                    pt.sum(
                        _logit_normal_history_logp(
                            z,
                            x_before,
                            y_before,
                            event_start,
                            event_end,
                            mu_eta,
                            sigma_eta,
                            beta_x,
                            beta_y,
                            n_quad=n_quad,
                        )
                    ),
                )

    return model


def sample_smc(
    model,
    *,
    draws: int = 2000,
    chains: int = 4,
    cores=None,
    random_seed: Optional[int] = None,
    prior_draws: int = 0,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
    progressbar: bool = True,
    mp_ctx=None,
    blas_cores=None,
    compute_convergence_checks: bool = True,
    retry_sequential: bool = True,
):
    chains = int(chains)
    resolved_cores = _resolve_cores(cores, chains=chains)

    with model:
        try:
            trace = _run_pymc_sample_smc(
                draws=draws,
                chains=chains,
                cores=resolved_cores,
                random_seed=random_seed,
                progressbar=progressbar,
                threshold=threshold,
                correlation_threshold=correlation_threshold,
                mp_ctx=mp_ctx,
                blas_cores=blas_cores,
                compute_convergence_checks=compute_convergence_checks,
            )
        except Exception as exc:
            if not (
                retry_sequential
                and resolved_cores > 1
                and _parallel_smc_failed(exc)
            ):
                raise

            warnings.warn(
                "Parallel PyMC SMC failed. Retrying sequentially with cores=1.",
                RuntimeWarning,
                stacklevel=2,
            )

            trace = _run_pymc_sample_smc(
                draws=draws,
                chains=chains,
                cores=1,
                random_seed=random_seed,
                progressbar=progressbar,
                threshold=threshold,
                correlation_threshold=correlation_threshold,
                mp_ctx=mp_ctx,
                blas_cores=blas_cores,
                compute_convergence_checks=compute_convergence_checks,
            )

        to_idata = getattr(pm, "to_inference_data", None)

        if to_idata is None:
            to_idata = getattr(pm, "to_inferencedata", None)

        if to_idata is None:
            raise RuntimeError("PyMC does not expose to_inference_data.")

        idata = to_idata(trace)
        _store_log_marginal_likelihood(idata, trace, chains=chains)

        prior = _sample_prior_predictive(
            draws=prior_draws,
            random_seed=random_seed,
        )

    if prior is not None:
        idata.extend(prior)

    return idata


def sample_nuts(
    model,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores=None,
    random_seed: Optional[int] = None,
    target_accept: float = 0.9,
    prior_draws: int = 0,
    progressbar: bool = True,
):
    resolved_cores = _resolve_cores(cores, chains=chains)

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=resolved_cores,
            random_seed=random_seed,
            target_accept=target_accept,
            return_inferencedata=True,
            progressbar=progressbar,
        )

        prior = _sample_prior_predictive(
            draws=prior_draws,
            random_seed=random_seed,
        )

    if prior is not None:
        idata.extend(prior)

    return idata


def _store_log_marginal_likelihood(idata, trace, *, chains: int) -> None:
    report = getattr(trace, "report", None)

    if report is None:
        return

    raw = getattr(report, "log_marginal_likelihood", None)

    if raw is None:
        return

    values = _parse_log_marginal_likelihood(raw, chains=chains)

    if values is None:
        return

    try:
        idata.sample_stats["log_marginal_likelihood"] = (("chain",), values)
    except Exception:
        idata.attrs["log_marginal_likelihood"] = values.tolist()


def _parse_log_marginal_likelihood(raw, *, chains: int):
    if not isinstance(raw, (list, tuple, np.ndarray)):
        try:
            return np.full(chains, float(raw), dtype=float)
        except Exception:
            return None

    if all(not isinstance(item, (list, tuple, np.ndarray)) for item in raw):
        arr = np.asarray(raw, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.size:
            return np.full(chains, float(arr[-1]), dtype=float)

        return None

    values = []

    for item in raw:
        arr = np.asarray(item, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.size:
            values.append(float(arr[-1]))

    if not values:
        return None

    values = np.asarray(values, dtype=float)

    if values.size < chains:
        values = np.concatenate(
            [values, np.full(chains - values.size, np.nan)]
        )

    return values[:chains]


def log_evidence(idata) -> float:
    sample_stats = getattr(idata, "sample_stats", None)

    if sample_stats is not None and "log_marginal_likelihood" in sample_stats:
        values = np.asarray(sample_stats["log_marginal_likelihood"]).ravel()
    elif "log_marginal_likelihood" in idata.attrs:
        values = np.asarray(idata.attrs["log_marginal_likelihood"]).ravel()
    else:
        raise RuntimeError("No SMC log marginal likelihood found.")

    values = values[np.isfinite(values)]

    if values.size == 0:
        raise RuntimeError("SMC log marginal likelihood is empty.")

    return float(np.mean(values))


def compare_evidence(idatas: Mapping[str, object]) -> pd.DataFrame:
    rows = []

    for name, idata in idatas.items():
        rows.append(
            {
                "model": name,
                "log_evidence": log_evidence(idata),
            }
        )

    out = pd.DataFrame(rows)
    best = float(out["log_evidence"].max())

    out["log_bf_vs_best"] = out["log_evidence"] - best
    out["log10_bf_vs_best"] = out["log_bf_vs_best"] / np.log(10)
    out["log10_bf_best_over_model"] = -out["log10_bf_vs_best"]

    return out.sort_values("log_evidence", ascending=False).reset_index(drop=True)


def fit_decision_models(
    df: pd.DataFrame,
    *,
    specs: list[ModelSpec] | None = None,
    duration: float = 1.0,
    draws: int = 1000,
    chains: int = 2,
    cores=None,
    random_seed: Optional[int] = None,
    progressbar: bool = True,
    lambda_prior_bounds=(-5.0, 2.0),
    sigma_lambda_prior: float = 1.0,
    p0_prior=(1.0, 1.0),
    sigma_eta_prior: float = 1.0,
    beta_prior_sd: float = 1.0,
    n_quad: int = 30,
    prior_draws: int = 0,
    method: str = "smc",
):
    data = prepare_data(df)

    if specs is None:
        specs = default_model_specs()

    models = {}
    idatas = {}

    for i, spec in enumerate(specs):
        seed = None if random_seed is None else random_seed + i

        model = build_model(
            data,
            spec,
            duration=duration,
            lambda_prior_bounds=lambda_prior_bounds,
            sigma_lambda_prior=sigma_lambda_prior,
            p0_prior=p0_prior,
            sigma_eta_prior=sigma_eta_prior,
            beta_prior_sd=beta_prior_sd,
            n_quad=n_quad,
        )

        if method == "smc":
            idata = sample_smc(
                model,
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seed,
                progressbar=progressbar,
                prior_draws=prior_draws,
            )
        elif method == "nuts":
            idata = sample_nuts(
                model,
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seed,
                progressbar=progressbar,
                prior_draws=prior_draws,
            )
        else:
            raise ValueError("method must be either 'smc' or 'nuts'.")

        models[spec.name] = model
        idatas[spec.name] = idata

    evidence = compare_evidence(idatas) if method == "smc" else None

    return models, idatas, evidence


def posterior_summary(idata, var_names=None):
    if var_names is None:
        var_names = [
            "mu_lambda",
            "sigma_lambda",
            "p0_centre",
            "mu_p0",
            "mu_eta",
            "sigma_eta",
            "beta_x",
            "beta_y",
            "odds_ratio_x",
            "odds_ratio_y",
        ]

    available = [name for name in var_names if name in idata.posterior]

    return az.summary(idata, var_names=available)


def _np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def draw_p0_population(
    idata,
    *,
    n_parameter_draws: int | None = 1000,
    n_population_draws: int = 1000,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)

    if "mu_eta" not in idata.posterior:
        raise ValueError("idata.posterior must contain 'mu_eta'.")

    if "sigma_eta" not in idata.posterior:
        raise ValueError("idata.posterior must contain 'sigma_eta'.")

    mu_eta = np.asarray(idata.posterior["mu_eta"]).ravel()
    sigma_eta = np.asarray(idata.posterior["sigma_eta"]).ravel()

    if n_parameter_draws is not None and n_parameter_draws < mu_eta.size:
        idx = rng.choice(mu_eta.size, size=int(n_parameter_draws), replace=False)
        mu_eta = mu_eta[idx]
        sigma_eta = sigma_eta[idx]

    eta = rng.normal(
        loc=mu_eta[:, None],
        scale=sigma_eta[:, None],
        size=(mu_eta.size, int(n_population_draws)),
    )

    return _np_sigmoid(eta)


def summarise_p0_population(
    idata,
    *,
    n_parameter_draws: int | None = 1000,
    n_population_draws: int = 1000,
    seed: int | None = None,
):
    p0 = draw_p0_population(
        idata,
        n_parameter_draws=n_parameter_draws,
        n_population_draws=n_population_draws,
        seed=seed,
    )

    values = p0.ravel()

    return pd.Series(
        {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.5)),
            "q75": float(np.quantile(values, 0.75)),
            "q975": float(np.quantile(values, 0.975)),
        }
    )


def savage_dickey(idata, var_name: str, ref: float = 0.0) -> dict[str, float]:
    if not hasattr(idata, "prior"):
        raise ValueError("idata must contain prior samples.")

    if var_name not in idata.posterior:
        raise ValueError(f"{var_name!r} not found in posterior.")

    if var_name not in idata.prior:
        raise ValueError(f"{var_name!r} not found in prior.")

    prior = np.asarray(idata.prior[var_name]).ravel()
    posterior = np.asarray(idata.posterior[var_name]).ravel()

    prior = prior[np.isfinite(prior)]
    posterior = posterior[np.isfinite(posterior)]

    if prior.size < 2 or posterior.size < 2:
        raise ValueError("Need at least two prior and posterior samples.")

    if np.std(prior) == 0.0 or np.std(posterior) == 0.0:
        raise ValueError(f"{var_name} has zero sample variance.")

    prior_density = float(gaussian_kde(prior)(ref)[0])
    posterior_density = float(gaussian_kde(posterior)(ref)[0])

    return {
        "parameter": var_name,
        "ref": float(ref),
        "prior_density": prior_density,
        "posterior_density": posterior_density,
        "BF01": _safe_ratio(posterior_density, prior_density),
        "BF10": _safe_ratio(prior_density, posterior_density),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)

    if denominator == 0.0:
        if numerator == 0.0:
            return np.nan

        return np.inf if numerator > 0.0 else -np.inf

    return numerator / denominator


def history_effect_bayes_factors(idata) -> pd.DataFrame:
    rows = []

    if "beta_x" in idata.posterior and "beta_x" in getattr(idata, "prior", {}):
        rows.append(savage_dickey(idata, "beta_x", ref=0.0))

    if "beta_y" in idata.posterior and "beta_y" in getattr(idata, "prior", {}):
        rows.append(savage_dickey(idata, "beta_y", ref=0.0))

    return pd.DataFrame(rows)


def print_fit(idata) -> None:
    print(posterior_summary(idata))

    if hasattr(idata, "prior"):
        bf = history_effect_bayes_factors(idata)

        if len(bf):
            print(bf)

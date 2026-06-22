from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContactKillParams:
    n_cells: int
    mean_lambda: float = 4.0
    sigma_lambda: float = 2.0
    p0: float = 0.25
    sigma_eta: float = 0.0
    beta_x: float = 0.0
    beta_y: float = 0.0
    duration: float = 1.0

    @property
    def mu_eta(self) -> float:
        return float(logit(self.p0))

    @property
    def mu_p0(self) -> float:
        return self.p0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.asarray(p, dtype=float)

    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("p must be strictly between 0 and 1.")

    return np.log(p / (1.0 - p))


def gamma_from_mean_sd(rng, mean, sd, size):
    if mean <= 0.0:
        raise ValueError("mean_lambda must be positive.")

    if sd < 0.0:
        raise ValueError("sigma_lambda must be non-negative.")

    if sd == 0.0:
        return np.full(size, mean, dtype=float)

    shape = mean**2 / sd**2
    scale = sd**2 / mean

    return rng.gamma(shape=shape, scale=scale, size=size)


def eta_from_normal(rng, mu_eta, sigma_eta, size):
    if sigma_eta < 0.0:
        raise ValueError("sigma_eta must be non-negative.")

    if sigma_eta == 0.0:
        return np.full(size, mu_eta, dtype=float)

    return rng.normal(loc=mu_eta, scale=sigma_eta, size=size)


def decision_probability_from_eta(
    eta,
    x_before,
    y_before,
    beta_x=0.0,
    beta_y=0.0,
):
    return sigmoid(eta + beta_x * x_before + beta_y * y_before)


def simulate_contact_kill(
    params: ContactKillParams,
    seed=None,
    return_latent=False,
):
    if params.n_cells <= 0:
        raise ValueError("n_cells must be positive.")

    if params.duration <= 0.0:
        raise ValueError("duration must be positive.")

    rng = np.random.default_rng(seed)

    lambdas = gamma_from_mean_sd(
        rng=rng,
        mean=params.mean_lambda,
        sd=params.sigma_lambda,
        size=params.n_cells,
    )

    etas = eta_from_normal(
        rng=rng,
        mu_eta=params.mu_eta,
        sigma_eta=params.sigma_eta,
        size=params.n_cells,
    )

    p0s = sigmoid(etas)
    rows = []

    for cell_id, (lam, eta, p0) in enumerate(zip(lambdas, etas, p0s)):
        n_contact = int(rng.poisson(params.duration * lam))
        x = 0
        y = 0
        history = []

        for _ in range(n_contact):
            p = decision_probability_from_eta(
                eta=eta,
                x_before=x,
                y_before=y,
                beta_x=params.beta_x,
                beta_y=params.beta_y,
            )

            z = int(rng.random() < p)
            history.append(z)

            if z == 1:
                y += 1
            else:
                x += 1

        row = {
            "cell_id": cell_id,
            "history": tuple(history),
            "n_contact": n_contact,
            "n_nonlethal": x,
            "n_lethal": y,
        }

        if return_latent:
            row["lambda"] = lam
            row["eta"] = eta
            row["p0"] = p0

        rows.append(row)

    return pd.DataFrame(rows)


def expand_histories(df: pd.DataFrame):
    if "history" not in df:
        raise ValueError("df must contain a 'history' column.")

    rows = []

    for fallback_cell_id, row in enumerate(df.itertuples(index=False)):
        x = 0
        y = 0
        cell_id = getattr(row, "cell_id", fallback_cell_id)

        for event_id, z in enumerate(row.history):
            z = int(z)

            if z not in (0, 1):
                raise ValueError("history must contain only 0 and 1.")

            rows.append(
                {
                    "cell_id": cell_id,
                    "event_id": event_id,
                    "x_before": x,
                    "y_before": y,
                    "z": z,
                }
            )

            if z == 1:
                y += 1
            else:
                x += 1

    return pd.DataFrame(rows)


def true_parameter_dict(params: ContactKillParams):
    return {
        "mu_lambda": params.mean_lambda,
        "sigma_lambda": params.sigma_lambda,
        "mu_eta": params.mu_eta,
        "sigma_eta": params.sigma_eta,
        "mu_p0": params.p0,
        "beta_x": params.beta_x,
        "beta_y": params.beta_y,
        "odds_ratio_x": float(np.exp(params.beta_x)),
        "odds_ratio_y": float(np.exp(params.beta_y)),
    }


def sample_population_p0(params: ContactKillParams, size=10000, seed=None):
    rng = np.random.default_rng(seed)

    eta = eta_from_normal(
        rng=rng,
        mu_eta=params.mu_eta,
        sigma_eta=params.sigma_eta,
        size=size,
    )

    return sigmoid(eta)
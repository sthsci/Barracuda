#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=fusion")

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import LogLocator, MultipleLocator, FuncFormatter
import arviz as az
import pandas as pd
import seaborn as sns


plt.rcParams.update(
	{
		"font.family": ["Monaco", "DejaVu Sans Mono", "monospace"],
		"mathtext.fontset": "stix",
		"legend.fontsize": 12,
		"xtick.labelsize": 11,
		"ytick.labelsize": 11,
		"axes.titlesize": 13,
		"axes.labelsize": 12,
	}
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAP_ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHAP_ROOT))

from Simulator.model_contact_simple import simulate_population_simple  # noqa: E402
from Inference import contact_inference_hetero as cih  # type: ignore  # noqa: E402


DistName = str


def _legend_white(ax, loc: str = "best"):
	leg = ax.legend(loc=loc, frameon=True, fontsize=10, edgecolor="black")
	if leg is not None:
		fr = leg.get_frame()
		fr.set_facecolor("white")
		fr.set_alpha(1.0)


def _get_rgba_colors(cmap_name: str, n: int) -> List[Tuple[float, float, float, float]]:
	cmap = plt.colormaps.get_cmap(str(cmap_name))
	vals = np.linspace(0.30, 0.90, int(n), dtype=float)
	arr = np.asarray(cmap(vals), dtype=float)
	if arr.ndim != 2 or arr.shape[1] != 4:
		raise RuntimeError("colormap did not return RGBA values")
	return [tuple(map(float, row)) for row in arr]


def tune_log_xticks(ax, *, num_major: int = 8, decimals: int = 2):
	def fmt(x, _):
		if x <= 0 or not np.isfinite(x):
			return ""
		p = float(np.log10(x))
		if np.isclose(p, round(p), atol=1e-10):
			return fr"$10^{{{int(round(p))}}}$"
		return f"{x:.{int(decimals)}g}"

	ax.set_xscale("log")
	ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=int(num_major)))
	ax.xaxis.set_major_formatter(FuncFormatter(fmt))
	ax.tick_params(axis="x", which="major", labelsize=10, length=6)
	ax.tick_params(axis="x", which="minor", length=3)


def _lognormal_mu_sigma_from_mean_sd(mean: float, sd: float) -> Tuple[float, float]:
	mean = float(mean)
	sd = float(sd)
	if mean <= 0:
		raise ValueError("lognormal mean must be > 0")
	if sd < 0:
		raise ValueError("lognormal sd must be >= 0")
	if sd == 0:
		return float(np.log(mean)), 0.0
	sigma2 = float(np.log1p((sd / mean) ** 2))
	mu = float(np.log(mean) - 0.5 * sigma2)
	return mu, float(np.sqrt(sigma2))


def _gamma_shape_rate_from_mean_sd(mean: float, sd: float) -> Tuple[float, float]:
	mean = float(mean)
	sd = float(sd)
	if mean <= 0:
		raise ValueError("gamma mean must be > 0")
	if sd < 0:
		raise ValueError("gamma sd must be >= 0")
	if sd == 0:
		return np.inf, np.inf
	shape = (mean / sd) ** 2
	rate = mean / (sd * sd)
	return float(shape), float(rate)


def sample_zero_inflated_rates(
	*,
	n_cells: int,
	dist: str,
	mean_pos: float,
	sd_pos: float,
	p_zero: float,
	seed: Optional[int] = None,
) -> np.ndarray:
	"""Sample per-cell constant rates with explicit mass at zero.

	Interpretation:
	- With probability p_zero, lambda_i = 0 (structural non-contact cells).
	- Otherwise, lambda_i ~ Dist(mean_pos, sd_pos) on (0, +inf).

	mean_pos and sd_pos parameterize the *positive* component.
	"""

	rng = np.random.default_rng(seed)
	n = int(n_cells)
	if n <= 0:
		raise ValueError("n_cells must be positive")
	p0 = float(p_zero)
	if not (0.0 <= p0 <= 1.0):
		raise ValueError("p_zero must be in [0, 1]")

	dist = str(dist).lower().strip()
	if dist not in {"gamma", "lognormal", "truncnorm"}:
		raise ValueError("dist must be one of: gamma, lognormal, truncnorm")

	is_zero = rng.random(n) < p0
	out = np.zeros(n, dtype=float)
	n_pos = int(np.sum(~is_zero))
	if n_pos == 0:
		return out

	mean_pos = float(mean_pos)
	sd_pos = float(sd_pos)
	if mean_pos <= 0:
		raise ValueError("mean_pos must be > 0")
	if sd_pos < 0:
		raise ValueError("sd_pos must be >= 0")

	if sd_pos == 0:
		out[~is_zero] = mean_pos
		return out

	if dist == "gamma":
		shape, rate = _gamma_shape_rate_from_mean_sd(mean_pos, sd_pos)
		out[~is_zero] = rng.gamma(shape=shape, scale=1.0 / rate, size=n_pos).astype(float)
	elif dist == "lognormal":
		mu, sigma = _lognormal_mu_sigma_from_mean_sd(mean_pos, sd_pos)
		out[~is_zero] = rng.lognormal(mean=mu, sigma=sigma, size=n_pos).astype(float)
	else:
		vals = rng.normal(loc=mean_pos, scale=sd_pos, size=n_pos).astype(float)
		vals = np.clip(vals, 0.0, None)
		out[~is_zero] = vals

	out[~np.isfinite(out)] = 0.0
	out[out < 0] = 0.0
	return out


def _pooled_dt(dt_list: Any) -> np.ndarray:
	if dt_list is None:
		return np.array([], dtype=float)
	parts: List[np.ndarray] = []
	for dt in dt_list if isinstance(dt_list, (list, tuple, np.ndarray)) else []:
		arr = np.asarray(dt, dtype=float)
		arr = arr[np.isfinite(arr) & (arr > 0)]
		if arr.size:
			parts.append(arr)
	return np.concatenate(parts) if parts else np.array([], dtype=float)


def _extract_posterior_1d(idata: az.InferenceData, var: str) -> np.ndarray:
	post = idata.posterior
	if var not in post:
		return np.array([], dtype=float)
	vals = post[var].stack(sample=("chain", "draw")).values.ravel()
	return vals[np.isfinite(vals)]


def _freq_from_counts(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	counts = np.asarray(counts, dtype=int)
	counts = counts[np.isfinite(counts)]
	if counts.size == 0:
		return np.array([0.0], dtype=float), np.array([0.0], dtype=float)
	max_k = int(np.max(counts))
	if max_k < 0:
		return np.array([0.0], dtype=float), np.array([0.0], dtype=float)
	bc = np.bincount(counts.clip(min=0), minlength=max_k + 1)
	freq = bc.astype(float) / float(counts.size)
	x = np.arange(freq.size, dtype=float)
	return x, freq


def plot_counts_and_dt(
	*,
	sims: Sequence[Dict[str, Any]],
	labels: Sequence[str],
	save_path: str | Path,
	cmap_name: str = "inferno",
	dpi: int = 350,
	show_dt: bool = True,
	dt_xlim: Optional[Tuple[float, float]] = (1e-3, 1e2),
) -> None:
	if len(sims) != len(labels):
		raise ValueError("sims and labels must have the same length")
	colors = _get_rgba_colors(str(cmap_name), len(labels))

	show_dt = bool(show_dt)
	if show_dt:
		fig = plt.figure(figsize=(13.5, 5.4), dpi=int(dpi))
		fig.patch.set_alpha(0.0)
		gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)
		ax_counts = fig.add_subplot(gs[0, 0])
		ax_dt = fig.add_subplot(gs[0, 1])
		ax_counts.set_facecolor("none")
		ax_dt.set_facecolor("none")
	else:
		fig = plt.figure(figsize=(6.8, 5.4), dpi=int(dpi))
		fig.patch.set_alpha(0.0)
		ax_counts = fig.add_subplot(1, 1, 1)
		ax_counts.set_facecolor("none")

	for col, sim, lab in zip(colors, sims, labels):
		counts = np.asarray(sim.get("n_end", []), dtype=int)
		x, freq = _freq_from_counts(counts)
		# ax_counts.plot(x, freq, color=col, linewidth=2.0, marker="o", markersize=3.5, label=lab)
		if counts.size:
			bins = np.arange(int(counts.max()) + 2) - 0.5  # centers bins at integers: 0,1,2,...
			w = np.ones_like(counts, dtype=float) / float(counts.size)
			ax_counts.hist(counts, bins=bins, weights=w, histtype="step", linewidth=2.0, color=col, label=lab)
			ax_counts.hist(counts, bins=bins, weights=w, histtype="stepfilled", alpha=0.15, color=col)

	ax_counts.set_title("Contact number distribution", fontweight="bold")
	ax_counts.set_xlabel("Contacts per cell")
	ax_counts.set_ylabel("Frequency")
	ax_counts.set_xlim(left=-0.5)
	ax_counts.xaxis.set_major_locator(MultipleLocator(1))
	ax_counts.grid(True, alpha=0.30)
	_legend_white(ax_counts, loc="upper right")

	if show_dt:
		# inter contact duration plot
		# Use shared log-spaced bins across all conditions (like the Homo script)
		pooled_dt = (
			np.concatenate([_pooled_dt(sim.get("dt_list", [])) for sim in sims])
			if sims
			else np.array([], dtype=float)
		)
		pooled_dt = np.asarray(pooled_dt, dtype=float)
		pooled_dt = pooled_dt[np.isfinite(pooled_dt) & (pooled_dt > 0)]
		dt_bins = None
		if pooled_dt.size:
			lo = max(float(np.min(pooled_dt)), 1e-12)
			hi = float(np.max(pooled_dt))
			dt_bins = np.logspace(np.log10(lo), np.log10(hi), num=48)

		for col, sim, lab in zip(colors, sims, labels):
			dt = _pooled_dt(sim.get("dt_list", []))
			dt = dt[np.isfinite(dt) & (dt > 0)]
			if dt.size == 0 or dt_bins is None:
				continue
			ax_dt.hist(
				dt,
				bins=dt_bins,
				density=True,
				histtype="stepfilled",
				alpha=0.15,
				color=col,
				linewidth=1.8,
				label=lab,
			)
			ax_dt.hist(
				dt,
				bins=dt_bins,
				density=True,
				histtype="step",
				alpha=1.0,
				color=col,
				linewidth=1.8,
			)
			# Reference (line version):
			# h, edges = np.histogram(dt, bins=dt_bins, density=True)
			# centers = np.sqrt(edges[:-1] * edges[1:])
			# ax_dt.plot(centers, h, color=col, linewidth=2.0, label=lab)

		if dt_xlim is not None:
			ax_dt.set_xlim(float(dt_xlim[0]), float(dt_xlim[1]))
		tune_log_xticks(ax_dt, num_major=7, decimals=2)
		ax_dt.set_title("Inter-contact Δt", fontweight="bold")
		ax_dt.set_xlabel("Δt")
		ax_dt.set_ylabel("Density")
		ax_dt.grid(True, which="both", alpha=0.30)
		_legend_white(ax_dt, loc="best")

	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(save_path.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight", transparent=True)
	plt.close(fig)
	print("Saved:", str(save_path.with_suffix(".png")))


def plot_posteriors(
	idatas: Sequence[Tuple[str, az.InferenceData]],
	*,
	ground_truth: Optional[Dict[str, Dict[str, float]]] = None,
	parameters: Tuple[str, str, str] = ("mean_lambda_pos", "p_zero", "sd_lambda_pos"),
	parameter_display: Optional[Dict[str, str]] = None,
	hdi_prob: float = 0.95,
	sample_size: int = 200000,
	save_path: str | Path = "posteriors",
	cmap_name: str = "inferno",
	font_scale: float = 0.7,
	diagonal_style: str = "hist",
	marginal_style: str = "circle",
	seed: Optional[int] = None,
	dpi: int = 300,
) -> None:
	"""Plot joint posteriors (lower-triangle + diagonals) for multiple conditions.

	This follows the plotting pattern you shared: a GridSpec matrix where the
	upper triangle is hidden, diagonals are marginals, and lower-triangle is
	joint density.
	"""
	sns.set_context("talk", font_scale=float(font_scale))
	cmap = plt.colormaps.get_cmap(str(cmap_name))
	colors = cmap(np.linspace(0.3, 0.9, len(idatas)))
	rng = np.random.default_rng(seed)

	def _posterior_vals(posterior, name: str) -> np.ndarray:
		"""Return flattened posterior samples for `name`, with fallbacks.

		Supports legacy/alternate names used by different inference parameterizations:
		- mean_lambda_pos: mean_pos or 10**eta
		- sd_lambda_pos: sd_pos or 10**eta_sd
		"""
		name = str(name)
		if name in posterior:
			vals = posterior[name].stack(sample=("chain", "draw")).values.ravel()
			vals = np.asarray(vals, dtype=float)
			return vals[np.isfinite(vals)]

		if name == "mean_lambda_pos":
			if "mean_pos" in posterior:
				vals = posterior["mean_pos"].stack(sample=("chain", "draw")).values.ravel()
				vals = np.asarray(vals, dtype=float)
				return vals[np.isfinite(vals)]
			if "eta" in posterior:
				eta = posterior["eta"].stack(sample=("chain", "draw")).values.ravel()
				eta = np.asarray(eta, dtype=float)
				eta = eta[np.isfinite(eta)]
				return np.power(10.0, eta, dtype=float)

		if name == "sd_lambda_pos":
			if "sd_pos" in posterior:
				vals = posterior["sd_pos"].stack(sample=("chain", "draw")).values.ravel()
				vals = np.asarray(vals, dtype=float)
				return vals[np.isfinite(vals)]
			if "eta_sd" in posterior:
				eta = posterior["eta_sd"].stack(sample=("chain", "draw")).values.ravel()
				eta = np.asarray(eta, dtype=float)
				eta = eta[np.isfinite(eta)]
				return np.power(10.0, eta, dtype=float)

		return np.array([], dtype=float)

	if parameter_display is None:
		parameter_display = {
			"mean_lambda_pos": r"$\mu_{\lambda}$",
			"sd_lambda_pos": r"$\sigma_{\lambda}$",
			"p_zero": r"$p_0$",
		}

	label_to_df: Dict[str, pd.DataFrame] = {}
	params = list(parameters)
	for label, idata in idatas:
		posterior = idata.posterior
		df = pd.DataFrame()
		for p in params:
			vals = _posterior_vals(posterior, p)
			if vals.size == 0:
				continue
			if (sample_size is not None) and (vals.size > int(sample_size)):
				idx = rng.choice(vals.size, int(sample_size), replace=False)
				vals = vals[idx]
			df[p] = vals
		if not df.empty:
			df["label"] = label
			label_to_df[label] = df

	def _robust_limits(all_vals: np.ndarray) -> Optional[Tuple[float, float]]:
		all_vals = np.asarray(all_vals, dtype=float)
		all_vals = all_vals[np.isfinite(all_vals)]
		if all_vals.size == 0:
			return None
		# Robust limits to avoid extreme tails dominating axis ranges
		lo = float(np.quantile(all_vals, 0.005))
		hi = float(np.quantile(all_vals, 0.995))
		if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
			lo = float(np.min(all_vals))
			hi = float(np.max(all_vals))
		if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
			return None
		pad = 0.03 * (hi - lo)
		return lo - pad, hi + pad

	param_xlims: Dict[str, Tuple[float, float]] = {}
	for p in params:
		cols: List[np.ndarray] = []
		for _lab, _df in label_to_df.items():
			if p in _df.columns:
				arr = np.asarray(_df[p].values, dtype=float)
				arr = arr[np.isfinite(arr)]
				if arr.size:
					cols.append(arr)
		if cols:
			lims = _robust_limits(np.concatenate(cols))
			if lims is not None:
				param_xlims[p] = lims

	npar = len(params)
	fig = plt.figure(figsize=(5 * npar, 5 * npar), dpi=int(dpi))
	fig.patch.set_alpha(0.0)
	gs = gridspec.GridSpec(npar, npar, wspace=0.35, hspace=0.35)
	gaxes = np.empty((npar, npar), dtype=object)

	for irow, rowpar in enumerate(params):
		for icol, colpar in enumerate(params):
			ax = plt.subplot(gs[irow, icol])
			gaxes[irow, icol] = ax
			ax.set_facecolor("none")

			if icol > irow:
				ax.axis("off")
				continue

			for color, (label, df) in zip(colors, label_to_df.items()):
				if icol == irow:
					if rowpar not in df.columns:
						continue
					vals = df[rowpar].dropna().values
					if vals.size == 0:
						continue
					if diagonal_style == "kde":
						sns.kdeplot(
							vals,
							ax=ax,
							fill=True,
							color=color,
							alpha=0.2,
							linewidth=1.5,
							label=label if irow == 0 else None,
						)
					elif diagonal_style == "hist":
						sns.histplot(
							vals,
							bins=30,
							stat="density",
							kde=False,
							ax=ax,
							color=color,
							alpha=0.18,
							element="step",
							fill=True,
						)
						sns.histplot(
							vals,
							bins=30,
							stat="density",
							kde=False,
							ax=ax,
							color=color,
							alpha=1.0,
							element="step",
							fill=False,
							linewidth=1.8,
							label=label if irow == 0 else None,
						)

					try:
						lo, hi = az.hdi(vals, hdi_prob=float(hdi_prob))
						ax.axvspan(float(lo), float(hi), color=color, alpha=0.1, linewidth=0)
					except Exception:
						pass

					if ground_truth is not None and label in ground_truth and rowpar in ground_truth[label]:
						ax.axvline(float(ground_truth[label][rowpar]), color=color, linestyle="-", linewidth=1.8)

					ax.grid(alpha=0.2)
				else:
					if (colpar not in df.columns) or (rowpar not in df.columns):
						continue
					if marginal_style == "circle":
						sns.kdeplot(
							x=df[colpar],
							y=df[rowpar],
							ax=ax,
							fill=False,
							color=color,
							alpha=0.6,
							levels=7,
							linewidths=1.0,
						)
					elif marginal_style == "pixel":
						sns.histplot(
							x=df[colpar],
							y=df[rowpar],
							bins=60,
							pthresh=0.01,
							cmap=str(cmap_name),
							cbar=False,
							ax=ax,
						)

			if icol != irow:
				ax.grid(alpha=0.3)
				for color, (label, _df) in zip(colors, label_to_df.items()):
					if ground_truth is not None and label in ground_truth:
						gt = ground_truth[label]
						if colpar in gt and rowpar in gt:
							ax.scatter(
								gt[colpar],
								gt[rowpar],
								marker="*",
								color=color,
								s=80,
								linewidths=2.0,
								zorder=1000,
							)

			if icol == irow:
				ax.set_xlabel(parameter_display.get(rowpar, rowpar))
				ax.set_ylabel("Density")
			else:
				ax.set_xlabel(parameter_display.get(colpar, colpar))
				ax.set_ylabel(parameter_display.get(rowpar, rowpar))

			# Make x-axis consistent within each column (based on the x-parameter)
			xpar = rowpar if icol == irow else colpar
			if xpar in param_xlims:
				ax.set_xlim(*param_xlims[xpar])

			ax.margins(0.05)

	handles, labels = gaxes[0, 0].get_legend_handles_labels()
	if handles:
		fig.legend(
			handles,
			labels,
			loc="center",
			bbox_to_anchor=(0.85, 0.85),
			frameon=True,
			edgecolor="black",
			fontsize=10,
		)

	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	fig.savefig(save_path.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight", transparent=True)
	plt.close(fig)
	print("Saved joint posterior plot:", str(save_path.with_suffix(".png")))


def _parse_float_list(s: str) -> List[float]:
	items = [x.strip() for x in str(s).split(",") if x.strip()]
	return [float(x) for x in items]


def _parse_pair(s: str, *, name: str) -> Tuple[float, float]:
	vals = _parse_float_list(s)
	if len(vals) != 2:
		raise ValueError(f"{name} must be two comma-separated floats (e.g. -5,2)")
	return float(vals[0]), float(vals[1])


def main(argv: Optional[Iterable[str]] = None) -> None:
	p = argparse.ArgumentParser(
		description=(
			"Simulate a heterogeneous, zero-inflated constant-rate contact model and infer (mean, sd, p0) via PyMC. "
			"Supports running 3 parameter-combinations for side-by-side comparison."
		)
	)
	p.add_argument("--n_cells", type=int, default=800)
	p.add_argument("--T", type=float, default=40.0)
	p.add_argument("--seed", type=int, default=None)

	# Distributions
	# - Simulation distribution controls how lambda+ is generated.
	# - Inference distribution controls which family the model assumes.
	# Backwards compatible flags:
	# - --dist_SIMUL (old name)
	# - --infer_dist (old name)
	# - --dist (alias for simulation dist)
	p.add_argument(
		"--dist_simul",
		"--dist_SIMUL",
		"--dist",
		dest="dist_simul",
		type=str,
		choices=["gamma", "lognormal", "truncnorm"],
		default="gamma",
		help="Distribution used to simulate the positive component (lambda+)",
	)
	p.add_argument(
		"--dist_infer",
		"--infer_dist",
		dest="dist_infer",
		type=str,
		choices=["auto", "gamma", "lognormal", "truncnorm"],
		default="auto",
		help="Distribution assumed by inference (auto = match simulation)",
	)
    
	# Single-condition parameters (kept for backwards-compat)
	p.add_argument("--mean", type=float, default=0.05, help="True mean of positive λ component")
	p.add_argument("--sd", type=float, default=0.03, help="True sd of positive λ component")
	p.add_argument("--p0", type=float, default=0.25, help="True zero-inflated proportion")
	# Multi-condition parameters (3 conditions like the Homo script)
	p.add_argument("--means", type=str, default="0.02, 0.05, 0.08", help="Comma-separated means for 3 conditions")
	p.add_argument("--sds", type=str, default="0.05, 0.03, 0", help="Comma-separated sds for 3 conditions")
	p.add_argument("--p0s", type=str, default="0, 0.25, 0.1", help="Comma-separated p0s for 3 conditions")

	p.add_argument("--infer_mode", type=str, choices=["counts", "both"], default="counts")

	p.add_argument("--posterior_samples", type=int, default=7000)
	p.add_argument("--posterior_tune", type=int, default=3000)
	p.add_argument("--posterior_chains", type=int, default=4)
	p.add_argument(
		"--cores",
		type=int,
		default=0,
		help="CPU processes for PyMC sampling (0 = all available cores; effective use is min(chains, cores))",
	)
	p.add_argument("--posterior_target", type=float, default=0.98)

	p.add_argument("--dpi", type=int, default=400)
	p.add_argument("--posterior_plot_samples", type=int, default=6000)
	p.add_argument("--dt_xlim", type=str, default="1e-3,1e2")
	p.add_argument("--cmap", type=str, default="inferno")
	# Priors (log10-uniform bounds for mean/sd of positive-rate component, and Beta(alpha,beta) for p_zero)
	p.add_argument("--mean_prior_bounds", type=str, default="-5,2", help="log10 bounds for mean(lambda+) prior: low,high")
	p.add_argument("--sd_prior_bounds", type=str, default="-5,2", help="log10 bounds for sd(lambda+) prior: low,high")
	p.add_argument("--p_prior", type=str, default="1,5", help="Beta prior for p_zero: alpha,beta")
	args = p.parse_args(list(argv) if argv is not None else None)

	dist_simul = str(args.dist_simul).strip().lower()
	dist_infer = str(args.dist_infer).strip().lower()
	infer_dist = dist_simul if dist_infer in {"", "auto", "match"} else dist_infer

	dt_xlim = None
	if str(args.dt_xlim).strip():
		a, b = [float(x.strip()) for x in str(args.dt_xlim).split(",")]
		dt_xlim = (a, b)

	# Determine whether we run 1 condition or 3
	means_list: List[float]
	sds_list: List[float]
	p0s_list: List[float]
	if str(args.means).strip() or str(args.sds).strip() or str(args.p0s).strip():
		means_list = _parse_float_list(str(args.means))
		sds_list = _parse_float_list(str(args.sds))
		p0s_list = _parse_float_list(str(args.p0s))
		if not (len(means_list) == len(sds_list) == len(p0s_list) == 3):
			raise ValueError("--means/--sds/--p0s must each provide exactly 3 comma-separated values")
	else:
		means_list = [float(args.mean)]
		sds_list = [float(args.sd)]
		p0s_list = [float(args.p0)]

	mean_prior_bounds = _parse_pair(str(args.mean_prior_bounds), name="--mean_prior_bounds")
	sd_prior_bounds = _parse_pair(str(args.sd_prior_bounds), name="--sd_prior_bounds")
	p_prior_bounds = _parse_pair(str(args.p_prior), name="--p_prior")

	sims: List[Dict[str, Any]] = []
	idatas: List[Tuple[str, az.InferenceData]] = []
	ground_truth: Dict[str, Dict[str, float]] = {}

	cores = None if int(args.cores) == 0 else int(args.cores)
	for j, (m, s, p0) in enumerate(zip(means_list, sds_list, p0s_list)):
		label = fr"$\mu={m:g}$, $\sigma={s:g}$, $p_0={p0:g}$"
		rates = sample_zero_inflated_rates(
			n_cells=int(args.n_cells),
			dist=str(dist_simul),
			mean_pos=float(m),
			sd_pos=float(s),
			p_zero=float(p0),
			seed=(int(args.seed) + 2026 + 1000 * j) if args.seed is not None else None,
		)

		sim = simulate_population_simple(
			n_cells=int(args.n_cells),
			T=float(args.T),
			rates=rates,
			seed=(int(args.seed) + 999 + 1000 * j) if args.seed is not None else None,
		)
		sim["lambda_dist"] = str(dist_simul)
		sim["infer_dist"] = str(infer_dist)
		sim["lambda_mean_pos"] = float(m)
		sim["lambda_sd_pos"] = float(s)
		sim["lambda_p0"] = float(p0)
		sims.append(sim)

		counts = np.asarray(sim.get("n_end", []), dtype=int)
		dt_list = sim.get("dt_list", [])
		if str(args.infer_mode) == "counts":
			idata = cih.inference_counts_hetero(
				kills_per_cell=counts,
				obs_time=float(args.T),
				dist=infer_dist,  # type: ignore[arg-type]
				draws=int(args.posterior_samples),
				tune=int(args.posterior_tune),
				chains=int(args.posterior_chains),
				cores=cores,
				target_accept=float(args.posterior_target),
				mean_prior_bounds=mean_prior_bounds,
				sd_prior_bounds=sd_prior_bounds,
				p_prior_bounds=p_prior_bounds,
			)
		else:
			idata = cih.inference_both_hetero(
				kills_per_cell=counts,
				dt_list=dt_list,
				obs_time=float(args.T),
				dist=infer_dist,  # type: ignore[arg-type]
				draws=int(args.posterior_samples),
				tune=int(args.posterior_tune),
				chains=int(args.posterior_chains),
				cores=cores,
				target_accept=float(args.posterior_target),
				mean_prior_bounds=mean_prior_bounds,
				sd_prior_bounds=sd_prior_bounds,
				p_prior_bounds=p_prior_bounds,
			)

		idatas.append((label, idata))
		ground_truth[label] = {"mean_lambda_pos": float(m), "sd_lambda_pos": float(s), "p_zero": float(p0)}

	base_dir = Path(f"results_{dist_simul}__infer_{infer_dist}")
	# base_dir = Path(f"results_hetero/{dist_simul}__infer_{infer_dist}/{args.n_cells}__{args.T}__{args.infer_mode}")
		# Keep old naming for single runs
		# base_dir = Path(
		# 	f"results_hetero/{dist_simul}__infer_{infer_dist}/{args.n_cells}__{args.T}__m{means_list[0]}__s{sds_list[0]}__p0{p0s_list[0]}__{args.infer_mode}"
		# )

	plot_counts_and_dt(
		sims=sims,
		labels=[lab for lab, _ in idatas],
		save_path=base_dir / f"{args.n_cells}_{args.infer_mode}_simu",
		cmap_name=str(args.cmap),
		dpi=int(args.dpi),
		show_dt=(str(args.infer_mode) == "both"),
		dt_xlim=dt_xlim,
	)
	plot_posteriors(
		idatas,
		ground_truth=ground_truth,
		parameters=("mean_lambda_pos", "sd_lambda_pos", "p_zero"),
		save_path=base_dir / f"{args.n_cells}_{args.infer_mode}_posterior",
		cmap_name=str(args.cmap),
		sample_size=int(args.posterior_plot_samples),
		dpi=int(args.dpi),
		seed=int(args.seed) if args.seed is not None else None,
		diagonal_style="hist",
		marginal_style="circle",
	)


if __name__ == "__main__":
	main()

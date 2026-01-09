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

from Simulator import model_contact_I as model  # noqa: E402
from Inference import contact_inference as cih  # type: ignore  # noqa: E402




# Plotting helper functions

def _legend_white(ax, loc: str = "best"):
    leg = ax.legend(loc=loc, frameon=True, fontsize=10, edgecolor="black")
    if leg is not None:
        fr = leg.get_frame()
        fr.set_facecolor("white")
        fr.set_alpha(1.0)

def _get_rgba_colors(cmap_name: str, n: int) -> List[Tuple[float, float, float, float]]:
	cmap = plt.colormaps.get_cmap(str(cmap_name))
	vals = np.linspace(0.35, 0.75, int(n), dtype=float)
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
		if counts.size:
			bins = np.arange(int(counts.max()) + 2) - 0.5
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
	sns.set_context("talk", font_scale=float(font_scale))
	cmap = plt.colormaps.get_cmap(str(cmap_name))
	colors = cmap(np.linspace(0.3, 0.9, len(idatas)))
	rng = np.random.default_rng(seed)

	def _posterior_vals(posterior, name: str) -> np.ndarray:
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



def _legend_white(ax, loc: str = "best"):
	leg = ax.legend(loc=loc, frameon=True, fontsize=10, edgecolor="black")
	if leg is not None:
		fr = leg.get_frame()
		fr.set_facecolor("white")
		fr.set_alpha(1.0)


def _get_rgba_colors(cmap_name: str, n: int) -> List[Tuple[float, float, float, float]]:
	cmap = plt.colormaps.get_cmap(str(cmap_name))
	vals = np.linspace(0.1, 0.9, int(n), dtype=float)
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
	size = [14, 6],
	sims: Sequence[Dict[str, Any]],
	labels: Sequence[str],
	save_path: str | Path,
	cmap_name: str = "inferno",
	dpi: int = 350,
	# show_dt: bool = True,
	counts_max: Optional[int] = None,
	dt_xlim: Optional[Tuple[float, float]] = (1e-3, 1e2),
) -> None:
	if len(sims) != len(labels):
		raise ValueError("sims and labels must have the same length")
	colors = _get_rgba_colors(str(cmap_name), len(labels))

	# show_dt = bool(show_dt)
	# if show_dt:
	# 	fig = plt.figure(figsize=(13.5, 5.4), dpi=int(dpi))
	# 	fig.patch.set_alpha(0.0)
	# 	gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)
	# 	ax_counts = fig.add_subplot(gs[0, 0])
	# 	ax_dt = fig.add_subplot(gs[0, 1])
	# 	ax_counts.set_facecolor("none")
	# 	ax_dt.set_facecolor("none")
	# else:
	# 	fig = plt.figure(figsize=(6.8, 5.4), dpi=int(dpi))
	# 	fig.patch.set_alpha(0.0)
	# 	ax_counts = fig.add_subplot(1, 1, 1)
	# 	ax_counts.set_facecolor("none")
	

	fig = plt.figure(figsize=(size[0], size[1]), dpi=int(dpi))
	fig.patch.set_alpha(0.0)
	gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.2)
	ax_counts = fig.add_subplot(gs[0, 0])
	ax_dt = fig.add_subplot(gs[0, 1])
	ax_counts.set_facecolor("none")
	ax_dt.set_facecolor("none")

	for col, sim, lab in zip(colors, sims, labels):
		counts = np.asarray(sim.get("n_contacts", []), dtype=int)
		x, freq = _freq_from_counts(counts)
		if counts.size:
			bins = np.arange(int(counts.max()) + 2) - 0.5
			w = np.ones_like(counts, dtype=float) / float(counts.size)
			ax_counts.hist(counts, bins=bins, weights=w, histtype="step", linewidth=2.0, color=col, label=lab)
			ax_counts.hist(counts, bins=bins, weights=w, histtype="stepfilled", alpha=0.15, color=col)

	ax_counts.set_title("Contact number distribution", fontweight="bold")
	ax_counts.set_xlabel("Contacts per cell")
	ax_counts.set_ylabel("Frequency")
	if counts_max is not None:
		ax_counts.set_xlim(right=float(counts_max))
	ax_counts.set_xlim(left=-0.5)
	ax_counts.xaxis.set_major_locator(MultipleLocator(1))
	ax_counts.grid(True, alpha=0.30)
	_legend_white(ax_counts, loc="upper right")

	# if show_dt:
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
		dt_bins = np.logspace(np.log10(lo), np.log10(hi), num=22)
		for col, sim, lab in zip(colors, sims, labels):
			dt = _pooled_dt(sim.get("dt_list", []))
			dt = dt[np.isfinite(dt) & (dt > 0)]
			if dt.size == 0 or dt_bins is None:
				continue
			ax_dt.hist(dt, bins=dt_bins, density=True, histtype="stepfilled", alpha=0.15, color=col, linewidth=1.8,)
			ax_dt.hist(dt, bins=dt_bins, density=True, histtype="step", alpha=1.0, color=col, linewidth=1.8, label=lab,)
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
	sns.set_context("talk", font_scale=float(font_scale))
	cmap = plt.colormaps.get_cmap(str(cmap_name))
	colors = cmap(np.linspace(0.3, 0.9, len(idatas)))
	rng = np.random.default_rng(seed)

	def _posterior_vals(posterior, name: str) -> np.ndarray:
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
			"mu_lambda": r"$\mu_{\lambda}$",
			"sigma_lambda": r"$\sigma_{\lambda}$",
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


class scenario:
    def __init__(
        self, name: str,
        mode: str,
        n_cell: float,
        T: float,
        obs_mode: str,
        T_sd: float = None,
        mu_lambda: float = 0.05,
        sd_lambda: float = None,
        p0_lambda: float = None,
        dis_mode: str = None,
        ):
        self.name = name
        self.mode = mode
        self.n_cell = n_cell
        self.T = T
        self.T_sd = T_sd
        self.obs_mode = obs_mode
        self.mu_lambda = mu_lambda
        self.sd_lambda = sd_lambda
        self.p0_lambda = p0_lambda
        self.dis_mode = dis_mode
        self.auto_convert()
        
    def auto_convert(self):
        assert self.mode in ["homogeneous", "heterogeneous"], "Mode must be either 'homogeneous' or 'heterogeneous'."
        assert self.obs_mode in ["Complete", "Truncated"], "obs_mode must be either 'Complete' or 'Truncated'."
        assert self.n_cell > 0, "n_cell must be positive."
        assert self.T > 0, "T must be positive."
        assert self.mu_lambda is not None, "mu_lambda must be provided."
        if self.mode == "homogeneous":
            # Clear heterogeneity parameters (but keep mu_lambda as the single rate)
            self.sd_lambda = None
            self.p0_lambda = None
            self.dis_mode = None
        if self.mode == "heterogeneous":
            assert self.mu_lambda is not None, "Heterogeneous mode requires mu_lambda."
            assert self.sd_lambda is not None, "Heterogeneous mode requires sd_lambda."
            assert self.p0_lambda is not None, "Heterogeneous mode requires p0_lambda."
            # If sd_lambda is 0, no distribution needed (all rates are identical)
            if self.sd_lambda == 0:
                self.dis_mode = None
            else:
                assert self.dis_mode is not None, "Heterogeneous mode with sd_lambda > 0 requires dis_mode."
        if self.obs_mode == "Truncated":
            assert self.T_sd is not None, "Truncated observation mode requires T_sd."

    def take_results(self, results_dir: dict):
        self.results_dir = results_dir


class Bayes:
    def __init__(
        self,
        name: str,
        idata_dic: dict,
    ):
        self.name = name
        self.idata_dic = idata_dic
    def collect(self, idata_name, idata):
        self.idata_dic[idata_name] = idata
        return self

    def collect_many(self, items: Dict[str, Any]):
        self.idata_dic.update(items)
        return self


def run_all_inference(sc, args) -> Dict[str, Any]:
    """Run all supported inference variants and return a dict of InferenceData."""
    base_kwargs = dict(
        initial_duration=False,  # True if using times_list (includes first waiting time)
        mode=str(args.inference_mode),
        draws=int(args.posterior_samples),
        tune=int(args.posterior_tune),
        chains=int(args.posterior_chains),
        cores=int(args.cores),
        lambda_prior_bounds=(-5.0, 2.0),
    )

    idata_homo = cih.inference_homo(
        sc.results_dir['n_contacts'],
        sc.results_dir['max_time'],
        sc.results_dir['dt_list'],
        **base_kwargs,
    )

    idata_Z2P = cih.inference_Z2P(
        sc.results_dir['n_contacts'],
        sc.results_dir['max_time'],
        sc.results_dir['dt_list'],
        p_prior_bounds=(1.0, 1.0),
        **base_kwargs,
    )

    idata_Dis2P = cih.inference_Dis2P(
        sc.results_dir['n_contacts'],
        sc.results_dir['max_time'],
        sc.results_dir['dt_list'],
        std_prior_factor=1.0,
        **base_kwargs,
    )

    idata_hetero3 = cih.inference_hetero3(
        sc.results_dir['n_contacts'],
        sc.results_dir['max_time'],
        sc.results_dir['dt_list'],
        p_prior_bounds=(1.0, 1.0),
        std_prior_factor=1.0,
        **base_kwargs,
    )

    return {
        "homo": idata_homo,
        "Z2P": idata_Z2P,
        "Dis2P": idata_Dis2P,
        "hetero3": idata_hetero3,
    }
        

def main(argv: Optional[Iterable[str]] = None) -> None:
    '''
    Input Parameters
    '''
    p = argparse.ArgumentParser(
		description=(
			"Simulate a heterogeneous, zero-inflated constant-rate contact model and infer (mean, sd, p0) via PyMC. "
			"Supports running 3 parameter-combinations for side-by-side comparison."
		)
	)
    ###
    # Synthetics Data Generation###
    p.add_argument("--n_cell", type=float, default=100, help="Number of killer cells in the simulation.")
    p.add_argument("--T", type=float, default=45.0, help="Total simulation time.")
    p.add_argument("--obs_mode", type=str, default="Complete", help="Observation mode: Complete or Truncated.")
    p.add_argument("--T_sd", type=float, default=None, help="Standard deviation of observation time for Truncated mode.")
    p.add_argument("--dis_mode", type=str, default='gamma', help="Distribution mode for heterogeneous lambda.")
    p.add_argument("--seed", type=int, default=2026, help="Random seed for simulation.")
    ###
    # Bayesian Inference###
    p.add_argument("--inference_mode", type=str, default="duration", help="Inference mode: duration or counts.")
    p.add_argument("--posterior_samples", type=int, default=7000)
    p.add_argument("--posterior_tune", type=int, default=3000)
    p.add_argument("--posterior_chains", type=int, default=4)
    p.add_argument("--cores", type=int, default=0, help="CPU processes for PyMC sampling (0 = all available cores; effective use is min(chains, cores))")
    
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.obs_mode == "Truncated":
        assert args.T_sd is not None and args.T_sd > 0, "T_sd must be positive for Truncated observation mode."
    
    
    
    '''Data Synthesis -- Simulation
    '''
    
    Scenarios = [
        scenario(
            name="Scenario 1",
            mode="homogeneous",
            n_cell=args.n_cell,
            T=args.T,
            T_sd=args.T_sd,
            obs_mode=args.obs_mode,
            mu_lambda=0.05
        ),
        scenario(
            name="Scenario 2",
            mode="heterogeneous",
            n_cell=args.n_cell,
            T=args.T,
            T_sd=args.T_sd,
            obs_mode=args.obs_mode,
            mu_lambda=0.03,
            sd_lambda=0.03,
            p0_lambda=0,      
            dis_mode=args.dis_mode,
        ),
        scenario(
            name="Scenario 3",
            mode="heterogeneous",
            n_cell=args.n_cell,
            T=args.T,
            T_sd=args.T_sd,
            obs_mode=args.obs_mode,
            mu_lambda=0.06,
            sd_lambda=0,
            p0_lambda=0.4,
        ),
        scenario(
            name="Scenario 4",
            mode="heterogeneous",
            n_cell=args.n_cell,
            T=args.T,
            T_sd=args.T_sd,
            obs_mode=args.obs_mode, 
            mu_lambda=0.08,
            sd_lambda=0.06,
            p0_lambda=0.2,      
            dis_mode=args.dis_mode,
        )
    ]
    
    for sc in Scenarios:
        print(f"--- Running {sc.name} ---")
        sim_data = model.simulate_Population(
            n_cells=sc.n_cell,
            T=sc.T,
            truncation_noise=sc.T_sd,
            obs_mode=sc.obs_mode,
            mode=sc.mode,
            mu_lambda=sc.mu_lambda,
            sd_lambda=sc.sd_lambda,
            p0_lambda=sc.p0_lambda,
            Dist_mode=sc.dis_mode,
            seed=args.seed,
        )
        sc.take_results(sim_data)
        print(f"Simulation completed for {sc.name}.")
    
    
    
    
    ''' Bayesian Inference 
    '''
    Bay: Dict[str, Dict[str, Any]] = {}
    for sc in Scenarios:
        items = run_all_inference(sc, args)
        bay = Bayes(name=sc.name, idata_dic={}).collect_many(items)
        Bay[sc.name] = bay.idata_dic

    '''
    Visualisation and Comparison 
    '''
    try:
        sims = [sc.results_dir for sc in Scenarios]
        labels = [sc.name for sc in Scenarios]
        base_dir = Path(f"./results_1/")
        base_dir.mkdir(parents=True, exist_ok=True)

        plot_counts_and_dt(
            size=[14,6],
            sims=sims,
            labels=labels,
            save_path=base_dir/"simulation",
            cmap_name="gist_earth",
            dpi=400,
            dt_xlim=(1e-2, 1e2),
            counts_max = 15
        )

        # Generate posterior triangles for each inference model
        model_names = ["homo", "Z2P", "Dis2P", "hetero3"]
        for model_name in model_names:
            idatas: List[Tuple[str, az.InferenceData]] = []
            ground_truth: Dict[str, Dict[str, float]] = {}
            for sc in Scenarios:
                iddict = Bay[sc.name]
                idata = iddict.get(model_name)
                if idata is None:
                    continue
                idatas.append((sc.name, idata))
                gt: Dict[str, float] = {}
                if sc.mu_lambda is not None:
                    gt["mean_lambda_pos"] = float(sc.mu_lambda)
                if sc.sd_lambda is not None:
                    gt["sd_lambda_pos"] = float(sc.sd_lambda)
                if sc.p0_lambda is not None:
                    gt["p_zero"] = float(sc.p0_lambda)
                if gt:
                    ground_truth[sc.name] = gt

            if idatas:
                if model_name == "homo":
                    parameters = ("lambda",)
                    parameter_display = {"lambda": r"$\lambda$"}
                elif model_name == "Z2P":
                    parameters = ("lambda", "p_zero")
                    parameter_display = {"lambda": r"$\lambda$", "p_zero": r"$\phi_0$"}
                elif model_name == "Dis2P":
                    parameters = ("mu_lambda", "sigma_lambda")
                    parameter_display = {
						"mu_lambda": r"$\mu_{\lambda}$",
						"sigma_lambda": r"$\sigma_{\lambda}$",
					}
                    parameter_display
                elif model_name == "hetero3":
                    parameters = ("mu_lambda", "sigma_lambda", "p_zero")
                    parameter_display = {
						"mu_lambda": r"$\mu_{\lambda}$",
						"sigma_lambda": r"$\sigma_{\lambda}$",
						"p_zero": r"$\phi_0$",
					}	
                
                plot_posteriors(
                    idatas,
                    ground_truth=ground_truth if ground_truth else None,
                    parameters=parameters,
                    parameter_display=parameter_display,
                    save_path=base_dir / f"posterior_triangle_{model_name}",
                    cmap_name="gist_earth",
                    sample_size=6000,
                    dpi=400,
                    diagonal_style="hist",
                    marginal_style="circle",
                )
    except Exception as e:
        print("Visualization failed:", str(e))
        

if __name__ == "__main__":
    main()
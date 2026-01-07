
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---- Path setup ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Simulator import model_eods as eods


def _ensure_empty_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)
	for p in path.iterdir():
		if p.is_dir():
			import shutil

			shutil.rmtree(p)
		else:
			p.unlink()


def _plot_heatmap(
	values: np.ndarray,
	*,
	x_ticks,
	y_ticks,
	xlabel: str,
	ylabel: str,
	title: str,
	cmap: str = "inferno",
	cmap_range: tuple[float, float] | None = None,
	vmin: float | None = None,
	vmax: float | None = None,
	save_path: Path,
	dpi: int = 300,
):
	fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=dpi)

	vals = np.asarray(values, dtype=float)
	masked = np.ma.masked_invalid(vals)
	base_cmap = plt.get_cmap(cmap)
	if cmap_range is not None:
		c0, c1 = float(cmap_range[0]), float(cmap_range[1])
		if not (0.0 <= c0 < c1 <= 1.0):
			raise ValueError("cmap_range must satisfy 0.0 <= low < high <= 1.0")
		colors = base_cmap(np.linspace(c0, c1, 256))
		cmap_obj = LinearSegmentedColormap.from_list(
			name=f"{base_cmap.name}_crop_{c0:.2f}_{c1:.2f}",
			colors=colors,
			N=256,
		)
	else:
		cmap_obj = base_cmap
	if hasattr(cmap_obj, "copy"):
		cmap_obj = cmap_obj.copy()
	cmap_obj.set_bad(alpha=0.0)

	im = ax.imshow(
		masked,
		origin="lower",
		aspect="auto",
		cmap=cmap_obj,
		vmin=vmin,
		vmax=vmax,
	)
	cbar = fig.colorbar(im, ax=ax)

	ax.set_xticks(np.arange(len(x_ticks)))
	ax.set_xticklabels([str(x) for x in x_ticks])
	ax.set_yticks(np.arange(len(y_ticks)))
	ax.set_yticklabels([str(y) for y in y_ticks])

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.grid(False)

	fig.tight_layout()
	fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", transparent=True)
	plt.close(fig)


def run_grid():
	outdir = Path(__file__).resolve().parent / "plot_comp"
	_ensure_empty_dir(outdir)

	# ---- Simulation settings ----
	n_killers = 1000
	max_event = 20
	rate0 = 0.5
	max_time = 60
	target_multiplier = 4.0
	target_rate_floor = 0.0
	seed0 = 66

	# ---- Parameter grid ----
	K0_values = [1, 2, 3, 4, 5, 6, 7,  8, 9,  10]
	p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	# ---- Heatmap color range tuning ----
	targets_vmin, targets_vmax = None, None
	kills_vmin, kills_vmax = None, None
	ratio_vmin, ratio_vmax = None, None

	# ---- Heatmap colormap cropping ----
	targets_cmap_range = (0, 1)
	kills_cmap_range = (0, 1)
	ratio_cmap_range = (0, 1)

	# ---- Model settings ----
	p_mode = "stochastic"
	p_stochastic_mode = "constant"
	alpha = 0.0
	beta = 0.0

	rmat = eods.rate_matrix(mode="constant", rate0=rate0, max_event=max_event)

	targets_remaining = np.zeros((len(K0_values), len(p_values)), dtype=float)
	targets_remaining_norm = np.zeros((len(K0_values), len(p_values)), dtype=float)
	mean_kills_per_killer = np.zeros((len(K0_values), len(p_values)), dtype=float)
	lethal_over_nonlethal_global = np.zeros((len(K0_values), len(p_values)), dtype=float)
	lethal_over_nonlethal_per_killer = np.zeros((len(K0_values), len(p_values)), dtype=float)
	t_end = np.zeros((len(K0_values), len(p_values)), dtype=float)

	rows = []
	run_idx = 0
	for iK, K0 in enumerate(K0_values):
		for ip, p0 in enumerate(p_values):
			seed = int(seed0) + run_idx
			run_idx += 1

			out = eods.sim_traj_global(
				n_killers=n_killers,
				r_matrix=rmat,
				p_mode=p_mode,
				p_stochastic_mode=p_stochastic_mode,
				p0=float(p0),
				alpha=float(alpha),
				beta=float(beta),
				capacity_mode="homogeneous",
				capacity_kwargs={"K0": int(K0)},
				max_time=float(max_time),
				max_event=int(max_event),
				seed=seed,
				target_multiplier=float(target_multiplier),
				target_rate_floor=float(target_rate_floor),
			)

			n_targets_init = float(out["n_targets_init"])
			targets_remaining[iK, ip] = float(out["targets_remaining"])
			targets_remaining_norm[iK, ip] = (
				float(out["targets_remaining"]) / n_targets_init if n_targets_init > 0 else np.nan
			)
			mean_kills_per_killer[iK, ip] = float(np.mean(out["y"]))
			total_nonlethal = float(np.sum(out["x"]))
			total_lethal = float(np.sum(out["y"]))
			lethal_over_nonlethal_global[iK, ip] = (
				total_lethal / total_nonlethal if total_nonlethal > 0 else np.nan
			)

			xi = np.asarray(out["x"], dtype=float)
			yi = np.asarray(out["y"], dtype=float)
			per_killer_ratio = np.where(xi > 0, yi / xi, np.nan)
			lethal_over_nonlethal_per_killer[iK, ip] = float(np.nanmean(per_killer_ratio))
			t_end[iK, ip] = float(out.get("t_end", np.nan))

			rows.append(
				{
					"K0": int(K0),
					"p0": float(p0),
					"seed": int(seed),
					"targets_init": int(out["n_targets_init"]),
					"targets_remaining": int(out["targets_remaining"]),
					"targets_remaining_norm": float(targets_remaining_norm[iK, ip]),
					"total_nonlethal_contacts": int(np.sum(out["x"])),
					"total_lethal_contacts": int(np.sum(out["y"])),
					"total_kills": int(np.sum(out["y"])),
					"mean_kills_per_killer": float(np.mean(out["y"])),
					"lethal_over_nonlethal_global": float(lethal_over_nonlethal_global[iK, ip]),
					"lethal_over_nonlethal_per_killer": float(
						lethal_over_nonlethal_per_killer[iK, ip]
					),
					"t_end": float(out.get("t_end", np.nan)),
				}
			)

	# ---- Save numeric results ----
	np.savez(
		outdir / "grid_results.npz",
		K0_values=np.asarray(K0_values, dtype=int),
		p_values=np.asarray(p_values, dtype=float),
		targets_remaining=targets_remaining,
		targets_remaining_norm=targets_remaining_norm,
		mean_kills_per_killer=mean_kills_per_killer,
		lethal_over_nonlethal_global=lethal_over_nonlethal_global,
		lethal_over_nonlethal_per_killer=lethal_over_nonlethal_per_killer,
		t_end=t_end,
	)

	# ---- Save CSV ----
	csv_path = outdir / "grid_results.csv"
	with csv_path.open("w", encoding="utf-8") as f:
		f.write(
			"K0,p0,seed,targets_init,targets_remaining,targets_remaining_norm,total_nonlethal_contacts,total_lethal_contacts,total_kills,mean_kills_per_killer,lethal_over_nonlethal_global,lethal_over_nonlethal_per_killer,t_end\n"
		)
		for r in rows:
			f.write(
				f"{r['K0']},{r['p0']},{r['seed']},{r['targets_init']},{r['targets_remaining']},{r['targets_remaining_norm']},{r['total_nonlethal_contacts']},{r['total_lethal_contacts']},{r['total_kills']},{r['mean_kills_per_killer']},{r['lethal_over_nonlethal_global']},{r['lethal_over_nonlethal_per_killer']},{r['t_end']}\n"
			)

	# ---- Plots ----
	_plot_heatmap(
		targets_remaining,
		x_ticks=p_values,
		y_ticks=K0_values,
		xlabel="Kill probability p",
		ylabel="Homogeneous capacity K0",
		title="Final remaining target cells",
		# cmap="YlGnBu",
		cmap_range=targets_cmap_range,
		vmin=targets_vmin,
		vmax=targets_vmax,
		save_path=outdir / "heatmap_targets_remaining.png",
	)

	_plot_heatmap(
		targets_remaining_norm,
		x_ticks=p_values,
		y_ticks=K0_values,
		xlabel="Kill probability p",
		ylabel="Homogeneous capacity K0",
		title="Normalised remaining target cells (remaining / initial)",
		# cmap="YlGnBu",
		cmap_range=targets_cmap_range,
		vmin=None,
		vmax=None,
		save_path=outdir / "heatmap_targets_remaining_normalised.png",
	)

	_plot_heatmap(
		mean_kills_per_killer,
		x_ticks=p_values,
		y_ticks=K0_values,
		xlabel="Kill probability p",
		ylabel="Homogeneous capacity K0",
		title="Mean kills per killer cell (final)",
		# cmap="YlGnBu",
		cmap_range=kills_cmap_range,
		vmin=kills_vmin,
		vmax=kills_vmax,
		save_path=outdir / "heatmap_mean_kills_per_killer.png",
	)

	_plot_heatmap(
		lethal_over_nonlethal_per_killer,
		x_ticks=p_values,
		y_ticks=K0_values,
		xlabel="Kill probability p",
		ylabel="Homogeneous capacity K0",
		title="Lethal/non-lethal contact ratio per killer (mean(y_i/x_i))",
		# cmap="YlGnBu",
		cmap_range=ratio_cmap_range,
		vmin=ratio_vmin,
		vmax=ratio_vmax,
		save_path=outdir / "heatmap_lethal_over_nonlethal.png",
	)

	print("Saved:", str(outdir / "heatmap_targets_remaining.png"))
	print("Saved:", str(outdir / "heatmap_targets_remaining_normalised.png"))
	print("Saved:", str(outdir / "heatmap_mean_kills_per_killer.png"))
	print("Saved:", str(outdir / "heatmap_lethal_over_nonlethal.png"))
	print("Saved:", str(outdir / "grid_results.csv"))
	print("Saved:", str(outdir / "grid_results.npz"))


if __name__ == "__main__":
	run_grid()


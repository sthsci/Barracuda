from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (Orca/) is on sys.path so we can import Simulator/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from Simulator import model_eods as eods
from Simulator import visualisation as vis


def run_simulation():
	n_killers = 1000
	max_event = 15
	outdir = Path(__file__).resolve().parent

	# ---- Parameters ----
	rate0 = 0.5
	seed = 66
	max_time = 100.0
	target_multiplier = 4.0
	target_rate_floor = 0.0

	# Kill probability settings
	p_mode = "deterministic"
	p_stochastic_mode = "constant"
	p0 = 0.30
	alpha = 0.25
	beta = 0.15

	# Capacity settings
	capacity_mode = "homogeneous"
	capacity_kwargs = {"K0": 10}

	rmat = eods.rate_matrix(mode="constant", rate0=rate0, max_event=max_event)

	out = eods.sim_traj_global(
		n_killers=n_killers,
		r_matrix=rmat,
		p_mode=p_mode,
		p_stochastic_mode=p_stochastic_mode,
		p0=p0,
		alpha=alpha,
		beta=beta,
		capacity_mode=capacity_mode,
		capacity_kwargs=capacity_kwargs,
		max_time=max_time,
		max_event=max_event,
		seed=seed,
		target_multiplier=target_multiplier,
		target_rate_floor=target_rate_floor,
	)

	total_kills = int(np.sum(out["y"]))
	print("t_end:", out["t_end"])
	print("targets init:", out["n_targets_init"], "remaining:", out["targets_remaining"])
	print("total kills:", total_kills)
	print("mean kills per killer:", float(np.mean(out["y"])) )

	# --- Visualisations ---
	sc = {
		"p_mode": p_mode,
		"p_stochastic_mode": p_stochastic_mode,
		"p0": p0,
		"alpha": alpha,
		"beta": beta,
		"capacity_mode": capacity_mode,
		"capacity_kwargs": dict(capacity_kwargs),
	}
	label = vis.scenario_label(
		sc,
		n_killers=n_killers,
		n_targets=out["n_targets_init"],
		rate0=rate0,
		show_counts=False,
		show_rate=False,
	)
	color = (0.2, 0.5, 0.8, 1.0)

	vis.decision_map(
		decision_data_group=[(out["decisions_list"], label, color)],
		max_event=max_event,
		save_png=True,
		png_path=str(outdir / "trajectories.png"),
		save_pdf=False,
		dpi=300,
	)

	vis.plot_targets_remaining(
		outs=[out],
		labels=[label],
		colors=[color],
		save_png=True,
		png_path=str(outdir / "targets_remaining.png"),
		save_pdf=False,
		dpi=300,
	)

	vis.plot_targets_remaining_normalised(
		outs=[out],
		labels=[label],
		colors=[color],
		save_png=True,
		png_path=str(outdir / "targets_remaining_normalised.png"),
		save_pdf=False,
		dpi=300,
	)

	vis.plot_killing_capacity_dynamics(
		outs=[out],
		labels=[label],
		colors=[color],
		normalise=True,
		save_png=True,
		png_path=str(outdir / "killing_capacity_dynamics_mean_std.png"),
		save_pdf=False,
		dpi=300,
	)

	vis.plot_killing_capacity_all_cells(
		out,
		color=color,
		normalise=True,
		save_png=True,
		png_path=str(outdir / "killing_capacity_dynamics_all_cells.png"),
		save_pdf=False,
		dpi=300,
	)

	vis.heatmap_contacts_vs_kills(
		out,
		use_total_contacts=True,
		color=color,
		save_png=True,
		png_path=str(outdir / "contacts_vs_kills_heatmap.png"),
		save_pdf=False,
		dpi=300,
	)

	vis.heatmap_contacts_vs_kills(
		out,
		use_total_contacts=False,
		color=color,
		save_png=True,
		png_path=str(outdir / "nonlethal_vs_lethal_heatmap.png"),
		save_pdf=False,
		dpi=300,
	)

	print("Saved:", str(outdir / "trajectories.png"))
	print("Saved:", str(outdir / "targets_remaining.png"))
	print("Saved:", str(outdir / "targets_remaining_normalised.png"))
	print("Saved:", str(outdir / "killing_capacity_dynamics_mean_std.png"))
	print("Saved:", str(outdir / "killing_capacity_dynamics_all_cells.png"))
	print("Saved:", str(outdir / "contacts_vs_kills_heatmap.png"))
	print("Saved:", str(outdir / "nonlethal_vs_lethal_heatmap.png"))


if __name__ == "__main__":
	run_simulation()


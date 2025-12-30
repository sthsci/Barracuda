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
	n_killers = 500
	max_event = 13
	outdir = Path(__file__).resolve().parent / "plots_1"
	outdir.mkdir(parents=True, exist_ok=True)

	# Clean output directory
	import shutil
	for p in outdir.iterdir():
		if p.is_dir():
			shutil.rmtree(p)
		else:
			p.unlink()

	# ---- Parameters ----
	rate0 = 0.5
	seed = 66
	max_time = 40.0
	target_multiplier = 4.0
	target_rate_floor = 0.0

	scenarios = [
		{
			"name": "p0=1",
			"p_mode": "deterministic",
			"p_stochastic_mode": "constant",
			"p0": 1,
			"alpha": 0,
			"beta": 0,
			"capacity_mode": "homogeneous",
			"capacity_kwargs": {"K0": 10},
		},
		{
			"name": "p0=0.5",
			"p_mode": "stochastic",
			"p_stochastic_mode": "constant",
			"p0": 0.5,
			"alpha": 0,
			"beta": 0,
			"capacity_mode": "homogeneous",
			"capacity_kwargs": {"K0": 10},
		},
        {
			"name": "p0=0.1",
			"p_mode": "stochastic",
			"p_stochastic_mode": "constant",
			"p0": 0.1,
			"alpha": 0,
			"beta": 0,
			"capacity_mode": "homogeneous",
			"capacity_kwargs": {"K0": 10},
		}
	]
	palette = vis._pick_colors(len(scenarios), cmap_name="YlGnBu")

	rmat = eods.rate_matrix(mode="constant", rate0=rate0, max_event=max_event)

	outs = []
	labels = []
	colors = []

	for j, sc in enumerate(scenarios):
		tag = f"{j:02d}__{sc['name']}"
		out = eods.sim_traj_global(
			n_killers=n_killers,
			r_matrix=rmat,
			p_mode=sc["p_mode"],
			p_stochastic_mode=sc["p_stochastic_mode"],
			p0=sc["p0"],
			alpha=sc["alpha"],
			beta=sc["beta"],
			capacity_mode=sc["capacity_mode"],
			capacity_kwargs=sc["capacity_kwargs"],
			max_time=max_time,
			max_event=max_event,
			seed=int(seed) + int(j),
			target_multiplier=target_multiplier,
			target_rate_floor=target_rate_floor,
		)

		total_kills = int(np.sum(out["y"]))
		print("---", sc["name"], "---")
		print("t_end:", out["t_end"])
		print("targets init:", out["n_targets_init"], "remaining:", out["targets_remaining"])
		print("total kills:", total_kills)
		print("mean kills per killer:", float(np.mean(out["y"])) )

		label = vis.scenario_label(
			{
				"p_mode": sc["p_mode"],
				"p_stochastic_mode": sc["p_stochastic_mode"],
				"p0": sc["p0"],
				"alpha": sc["alpha"],
				"beta": sc["beta"],
				"capacity_mode": sc["capacity_mode"],
				"capacity_kwargs": dict(sc["capacity_kwargs"]),
			},
			n_killers=n_killers,
			n_targets=out["n_targets_init"],
			rate0=rate0,
			show_counts=False,
			show_rate=False,
		)
		label = f"{sc['name']}"
        # label = f"{sc['name']}\n{label}"
		color = sc.get("color", palette[j % len(palette)])

		outs.append(out)
		labels.append(label)
		colors.append(color)

		# Save heatmaps separately per condition
		vis.heatmap_contacts_vs_kills(
			out,
			use_total_contacts=True,
			color=color,
			match_heatmap_to_marginals=True,
			title=f"{tag} | Contacts vs kills",
			save_png=True,
			png_path=str(outdir / f"contacts_vs_kills_heatmap__{tag}.png"),
			save_pdf=False,
			dpi=300,
		)
		vis.heatmap_contacts_vs_kills(
			out,
			use_total_contacts=False,
			color=color,
			match_heatmap_to_marginals=True,
			title=f"{tag} | Non-lethal vs kills",
			save_png=True,
			png_path=str(outdir / f"nonlethal_vs_lethal_heatmap__{tag}.png"),
			save_pdf=False,
			dpi=300,
		)

	# ---- Decision-making trajectories (all conditions in one frame) ----
	decision_data_group = [(out["decisions_list"], lab, col) for out, lab, col in zip(outs, labels, colors)]
	vis.decision_map(
		decision_data_group=decision_data_group,
		max_event=max_event,
		save_png=True,
		png_path=str(outdir / "decision_trajectories__ALL.png"),
		save_pdf=False,
		dpi=300,
	)

	# ---- Longitudinal plots together (mean ± std) ----
	vis.plot_contacts_per_killer_dynamics(
		outs=outs,
		labels=labels,
		colors=colors,
		save_png=True,
		png_path=str(outdir / "contacts_per_killer_mean_std__ALL.png"),
		save_pdf=False,
		dpi=300,
	)
	vis.plot_kills_per_killer_dynamics(
		outs=outs,
		labels=labels,
		colors=colors,
		save_png=True,
		png_path=str(outdir / "kills_per_killer_mean_std__ALL.png"),
		save_pdf=False,
		dpi=300,
	)

	# ---- Tumour (targets) remaining dynamics ----
	# vis.plot_targets_remaining(
	# 	outs=outs,
	# 	labels=labels,
	# 	colors=colors,
	# 	title="Tumour cells remaining",
	# 	xlabel="Time",
	# 	ylabel="Tumour cells remaining",
	# 	save_png=True,
	# 	png_path=str(outdir / "tumour_cells_remaining__ALL.png"),
	# 	save_pdf=False,
	# 	dpi=300,
	# )

	vis.plot_targets_remaining_normalised(
		outs=outs,
		labels=labels,
		colors=colors,
		title="Tumour cells remaining (normalised)",
		xlabel="Time",
		ylabel="Tumour cells remaining (normalised)",
		save_png=True,
		png_path=str(outdir / "tumour_cells_remaining_normalised__ALL.png"),
		save_pdf=False,
		dpi=300,
	)

	# # ---- Longitudinal plots together (all-cell trajectories) ----
	# import matplotlib.pyplot as plt

	# fig_c, ax_c = plt.subplots(figsize=(9.2, 4.6), dpi=300)
	# for out, color in zip(outs, colors):
	# 	vis.plot_contacts_per_killer_all_cells(out, ax=ax_c, color=color, alpha=0.05, lw=1.0)
	# ax_c.set_title("Contacts per killer cell (all trajectories)")
	# fig_c.tight_layout()
	# fig_c.savefig(str(outdir / "contacts_per_killer_all_cells__ALL.png"), dpi=300, bbox_inches="tight", transparent=True)

	# fig_k, ax_k = plt.subplots(figsize=(9.2, 4.6), dpi=300)
	# for out, color in zip(outs, colors):
	# 	vis.plot_kills_per_killer_all_cells(out, ax=ax_k, color=color, alpha=0.05, lw=1.0)
	# ax_k.set_title("Kills per killer cell (all trajectories)")
	# fig_k.tight_layout()
	# fig_k.savefig(str(outdir / "kills_per_killer_all_cells__ALL.png"), dpi=300, bbox_inches="tight", transparent=True)

	print("Saved:", str(outdir / "decision_trajectories__ALL.png"))
	print("Saved:", str(outdir / "contacts_per_killer_mean_std__ALL.png"))
	print("Saved:", str(outdir / "kills_per_killer_mean_std__ALL.png"))
	print("Saved:", str(outdir / "tumour_cells_remaining__ALL.png"))
	print("Saved:", str(outdir / "tumour_cells_remaining_normalised__ALL.png"))


if __name__ == "__main__":
	run_simulation()


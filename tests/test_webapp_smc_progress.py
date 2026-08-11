from __future__ import annotations

import pymc.smc.sampling as smc_sampling

from section_1.src import inference as backend


def test_pymc_stage_and_beta_are_forwarded_to_the_web_progress_callback() -> None:
    backend._install_smc_progress_bridge()
    events: list[tuple[int, float]] = []
    token = backend._smc_progress_callback.set(
        lambda stage, beta: events.append((stage, beta))
    )
    try:
        with smc_sampling.CustomProgress(disable=True) as progress:
            task_id = progress.add_task("Chain 0", status="Stage: 0 Beta: 0")
            progress.update(task_id, status="Stage: 3 Beta: 0.625")
    finally:
        backend._smc_progress_callback.reset(token)

    assert events == [(3, 0.625)]
